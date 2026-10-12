from __future__ import annotations

import os
import hashlib
import tqdm
import shutil
from pathlib import Path
from typing import Optional

import time
import cv2

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from lightning.fabric import Fabric
from torch.utils.data import DataLoader
import copy

from lib.helpers.save_helper import load_checkpoint
from lib.helpers.decode_helper import extract_dets_from_outputs
from lib.helpers.decode_helper import decode_detections
from lib.helpers.config_helper import Config
from lib.datasets.kitti.kitti_utils import Object3d, Calibration
from lib.datasets.utils import draw_calibrated_box3d
from lib.models.monodpt.depth_predictor.ddn_loss import DDNLoss

from utils import box_ops

def plot_boxes(img: np.array, boxes: list[Object3d], calib: Calibration, mask: Optional[list] = None):
    img_box = img.copy()
    for index, box in enumerate(boxes):
        color = (0, 255, 0) if mask is None or mask[index] else (0, 0, 255)
        img_box = draw_calibrated_box3d(
            img_box,
            box.generate_corners3d(),
            calib,
            color=color,
            thickness=2,
        )
    return img_box

def build_weighted_depth_from_logits(
    depth_logits, depth_num_bins, depth_min, depth_max, *, median_threshold=True
):
    bin_size = 2 * (depth_max - depth_min) / (depth_num_bins * (1 + depth_num_bins))
    bin_indice = torch.arange(depth_num_bins, device=depth_logits.device, dtype=depth_logits.dtype)
    bin_value = (bin_indice + 0.5).pow(2) * bin_size / 2 - bin_size / 8 + depth_min
    bin_value = torch.cat([
        bin_value,
        torch.tensor([depth_max], device=depth_logits.device, dtype=depth_logits.dtype),
    ])

    if median_threshold:
        # This threshold is only for the legacy visualization. The model's
        # own depth predictor uses an ordinary softmax over all depth bins.
        median_values = torch.median(depth_logits, dim=1, keepdim=True).values
        depth_logits = torch.where(depth_logits >= median_values, depth_logits, 0.0)

    depth_probs = F.softmax(depth_logits, dim=1)
    weighted_depth = (depth_probs * bin_value.reshape(1, -1, 1, 1)).sum(dim=1)

    return weighted_depth

def get_depthmap_true(outputs, targets, cfg):

    depth_map_logits = outputs['pred_depth_map_logits']

    mask = targets['mask_2d']

    _, N, H, W = depth_map_logits.shape
    if N > 1:
        depth_map_pred_thresholded = build_weighted_depth_from_logits(
            depth_map_logits,
            depth_num_bins=cfg.model.num_depth_bins,
            depth_min=cfg.model.depth_min,
            depth_max=cfg.model.depth_max,
            median_threshold=True,
        )
        depth_map_pred_softmax = build_weighted_depth_from_logits(
            depth_map_logits,
            depth_num_bins=cfg.model.num_depth_bins,
            depth_min=cfg.model.depth_min,
            depth_max=cfg.model.depth_max,
            median_threshold=False,
        )
        target_reference = depth_map_pred_softmax.unsqueeze(1)
    else:
        depth_map_pred_thresholded = depth_map_logits[:, 0]
        depth_map_pred_softmax = depth_map_logits[:, 0]
        target_reference = depth_map_logits

    num_gt_per_img = [m.sum() for m in mask]
    # (wip) cera un bug qui causato dalle shape hardcodate che rompevano la creazione della depthmap di GT
    if mask.sum() > 0:
        gt_boxes2d = torch.cat([b[m] for b, m in zip(targets['boxes'], mask) if m.any()]) * torch.tensor(
            [W, H, W, H], device=depth_map_logits.device
        )
        gt_boxes2d = box_ops.box_cxcywh_to_xyxy(gt_boxes2d)
        gt_center_depth = torch.cat([b[m] for b, m in zip(targets['depth'], mask) if m.any()]).squeeze(dim=1)
    else:
        gt_boxes2d = torch.empty((0, 4), dtype=torch.float32, device=depth_map_logits.device)
        gt_center_depth = torch.empty((0,), dtype=torch.float32, device=depth_map_logits.device)

    depth_map_true = DDNLoss.build_target_depth_from_3dcenter(
        target_reference, gt_boxes2d, gt_center_depth, num_gt_per_img
    )

    return depth_map_true, depth_map_pred_thresholded, depth_map_pred_softmax

def cmap_mono(mono, depth_min, depth_max):
    if depth_max <= depth_min:
        raise ValueError(f"Invalid depth display range [{depth_min}, {depth_max}]")
    mono = np.nan_to_num(mono, nan=depth_min, posinf=depth_max, neginf=depth_min)
    mono_jet = (np.clip(mono, depth_min, depth_max) - depth_min) / (depth_max - depth_min)
    mono_jet = (mono_jet * 255).astype(np.uint8)
    mono_jet = cv2.applyColorMap(mono_jet, cv2.COLORMAP_JET)

    return mono_jet


def cmap_mono_relative(mono, lower_percentile=2.0, upper_percentile=98.0):
    """Colorize one map by robust per-image contrast; colors are relative."""
    mono = np.asarray(mono, dtype=np.float32)
    finite = mono[np.isfinite(mono)]
    if finite.size == 0:
        return np.zeros((*mono.shape, 3), dtype=np.uint8), 0.0, 1.0
    lower, upper = np.percentile(finite, [lower_percentile, upper_percentile])
    if upper <= lower:
        upper = lower + 1e-6
    scaled = np.clip((np.nan_to_num(mono, nan=lower) - lower) / (upper - lower), 0, 1)
    color = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return color, float(lower), float(upper)


def sample_map_in_box(depth_map, box_xyxy, input_size, inner_fraction=0.5):
    """Return the median map value from the central part of an input-frame box."""
    depth_map = np.asarray(depth_map, dtype=np.float32)
    if depth_map.ndim != 2 or not np.isfinite(depth_map).any():
        return float("nan")
    input_h, input_w = input_size
    x1, y1, x2, y2 = np.asarray(box_xyxy, dtype=np.float32)
    if x2 <= x1 or y2 <= y1:
        return float("nan")

    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    half_w = (x2 - x1) * inner_fraction / 2
    half_h = (y2 - y1) * inner_fraction / 2
    map_h, map_w = depth_map.shape
    mx1 = int(np.floor((center_x - half_w) * map_w / input_w))
    mx2 = int(np.ceil((center_x + half_w) * map_w / input_w))
    my1 = int(np.floor((center_y - half_h) * map_h / input_h))
    my2 = int(np.ceil((center_y + half_h) * map_h / input_h))
    mx1, mx2 = np.clip([mx1, mx2], 0, map_w)
    my1, my2 = np.clip([my1, my2], 0, map_h)
    region = depth_map[my1:my2, mx1:mx2]
    region = region[np.isfinite(region)]
    if region.size == 0:
        px = int(np.clip(center_x * map_w / input_w, 0, map_w - 1))
        py = int(np.clip(center_y * map_h / input_h, 0, map_h - 1))
        return float(depth_map[py, px])
    return float(np.median(region))


def box_iou_xyxy(box_a, box_b):
    box_a = np.asarray(box_a, dtype=np.float32)
    box_b = np.asarray(box_b, dtype=np.float32)
    x1, y1 = np.maximum(box_a[:2], box_b[:2])
    x2, y2 = np.minimum(box_a[2:], box_b[2:])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    return intersection / max(area_a + area_b - intersection, 1e-6)


def draw_depth_readouts(image, lines):
    """Draw compact per-object depth measurements in a dark image overlay."""
    if not lines:
        return image
    image = image.copy()
    visible_lines = lines[:6]
    if len(lines) > len(visible_lines):
        visible_lines.append(f"... {len(lines) - len(visible_lines)} more objects")
    text_lines = ["GT z | DPT inner-box p50 | Mono inner-box p50 | best detection z (IoU)"]
    text_lines.extend(visible_lines)
    line_height = 19
    panel_h = line_height * len(text_lines) + 8
    panel_w = min(image.shape[1], 1040)
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
    image = cv2.addWeighted(overlay, 0.78, image, 0.22, 0)
    for line_index, text in enumerate(text_lines):
        cv2.putText(
            image,
            text,
            (8, 18 + line_index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return image


def label_plot_panel(image, title, size):
    """Resize a BGR panel, then add a title at the final display resolution."""
    image = image.copy()
    height, width = size
    if image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    cv2.rectangle(image, (0, 0), (image.shape[1] - 1, 26), (0, 0, 0), -1)
    cv2.putText(
        image,
        title,
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return image

def prepare_plot_tensor(images, size):
    h, w = size
    image_tensor = []
    for image in images:

        # Resize if needed
        src_h, src_w = image.shape[:2]
        if (src_h != h) or (src_w != w):
            if len(image.shape) == 2:
                image = cv2.resize(image, (w, h), cv2.INTER_NEAREST) # grayscale interp
            else:
                image = cv2.resize(image, (w, h))

        # Convert to RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        assert len(image.shape) == 3

        # convert to torch
        image = torch.from_numpy(image).permute(2,0,1)
        image_tensor.append(image)

    image_tensor = torch.concatenate(image_tensor, dim=1)
    return image_tensor


def prepare_plot_grid(rows, size):
    """Convert BGR/grayscale panels into a two-dimensional RGB tensor grid."""
    tensor_rows = []
    for row in rows:
        tensor_panels = [prepare_plot_tensor([panel], size) for panel in row]
        tensor_rows.append(torch.concatenate(tensor_panels, dim=2))
    return torch.concatenate(tensor_rows, dim=1)

class Tester(object):
    def __init__(self,
                 fabric: Fabric,
                 cfg: Config,
                 model: nn.Module,
                 dataloader: DataLoader,
                 checkpoint_dir: Path,
                 outputs_dir: Path,
                 model_name='monodpt',
                 ):

        self.cfg = cfg
        self.fabric = fabric
        self.model = model
        self.dataloader = dataloader
        self.max_objs = dataloader.dataset.max_objs  # todo  # max objects per images, defined in dataset
        self.class_name = dataloader.dataset.class_name # todo
        self.checkpoint_dir = checkpoint_dir
        # checkpoint_dir =  Path('/media/matte/data/pinim/export_small/logs/worthy-oath-23/checkpoints/')
        print( "==> Test checkpoint dir: {}".format( str( checkpoint_dir ) ) )
        self.outputs_dir = outputs_dir
        self.dataset_type = cfg.dataset.type 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.filename_format = cfg.dataset.filename_format if cfg.dataset.filename_format is not None else ['%06d']

        self.logged_images = 0
        self.images_to_log = 5

    @property
    def rank_zero(self):
        return self.fabric.is_global_zero

    def test(self):
        assert self.cfg.tester.mode in ['single', 'all']

        # test a single checkpoint
        if self.cfg.tester.mode == 'single' or not self.cfg.trainer.save_all:
            resume_str = str(self.cfg.trainer.resume_model) if self.cfg.trainer.resume_model else ""
            if resume_str.endswith('.pth'):
                checkpoint_path = Path(resume_str)
            elif self.cfg.trainer.save_all:
                checkpoint_path = self.checkpoint_dir / "checkpoint_epoch_{}.pth".format(self.cfg.tester.checkpoint)
            else:
                checkpoint_path = Path(self.checkpoint_dir / "checkpoint_best.pth")
                
            print("==> Test checkpoint path: {}".format(str(checkpoint_path)))
            assert checkpoint_path.is_file()

            _, _, _, trainer_state = load_checkpoint(
                fabric=self.fabric,
                model=self.model,
                optimizer=None,
                filename=checkpoint_path,
                logger=None)

            step = trainer_state.global_step if trainer_state else None
            self.inference(step=step)
            self.evaluate()

        # test all checkpoints in the given dir
        elif self.cfg.tester.mode == 'all' and self.cfg.trainer.save_all:
            start_epoch = int(self.cfg.tester.checkpoint)
            checkpoints_list = []
            for _, _, files in os.walk(self.checkpoint_dir):
                for f in files:
                    if f.endswith(".pth") and int(f[17:-4]) >= start_epoch:
                        checkpoints_list.append(self.checkpoint_dir / f)
            checkpoints_list.sort(key=lambda p: p.stat().st_mtime)

            for checkpoint in checkpoints_list:
                _, _, _, trainer_state = load_checkpoint(
                    fabric=self.fabric,
                    model=self.model,
                    optimizer=None,
                    filename=checkpoint,
                    logger=None)

                step = trainer_state.global_step if trainer_state else None
                self.inference(step=step)
                self.evaluate()

    def inference(self, step=None):
        self.logged_images = 0

        torch.set_grad_enabled(False)
        self.model.eval()
        checksum_before = self._model_checksum()
        print(f"==> Evaluating live model checksum: {checksum_before}")
        print(f"==> Prediction output directory: {self.outputs_dir}")

        if self.rank_zero:
            import shutil
            if self.outputs_dir.exists():
                shutil.rmtree(self.outputs_dir)
            self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.fabric.barrier()

        local_results = {}
        model_infer_time = 0.0

        # tqdm only on rank 0 to avoid clutter
        progress_bar = tqdm.tqdm(
            total=len(self.dataloader),
            disable=not self.rank_zero,
            desc="Evaluation Progress"
        )

        # Validation diagnostics accumulators
        diag_val_images = 0
        diag_gt_cars = 0
        diag_removed_by_depth = 0
        diag_rejected_proj = 0
        diag_skipped_encoding = 0

        diag_pred_decoded = 0
        diag_pred_surviving_conf = 0
        diag_pred_written = 0
        diag_pred_invalid_depth = 0
        diag_pred_outside_depth = 0
        diag_pred_invalid_geom = 0
        
        diag_ordered_img_ids = []
        
        # We will keep a representative prediction dump for the first 3 images
        diag_dump_images = {}
        for batch_idx, (inputs, calibs, targets, info) in enumerate(self.dataloader):

            img_sizes = info['img_size']

            start_time = time.time()
            outputs = self.model(inputs, calibs, targets, img_sizes, dn_args=0)
            model_infer_time += time.time() - start_time

            dets = extract_dets_from_outputs(outputs, K=self.max_objs, topk=self.cfg.tester.topk)
            dets = dets.detach().cpu()
            # print(f'DEBUG TESTER - DETS shape: {dets.shape}, first detection: {dets[0, 0, :] if dets.shape[1] > 0 else "none"}')
            # print('Calibs before', calibs[0])
            # print('len', len(calibs))
            # info = {key: val.detach().cpu().numpy() for key, val in info.items()}
            info = {
                key: val.detach().cpu().numpy() if isinstance(val, torch.Tensor) else val 
                for key, val in info.items()
            }
            # print(info.keys())
            
            # accumulate GT stats
            batch_size = len(info['img_id'])
            diag_val_images += batch_size
            diag_gt_cars += int(np.sum(info.get('stat_gt_cars', [0])))
            diag_removed_by_depth += int(np.sum(info.get('stat_removed_by_depth', [0])))
            diag_rejected_proj += int(np.sum(info.get('stat_rejected_proj', [0])))
            diag_skipped_encoding += int(np.sum(info.get('stat_skipped_encoding', [0])))
            for img_id in info['img_id']:
                diag_ordered_img_ids.append(int(img_id))

            diag_pred_decoded += batch_size * dets.shape[1]

            # Get original calibrations (before letterbox adjustments)
            original_calibs = [self.dataloader.dataset.get_calib(index) for index in info['img_id']]

            # calibs from dataloader are P2 tensors that have ALREADY been adjusted for letterbox
            # We need to convert them to Calibration objects for decode_detections
            # Create Calibration objects from the adjusted P2 matrices
            import copy
            calibs_for_decode = []
            for i, calib_orig in enumerate(original_calibs):
                calib_adjusted = copy.deepcopy(calib_orig)
                if 'pad_w' in info:
                    calib_adjusted.apply_image_transform(
                        float(info['resize_scale'][i]),
                        float(info['pad_w'][i]),
                        float(info['pad_h'][i]),
                    )
                calibs_for_decode.append(calib_adjusted)

            cls_mean_size = self.dataloader.dataset.cls_mean_size
            # print('cls_mean_size', cls_mean_size)


            # Decode using RESIZED calibrations (predictions are in resized space)
            decoded = decode_detections(
                dets=dets.numpy(),
                info=info,
                calibs=calibs_for_decode,
                cls_mean_size=cls_mean_size,
                threshold=self.cfg.tester.threshold,
            )

            if self.should_log_images(batch_idx, step):
                self.log_images(
                    outputs,
                    targets,
                    decoded,
                    inputs,
                    original_calibs,
                    calibs_for_decode,
                    info,
                    step,
                    batch_index=0,
                )

            # DEBUG: Check how many detections we got
            total_dets_before = sum(len(decoded[k]) for k in decoded.keys())
            # print(f"DEBUG: Decoded {total_dets_before} detections for {len(decoded)} images (before correction)")

            corrected_decoded= {}
            # print('INFO\n', info)
            if 'resize_scale' in info:
                # print('Correct detections ....')
                for i, img_id in enumerate(info['img_id']):
                    # if img_id not in decoded:
                    #     # print(f"DEBUG: img_id {img_id} not in decoded dict!")
                    #     continue
                    new_preds = []

                    # try:
                    orig_ds = info['orig_ds'][i]
                    # print(orig_ds)
                    # except:
                    #     proble
                    #     orig_ds = 'data'
                    if orig_ds not in corrected_decoded:
                        corrected_decoded[orig_ds] = {}
                    preds = decoded.get(img_id, [])
                    diag_pred_surviving_conf += len(preds)
                    
                    if len(diag_dump_images) < 3 and img_id not in diag_dump_images:
                        diag_dump_images[img_id] = {'preds': [], 'gt': [], 'calib': original_calibs[i]}

                    # print(f"DEBUG: Processing img_id {img_id}: {len(preds)} predictions before correction") 
                    s = info['resize_scale'][i]
                    pad_h = info['pad_h'][i]
                    pad_w = info['pad_w'][i]
                    for p1 in preds:
                        p = p1[:] # Copy to avoid modifying original in-place if needed

                        # 2. Correct 2D Bounding Box
                        # Formula: (Pixel_in_letterbox - Padding) / Scale
                        p[2] = (p[2] - pad_w) / s  # x1
                        p[3] = (p[3] - pad_h) / s  # y1
                        p[4] = (p[4] - pad_w) / s  # x2
                        p[5] = (p[5] - pad_h) / s  # y2
                        
                        # 3. 3D Coordinates (X, Y, Z) and Rotation (ry)
                        # CRITICAL FIX: The network predicts X, Y, Z in physical meters relative to the camera.
                        # Since decode_detections unprojects the points using the appropriately scaled calibration
                        # matrix (calibs_for_decode), the resulting coordinates p[9:12] and p[12] (ry) are ALREADY
                        # perfectly correct in the physical 3D world! 
                        # We must NOT overwrite them using the 2D bounding box center, which ruins the 3D projection!

                        # diagnostics checks
                        depth = np.array(p[11])
                        if not np.isfinite(depth) or np.isnan(depth):
                            diag_pred_invalid_depth += 1
                        if depth > getattr(self.dataloader.dataset, 'depth_threshold', 1000) or depth < 0:
                            diag_pred_outside_depth += 1
                        if not np.all(np.isfinite(p[9:12])) or not np.all(np.isfinite(p[6:9])):
                            diag_pred_invalid_geom += 1

                        new_preds.append(p)
                        if img_id in diag_dump_images:
                            diag_dump_images[img_id]['preds'].append(p)

                    corrected_decoded[orig_ds][img_id] = new_preds
                    # print(f"DEBUG: After correction, img_id {img_id} has {len(new_preds)} predictions")
                    # Instead of local_results.update(corrected_decoded)
                    for ds_name, img_dict in corrected_decoded.items():
                        if ds_name not in local_results:
                            local_results[ds_name] = {}
                        # Merge the current batch's images into the total results
                        local_results[ds_name].update(img_dict)

                # print(f"DEBUG: Total images in local_results: {len(local_results)}")
                total_dets_after = sum(len(local_results[k]) for k in local_results.keys())
                # print(f"DEBUG: Total detections after correction: {total_dets_after}")
                # print('Done.\n')
            else:
                # If no resize, still organize by orig_ds for consistency
                for i, img_id in enumerate(info['img_id']):
                    preds = decoded.get(img_id, [])
                    diag_pred_surviving_conf += len(preds)
                    
                    if len(diag_dump_images) < 3 and img_id not in diag_dump_images:
                        diag_dump_images[img_id] = {'preds': [], 'gt': [], 'calib': original_calibs[i]}

                    for p in preds:
                        depth = np.array(p[11])
                        if not np.isfinite(depth) or np.isnan(depth):
                            diag_pred_invalid_depth += 1
                        if depth > getattr(self.dataloader.dataset, 'depth_threshold', 1000) or depth < 0:
                            diag_pred_outside_depth += 1
                        if not np.all(np.isfinite(p[9:12])) or not np.all(np.isfinite(p[6:9])):
                            diag_pred_invalid_geom += 1
                            
                        if img_id in diag_dump_images:
                            diag_dump_images[img_id]['preds'].append(p)

                    orig_ds = info['orig_ds'][i]
                    if orig_ds not in local_results:
                        local_results[orig_ds] = {}
                    local_results[orig_ds][img_id] = preds

            progress_bar.update()

        progress_bar.close()

        if self.rank_zero:
            for ds_name, img_dict in local_results.items():
                for img_id, preds in img_dict.items():
                    diag_pred_written += len(preds)

            import json
            dump_data = []
            dump_text = ""
            for img_id, data in diag_dump_images.items():
                calib = data['calib']
                
                img_data = {
                    'image_id': int(img_id),
                    'calib': {'fx': float(calib.fu), 'fy': float(calib.fv), 'cx': float(calib.cu), 'cy': float(calib.cv)},
                    'gt_objects': [],
                    'predictions': []
                }
                
                dump_text += f"\n--- Image {img_id} ---\n"
                dump_text += f"Calib: fx={calib.fu:.2f}, fy={calib.fv:.2f}, cx={calib.cu:.2f}, cy={calib.cv:.2f}\n"
                dump_text += "GT Objects (Car):\n"
                gt_objects = self.dataloader.dataset.get_label(img_id)
                for obj in gt_objects:
                    if obj.cls_type != 'Car': continue
                    dump_text += f"  2D:[{obj.box2d[0]:.1f}, {obj.box2d[1]:.1f}, {obj.box2d[2]:.1f}, {obj.box2d[3]:.1f}] "
                    dump_text += f"3D:[{obj.pos[0]:.2f}, {obj.pos[1]:.2f}, {obj.pos[2]:.2f}] "
                    dump_text += f"Dim:[{obj.h:.2f}, {obj.w:.2f}, {obj.l:.2f}] "
                    dump_text += f"Depth:{obj.pos[2]:.2f} Alpha:{obj.alpha:.2f} Ry:{obj.ry:.2f}\n"
                    
                    img_data['gt_objects'].append({
                        '2d_bbox': [float(x) for x in obj.box2d],
                        '3d_pos': [float(x) for x in obj.pos],
                        'dim_hwl': [float(obj.h), float(obj.w), float(obj.l)],
                        'depth': float(obj.pos[2]),
                        'alpha': float(obj.alpha),
                        'rotation_y': float(obj.ry)
                    })
                    
                dump_text += "Predictions:\n"
                for p in data['preds']:
                    dump_text += f"  2D:[{p[2]:.1f}, {p[3]:.1f}, {p[4]:.1f}, {p[5]:.1f}] "
                    dump_text += f"3D:[{p[9]:.2f}, {p[10]:.2f}, {p[11]:.2f}] "
                    dump_text += f"Dim:[{p[6]:.2f}, {p[7]:.2f}, {p[8]:.2f}] "
                    dump_text += f"Depth:{p[11]:.2f} Alpha:{p[1]:.2f} Ry:{p[12]:.2f} Conf:{p[13]:.3f}\n"
                    
                    img_data['predictions'].append({
                        '2d_bbox': [float(p[2]), float(p[3]), float(p[4]), float(p[5])],
                        '3d_pos': [float(p[9]), float(p[10]), float(p[11])],
                        'dim_hwl': [float(p[6]), float(p[7]), float(p[8])],
                        'depth': float(p[11]),
                        'alpha': float(p[1]),
                        'rotation_y': float(p[12]),
                        'confidence': float(p[13])
                    })
                
                dump_data.append(img_data)

            print(dump_text)
            
            # Save dump to JSON file in a persistent directory (outputs_dir gets wiped every eval)
            step_val = step if step is not None else 0
            diag_dir = self.outputs_dir.parent / "diagnostics"
            diag_dir.mkdir(parents=True, exist_ok=True)
            json_path = diag_dir / f"eval_diagnostics_step_{step_val}.json"
            with open(json_path, 'w') as f:
                json.dump(dump_data, f, indent=4)
                
            # Order-sensitive checksum: sum of (index+1) * img_id
            order_sensitive_checksum = sum((i + 1) * img_id for i, img_id in enumerate(diag_ordered_img_ids))

            self.fabric.log_dict({
                "eval_diag/val_images_evaluated": float(diag_val_images),
                "eval_diag/gt_cars_total_in_file": float(diag_gt_cars),
                "eval_diag/gt_removed_by_depth_within_max_objs": float(diag_removed_by_depth),
                "eval_diag/gt_rejected_proj_outside_within_max_objs": float(diag_rejected_proj),
                "eval_diag/gt_skipped_encoding_within_max_objs": float(diag_skipped_encoding),
                "eval_diag/pred_raw_from_network": float(diag_pred_decoded),
                "eval_diag/pred_after_conf_threshold": float(diag_pred_surviving_conf),
                "eval_diag/pred_written_to_files": float(diag_pred_written),
                "eval_diag/pred_exhibits_invalid_depth": float(diag_pred_invalid_depth),
                "eval_diag/pred_exhibits_outside_depth": float(diag_pred_outside_depth),
                "eval_diag/pred_exhibits_invalid_geometry": float(diag_pred_invalid_geom),
                "eval_diag/val_images_ordered_checksum": float(order_sensitive_checksum)
            }, step=step_val)

        # save the result for evaluation.
        print('==> Saving ...')
        # print(local_results.keys())
        self.save_results(local_results)
        checksum_after = self._model_checksum()
        print(f"==> Live model checksum after evaluation: {checksum_after}")
        if step is not None:
            self.fabric.log("debug/eval_model_checksum_before", checksum_before, step=step)
            self.fabric.log("debug/eval_model_checksum_after", checksum_after, step=step)

    def _model_checksum(self):
        h = hashlib.sha256()

        for parameter in self.model.parameters():
            tensor = parameter.detach().float().cpu().contiguous()
            h.update(tensor.numpy().tobytes())

        return h.hexdigest()[:16]

    def should_log_images(self, batch_idx, step):
        rank_zero = self.fabric.is_global_zero
        return rank_zero and (self.logged_images < self.images_to_log) and (step is not None) and batch_idx % 32 == 0

    def log_images(
        self,
        outputs,
        targets,
        decoded,
        inputs,
        calibs_orig,
        calibs_model,
        info,
        step,
        batch_index,
    ):
        # Depth and region maps are produced on the model input grid. Visualize
        # all panels on that same grid instead of stretching them over the
        # dataset's original, potentially uncropped image.
        img_id = info['img_id'][batch_index]
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        input_rgb = inputs[batch_index].detach().float().cpu().numpy().transpose(1, 2, 0)
        input_rgb = np.clip((input_rgb * std + mean) * 255.0, 0, 255).astype(np.uint8)
        input_img = cv2.cvtColor(input_rgb, cv2.COLOR_RGB2BGR)
        img_h, img_w = input_img.shape[:2]

        # Decode predictions used the transformed calibration, so draw them
        # directly in model-input coordinates.
        box_pred_objs = []
        for pred_box in decoded[img_id]:
            class_name = self.class_name[int(pred_box[0])]
            kitti_str = f"{class_name} 0.0 0"
            for j in range(1, len(pred_box)):
                kitti_str += ' {:.2f}'.format(pred_box[j])
            obj_pred = Object3d(kitti_str)
            box_pred_objs.append(obj_pred)

        # Ground truth annotations remain in source-image coordinates. Apply
        # the exact image transform used by the dataset before drawing.
        box_true_obj = self.dataloader.dataset.get_label(img_id)
        source_calib = calibs_orig[batch_index]
        scale = float(info['resize_scale'][batch_index])
        offset_x = float(info['pad_w'][batch_index])
        offset_y = float(info['pad_h'][batch_index])
        flip = bool(info.get('flip', [False] * len(info['img_id']))[batch_index])
        mask = [obj.trucation <= 0.5 and obj.occlusion <= 2 for obj in box_true_obj]

        img_box_true = input_img.copy()
        gt_model_boxes = []
        for obj, visible in zip(box_true_obj, mask):
            color = (0, 255, 0) if visible else (0, 0, 255)
            box = np.asarray(obj.box2d, dtype=np.float32).copy()
            box[[0, 2]] = box[[0, 2]] * scale + offset_x
            box[[1, 3]] = box[[1, 3]] * scale + offset_y
            if flip:
                box[0], box[2] = img_w - box[2], img_w - box[0]
            box[[0, 2]] = np.clip(box[[0, 2]], 0, img_w - 1)
            box[[1, 3]] = np.clip(box[[1, 3]], 0, img_h - 1)
            gt_model_boxes.append(box.copy())
            if box[2] > box[0] and box[3] > box[1]:
                cv2.rectangle(
                    img_box_true,
                    (int(round(box[0])), int(round(box[1]))),
                    (int(round(box[2])), int(round(box[3]))),
                    color,
                    2,
                )
            img_box_true = draw_calibrated_box3d(
                img_box_true,
                obj.generate_corners3d(),
                source_calib,
                color=(0, 180, 255),
                thickness=2,
                scale=scale,
                offset_x=offset_x,
                offset_y=offset_y,
                flip=flip,
            )

        img_box_pred = plot_boxes(input_img, box_pred_objs, calibs_model[batch_index])

        # Compare the raw DPT prior, the ordinary MonoDPT depth expectation,
        # and the legacy median-thresholded display on one fixed metric scale.
        depth_min = float(self.cfg.model.depth_min)
        depth_max = float(self.cfg.model.depth_max)
        depth_maps_true, depth_pred_thresholded, depth_pred_softmax = get_depthmap_true(
            outputs, targets, self.cfg
        )
        depth_true_map = depth_maps_true[batch_index].detach().float().cpu().numpy()
        depth_softmax_tensor_all = depth_pred_softmax
        depth_softmax_map = depth_pred_softmax[batch_index].detach().float().cpu().numpy()
        depth_maps_true = cmap_mono(depth_true_map, depth_min, depth_max)
        depth_pred_thresholded = cmap_mono(
            depth_pred_thresholded[batch_index].cpu().numpy(), depth_min, depth_max
        )
        depth_pred_softmax = cmap_mono(depth_softmax_map, depth_min, depth_max)

        depth_prior = outputs.get('pred_depth_prior')
        if depth_prior is not None:
            depth_prior_map = depth_prior[batch_index, 0].detach().float().cpu().numpy()
            depth_prior_fixed = cmap_mono(depth_prior_map, depth_min, depth_max)
            depth_prior_relative, prior_low, prior_high = cmap_mono_relative(depth_prior_map)
        else:
            depth_prior_map = np.full((img_h, img_w), np.nan, dtype=np.float32)
            depth_prior_fixed = np.zeros((img_h, img_w, 3), dtype=np.uint8)
            depth_prior_relative = depth_prior_fixed.copy()
            prior_low, prior_high = float("nan"), float("nan")

        # Upscale prediction to the target-map grid and show absolute error
        # only where the GT depth target has supervision.
        depth_softmax_tensor = depth_softmax_tensor_all[batch_index : batch_index + 1]
        depth_softmax_up = F.interpolate(
            depth_softmax_tensor.unsqueeze(1),
            size=depth_true_map.shape,
            mode="bilinear",
            align_corners=False,
        )[0, 0].detach().cpu().numpy()
        valid_depth = np.isfinite(depth_true_map) & (depth_true_map > 0)
        depth_error = np.zeros_like(depth_true_map, dtype=np.float32)
        depth_error[valid_depth] = np.abs(
            depth_softmax_up[valid_depth] - depth_true_map[valid_depth]
        )
        error_max = min(20.0, depth_max)
        depth_error_color = cmap_mono(depth_error, 0.0, error_max)
        depth_error_color[~valid_depth] = 0

        # Sample the inner half of each GT box to avoid mixing the object with
        # surrounding background. Match the best same-class 2D detection.
        readout_lines = []
        for obj, gt_box in zip(box_true_obj, gt_model_boxes):
            dpt_value = sample_map_in_box(
                depth_prior_map, gt_box, (img_h, img_w)
            )
            mono_value = sample_map_in_box(
                depth_softmax_map, gt_box, (img_h, img_w)
            )
            matching_dets = []
            for pred_box in decoded[img_id]:
                pred_class_name = self.class_name[int(pred_box[0])]
                if pred_class_name != obj.cls_type:
                    continue
                iou = box_iou_xyxy(gt_box, pred_box[2:6])
                if iou >= 0.1:
                    matching_dets.append((iou, float(pred_box[11])))
            if matching_dets:
                best_iou, detection_depth = max(matching_dets, key=lambda item: item[0])
                detection_text = f"{detection_depth:.1f}m ({best_iou:.2f})"
            else:
                detection_text = "--"
            dpt_text = f"{dpt_value:.2f}" if np.isfinite(dpt_value) else "--"
            mono_text = f"{mono_value:.1f}m" if np.isfinite(mono_value) else "--"
            readout_lines.append(
                f"{obj.cls_type}: GT {obj.pos[2]:.1f}m | DPT p50 {dpt_text} | "
                f"Mono p50 {mono_text} | det {detection_text}"
            )
        img_box_true = draw_depth_readouts(img_box_true, readout_lines)

        depth_range = f"{depth_min:g}-{depth_max:g} m"
        depth_maps_true = label_plot_panel(
            depth_maps_true, f"GT center depth ({depth_range})", (img_h, img_w)
        )
        depth_pred_thresholded = label_plot_panel(
            depth_pred_thresholded,
            f"MonoDPT median-threshold ({depth_range})",
            (img_h, img_w),
        )
        depth_pred_softmax = label_plot_panel(
            depth_pred_softmax,
            f"MonoDPT softmax expected depth ({depth_range})",
            (img_h, img_w),
        )
        depth_prior_relative = label_plot_panel(
            depth_prior_relative,
            f"DPT prior relative p2-p98 [{prior_low:.2f}, {prior_high:.2f}]",
            (img_h, img_w),
        )
        depth_prior_fixed = label_plot_panel(
            depth_prior_fixed, f"DPT prior fixed range ({depth_range})", (img_h, img_w)
        )
        depth_error_color = label_plot_panel(
            depth_error_color,
            f"Abs error on GT pixels (0-{error_max:g} m; clipped)",
            (img_h, img_w),
        )

        # plot region seg
        obj_region_true = (targets['obj_region'][batch_index].cpu().numpy() * 255 ).astype(np.uint8)
        obj_region_pred = (outputs['pred_region_prob'][-1][batch_index][0].cpu().numpy() * 255).astype(np.uint8)

        target_h, target_w, _ = img_box_true.shape
        panel_rows = [
            [img_box_true, img_box_pred],
            [depth_maps_true, depth_pred_thresholded],
            [depth_prior_relative, depth_prior_fixed],
            [depth_pred_softmax, depth_error_color],
            [obj_region_true, obj_region_pred],
        ]
        log_image = prepare_plot_grid(panel_rows, (target_h, target_w))

        # log_image_np = log_image.numpy().transpose(1,2,0)
        # cv2.imwrite(
        #     f"logs/log_image_{self.logged_images}.png",
        #     cv2.cvtColor(log_image_np, cv2.COLOR_RGB2BGR),
        # )
        # raise Exception

        self.fabric.logger.log_image(
            f"val_images_{self.logged_images}",
            [log_image],
            step=step,
        )

        self.logged_images += 1

    def save_results(self, results):
        output_dir = self.outputs_dir
        output_dir.mkdir(exist_ok=True, parents=True)

        total_files_saved = 0
        empty_files_saved = 0
        orig_ds_id = 0
        for orig_ds, images in results.items():
            orig_ds_id+=1
            ds_output_dir = output_dir / orig_ds
            # print(ds_output_dir)
            ds_output_dir.mkdir(exist_ok=True, parents=True)
            for img_id, predictions in images.items():
                if self.dataset_type == 'KITTI':
                    output_path = ds_output_dir / '{:06d}.txt'.format(img_id)
                elif self.dataset_type == 'INDY':
                    output_path = os.path.join(ds_output_dir, '{:06d}.txt'.format(img_id))
                elif self.dataset_type == 'CUSTOM':
                    output_path = os.path.join(ds_output_dir, (self.filename_format[orig_ds_id-1] + '.txt') % img_id)
                else:
                    id_out = output_dir / self.dataloader.dataset.get_sensor_modality(img_id)
                    id_out.mkdir(exist_ok=True)
                    output_path = id_out / self.dataloader.dataset.get_sample_token(img_id) + '.txt'

                f = open(output_path, 'w')
                num_preds = len(results[orig_ds][img_id])
                if num_preds == 0:
                    empty_files_saved += 1

                for i in range(num_preds):
                    class_name = self.class_name[int(results[orig_ds][img_id][i][0])]
                    f.write('{} 0.0 0'.format(class_name))
                    for j in range(1, len(results[orig_ds][img_id][i])):
                        f.write(' {:.4f}'.format(results[orig_ds][img_id][i][j]))
                    f.write('\n')
                f.close()
                total_files_saved += 1

        print(f'DEBUG: Saved {total_files_saved} files total ({empty_files_saved} were empty - no detections)')
        prediction_checksum = self._prediction_checksum(output_dir)
        print(f"DEBUG: Prediction checksum: {prediction_checksum}")

    def _prediction_checksum(self, output_dir):
        h = hashlib.sha256()

        for path in sorted(output_dir.rglob("*.txt")):
            h.update(str(path.relative_to(output_dir)).encode())
            h.update(path.read_bytes())

        return h.hexdigest()[:16]

    def evaluate(self):
        self.fabric.barrier()
        # print('Inside evaluate from tester...')
        if not self.rank_zero:
            return None

        results_dir = self.outputs_dir
        assert results_dir.is_dir(), f"Results dir not found: {results_dir}"
        all_eval_results = {}

        # Iterate through each dataset subdirectory (e.g., 'kitti', 'waymo')
        # This assumes the folders were created by the save_results logic
        dataset_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
        
        if not dataset_dirs:
            print(f"Warning: No dataset subdirectories found in {results_dir}")
            return None

        for ds_dir in dataset_dirs:
            ds_name = ds_dir.name
            print(f'==> Evaluating dataset: {ds_name}')

           
            result = self.dataloader.dataset.eval(results_dir=ds_dir, logger=None)
            
            all_eval_results[ds_name] = result
            print(f"Evaluation complete for {ds_name}:", result)

        return all_eval_results
