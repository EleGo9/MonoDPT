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
from lib.models.monodpt.depth_predictor.ddn_loss import DDNLoss

from utils import box_ops

B = 0

def plot_boxes(img: np.array, boxes: list[Object3d], calib: Calibration, mask: Optional[list] = None):
    corners_3d = []
    for box in boxes:
        box_coords_3d = box.generate_corners3d()
        corners_3d.append(box_coords_3d)

    img_box = img.copy()

    if len(corners_3d) > 0:
        corners_3d = np.stack(corners_3d)
        _, box_coords_2d = calib.corners3d_to_img_boxes(corners_3d)

        img_box = draw_boxes(img_box, box_coords_2d, mask=mask)

    return img_box

def draw_boxes(img, corners_2d_all, thickness=2, mask=None):

    """Draw 3D bounding box in 2D image."""
    corners_2d_all = corners_2d_all.astype(int) # .T
    for index, corners_2d in enumerate(corners_2d_all):
        if mask is not None:
            if mask[index]:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)  # red
        else:
            color = (0, 255, 0)

        # 4 bottom edges
        for i, j in zip([0,1,2,3],[1,2,3,0]):
            cv2.line(img, tuple(corners_2d[i]), tuple(corners_2d[j]), color, thickness)
        # 4 top edges
        for i, j in zip([4,5,6,7],[5,6,7,4]):
            cv2.line(img, tuple(corners_2d[i]), tuple(corners_2d[j]), color, thickness)
        # vertical edges
        for i, j in zip(range(4), range(4,8)):
            cv2.line(img, tuple(corners_2d[i]), tuple(corners_2d[j]), color, thickness)
    return img

def build_weighted_depth_from_logits(depth_logits):
    depth_num_bins = 80
    depth_min = 1e-3
    depth_max = 60.0
    bin_size = 2 * (depth_max - depth_min) / (depth_num_bins * (1 + depth_num_bins))
    bin_indice = torch.linspace(0, depth_num_bins - 1, depth_num_bins)
    bin_value = (bin_indice + 0.5).pow(2) * bin_size / 2 - bin_size / 8 + depth_min
    bin_value = torch.cat([bin_value, torch.tensor([depth_max])], dim=0).to(depth_logits.device)

    # Calculate the median value along the depth_num_bins dimension
    median_values = torch.median(depth_logits, dim=1, keepdim=True).values
    # Apply the threshold: set values below the median to zero
    thresholded_logits = torch.where(depth_logits >= median_values, depth_logits,
                                     torch.tensor(0.0).to(depth_logits.device))

    depth_probs = F.softmax(thresholded_logits, dim=1)
    weighted_depth = (depth_probs * bin_value.reshape(1, -1, 1, 1)).sum(dim=1)

    return weighted_depth

def get_depthmap_true(outputs, targets):

    depth_map_logits = outputs['pred_depth_map_logits']

    mask = targets['mask_2d']

    _, N, H, W = depth_map_logits.shape
    if N > 1:
        depth_map_logits = build_weighted_depth_from_logits(depth_map_logits).unsqueeze(1)

    num_gt_per_img = [m.sum() for m in mask]
    # (wip) cera un bug qui causato dalle shape hardcodate che rompevano la creazione della depthmap di GT
    if mask.sum() > 0:
        gt_boxes2d = torch.cat([b[m] for b, m in zip(targets['boxes'], mask) if m.any()]) * torch.tensor([W, H, W, H], device='cuda')  # torch.tensor([80, 24, 80, 24], device='cuda')
        gt_boxes2d = box_ops.box_cxcywh_to_xyxy(gt_boxes2d)
        gt_center_depth = torch.cat([b[m] for b, m in zip(targets['depth'], mask) if m.any()]).squeeze(dim=1)
    else:
        gt_boxes2d = torch.empty((0,4), dtype=torch.float32, device='cuda')
        gt_center_depth = torch.empty((0,), dtype=torch.float32, device='cuda')

    depth_map_true = DDNLoss.build_target_depth_from_3dcenter(depth_map_logits, gt_boxes2d, gt_center_depth, num_gt_per_img)

    return depth_map_true, depth_map_logits

def cmap_mono(mono):
    mono_jet = (mono - mono.min()) / (mono.max() - mono.min())
    mono_jet = (mono_jet * 255).astype(np.uint8)
    mono_jet = cv2.applyColorMap(mono_jet, cv2.COLORMAP_JET)

    return mono_jet

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
            if self.cfg.trainer.save_all:
                checkpoint_path = self.checkpoint_dir / "checkpoint_epoch_{}.pth".format(self.cfg.tester.checkpoint)
            else:
                checkpoint_path = Path(self.checkpoint_dir / "checkpoint_best.pth")
                # checkpoint_path =  Path('/media/matte/data/pinim/export_small/logs/worthy-oath-23/checkpoints/checkpoint_best.pth') #
                
            print("==> Test checkpoint path: {}".format(str(checkpoint_path)))
            assert checkpoint_path.is_file()

            load_checkpoint(
                fabric=self.fabric,
                model=self.model,
                optimizer=None,
                filename=checkpoint_path,
                logger=None)

            #self.model.to(self.device)
            self.inference()
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
                load_checkpoint(
                    fabric=self.fabric,
                    model=self.model,
                    optimizer=None,
                    filename=checkpoint,
                    logger=None)

                #self.model.to(self.device)
                self.inference()
                self.evaluate()

    def inference(self, step=None):
        self.logged_images = 0

        torch.set_grad_enabled(False)
        self.model.eval()
        checksum_before = self._model_checksum()
        print(f"==> Evaluating live model checksum: {checksum_before:.9e}")
        print(f"==> Prediction output directory: {self.outputs_dir}")

        local_results = {}
        model_infer_time = 0.0

        # tqdm only on rank 0 to avoid clutter
        progress_bar = tqdm.tqdm(
            total=len(self.dataloader),
            disable=not self.rank_zero,
            desc="Evaluation Progress"
        )


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

            # Get original calibrations (before letterbox adjustments)
            original_calibs = [self.dataloader.dataset.get_calib(index) for index in info['img_id']]

            # calibs from dataloader are P2 tensors that have ALREADY been adjusted for letterbox
            # We need to convert them to Calibration objects for decode_detections
            # Create Calibration objects from the adjusted P2 matrices
            import copy
            calibs_for_decode = []
            for i, calib_orig in enumerate(original_calibs):
                calib_adjusted = copy.deepcopy(calib_orig)
                # The P2 tensor from dataloader is already adjusted, so we apply the same adjustment
                if 'resize_scale' in info:
                    resize_scale = info['resize_scale'][i]
                    pad_w = info['pad_w'][i]
                    pad_h = info['pad_h'][i]

                    calib_adjusted.cu = calib_orig.cu * resize_scale + pad_w
                    calib_adjusted.cv = calib_orig.cv * resize_scale + pad_h
                    calib_adjusted.fu = calib_orig.fu * resize_scale
                    calib_adjusted.fv = calib_orig.fv * resize_scale
                    calib_adjusted.P2[0, 0] = calib_adjusted.fu
                    calib_adjusted.P2[1, 1] = calib_adjusted.fv
                    calib_adjusted.P2[0, 2] = calib_adjusted.cu
                    calib_adjusted.P2[1, 2] = calib_adjusted.cv
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
                        x3d_orig = np.array((p[2] + p[4]) / 2)
                        y3d_orig = np.array((p[3] + p[5]) / 2) # Or use p[5] if location is at the bottom
                        
                        depth = np.array(p[11]) # Depth (Z) remains unchanged by resolution)p[11] # Depth (Z) remains unchanged by resolution
                        
                        # Use the original calibration to get metric X, Y, Z
                        # .reshape(-1) ensures we get a flat [X, Y, Z] array
                        new_locs = original_calibs[i].img_to_rect(x3d_orig, y3d_orig, depth).reshape(-1)
                        
                        # KITTI 'y' location is typically the center of the object. 
                        # If your model predicts the bottom-face center, add half height back:
                        new_locs[1] += p[6] / 2 
                        
                        p[9] = new_locs[0]  # Corrected X (meters)
                        p[10] = new_locs[1] # Corrected Y (meters)
                        p[11] = new_locs[2] # Corrected Z (meters)

                        # Recalculate ry using original calibration and original x-coordinate
                        # ry depends on the calibration parameters (cu, fu) and the pixel position
                        alpha = p[1]
                        ry_corrected = original_calibs[i].alpha2ry(alpha, x3d_orig)
                        p[12] = ry_corrected

                        new_preds.append(p)


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
                    # if img_id in decoded:
                    preds = decoded.get(img_id, [])
                    orig_ds = info['orig_ds'][i]
                    if orig_ds not in local_results:
                        local_results[orig_ds] = {}
                    local_results[orig_ds][img_id] = preds

            progress_bar.update()

        progress_bar.close()

        # save the result for evaluation.
        print('==> Saving ...')
        # print(local_results.keys())
        self.save_results(local_results)
        checksum_after = self._model_checksum()
        print(f"==> Live model checksum after evaluation: {checksum_after:.9e}")
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

    def log_images(self, outputs, targets, decoded, calibs_orig, info, step):

        # Use ORIGINAL calibration for visualization (predictions already converted to original coords)
        calib = calibs_orig[B]
        img_id = info['img_id'][B]
        input_img = self.dataloader.dataset.get_image(img_id)
        input_img = np.array(input_img)

        # Get list of Object3d from predictions (already in original coordinates)
        box_pred_objs = []
        for pred_box in decoded[img_id]:
            class_name = self.class_name[int(pred_box[0])]
            kitti_str = f"{class_name} 0.0 0"
            for j in range(1, len(pred_box)):
                kitti_str += ' {:.2f}'.format(pred_box[j])
            obj_pred = Object3d(kitti_str)
            box_pred_objs.append(obj_pred)

        # Get list of Object3d from ground truth (in original coordinates)
        mask = targets['mask_2d'][B]
        box_true_obj = self.dataloader.dataset.get_label(img_id)  # type: list[Object3d]
        # box_true_obj = [b for b, k in zip(box_true_obj, mask) if k] # filter GT boxes

        # Plot 3D true-pred boxes (both in original coordinates now)
        img_box_true = plot_boxes(input_img, box_true_obj, calib, mask)
        img_box_pred = plot_boxes(input_img, box_pred_objs, calib)

        """cv2.imwrite("box_true_orig.png", img_box_true)
        cv2.imwrite("box_pred_orig.png", img_box_pred)"""

        # Plot depth mask pred
        depth_maps_true, depth_maps_pred = get_depthmap_true(outputs, targets)
        depth_maps_true = cmap_mono(depth_maps_true[B].cpu().numpy())
        depth_maps_pred = cmap_mono(depth_maps_pred[B][0].cpu().numpy())

        """cv2.imwrite("depth_maps_true.png", depth_maps_true)
        cv2.imwrite("depth_maps_pred.png", depth_maps_pred)"""

        # plot region seg
        obj_region_true = (targets['obj_region'][B].cpu().numpy() * 255 ).astype(np.uint8)
        obj_region_pred = (outputs['pred_region_prob'][-1][B][0].cpu().numpy() * 255).astype(np.uint8)

        target_h, target_w, _ = img_box_true.shape
        true_image = prepare_plot_tensor([img_box_true, depth_maps_true, obj_region_true], (target_h, target_w))
        pred_image = prepare_plot_tensor([img_box_pred, depth_maps_pred, obj_region_pred], (target_h, target_w))

        log_image = torch.concatenate([true_image, pred_image], dim=2)

        """log_image_np = log_image.numpy().transpose(1,2,0)
        cv2.imwrite(f"log_image_{self.logged_images}.png", log_image_np)
        raise Exception"""

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
                        f.write(' {:.2f}'.format(results[orig_ds][img_id][i][j]))
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
