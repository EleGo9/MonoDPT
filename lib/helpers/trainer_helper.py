import tqdm
from pathlib import Path
from dataclasses import dataclass

import torch
import numpy as np
from torch import Tensor, nn, optim


from lightning.fabric import Fabric
from torch.utils.data import DataLoader
from lib.helpers.save_helper import get_checkpoint_state
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.save_helper import save_checkpoint
from lib.helpers.save_helper import load_depthany_checkpoint
from lib.helpers.config_helper import Config

@dataclass
class TrainingState:
    global_iteration: int
    global_step: int

class Trainer(object):
    def __init__(self,
                 fabric: Fabric,
                 cfg: Config,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 train_loader: DataLoader,
                 test_loader: DataLoader,
                 lr_scheduler: optim.lr_scheduler._LRScheduler,
                 loss: nn.Module,
                 model_name: str,
                 checkpoint_dir: Path):

        self.fabric = fabric
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr_scheduler = lr_scheduler
        self.epoch = 0
        self.best_result = 0
        self.best_epoch = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detr_loss = loss
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir
        self.tester = None

        if cfg.trainer.depthany_model is not None:
            assert cfg.trainer.depthany_model.is_file()
            # Load Depth Anything V2 checkpoint with custom key mapping
            # Map checkpoint keys to model keys:
            # 'pretrained.*' -> 'backbone.0.backbone.*' (DINOv2 encoder inside backbone)
            # 'depth_head.*' -> 'depth_predictor.dpt_head.*' (DPT head inside depth predictor)
            load_depthany_checkpoint(
                fabric=self.fabric,
                model=self.model,
                filename=cfg.trainer.depthany_model,
                key_mapping={
                    'pretrained': 'backbone.0.backbone',
                    'depth_head': 'backbone.0.dpt_head' #<----------- ATTENZIONE!! (wip)
                }
            )

        # loading pretrain/resume model
        if cfg.trainer.pretrain_model is not None:
            assert cfg.trainer.pretrain_model.is_file()
            load_checkpoint(
                fabric=self.fabric,
                model=self.model,
                optimizer=None,
                filename=cfg.trainer.pretrain_model,
                logger=None)

        if cfg.trainer.resume_model is not None:
            resume_model_path = self.cfg.logdir / self.cfg.trainer.resume_model / "checkpoints/checkpoint.pth" # todo: put in pydantic
            assert resume_model_path.is_file()
            self.epoch, self.best_result, self.best_epoch, self.state = load_checkpoint(
                fabric=self.fabric,
                model=self.model,
                optimizer=self.optimizer,
                filename=resume_model_path,
                logger=None)

            self.lr_scheduler.last_epoch = self.epoch - 1
            print("Loading Checkpoint... Best Result:{}, Best Epoch:{}".format(self.best_result, self.best_epoch))
        else:
            self.state = TrainingState(global_step=0, global_iteration=0)

    @property
    def rank_zero(self):
        return self.fabric.is_global_zero

    def log(self, key, value, step):
        if step % 2 == 0:
            self.fabric.log(key, value, step=step)
        
    def train(self):
        start_epoch = self.epoch

        progress_bar = tqdm.tqdm(range(start_epoch, self.cfg.trainer.max_epoch),
                                 dynamic_ncols=True, leave=True, desc='epochs',
                                 disable=not self.rank_zero)
        best_result = self.best_result
        best_epoch = self.best_epoch

        # self.tester.inference(step=self.state.global_step)
        # cur_result = self.tester.evaluate()
        # exit(0)
        for epoch in range(start_epoch, self.cfg.trainer.max_epoch):
            # reset random seed
            # ref: https://github.com/pytorch/pytorch/issues/5059
            np.random.seed(np.random.get_state()[1][0] + epoch)
            # train one epoch
            self.train_one_epoch(epoch)
            self.epoch += 1

            # save trained model
            if (self.epoch % self.cfg.trainer.save_frequency) == 0:

                if self.rank_zero:
                    if self.cfg.trainer.save_all:
                        ckpt_name = self.checkpoint_dir / 'checkpoint_epoch_%d' / self.epoch
                    else:
                        ckpt_name = self.checkpoint_dir / 'checkpoint'

                    # node: by using fabri.save we dont need to deal with rank_0 here
                    save_checkpoint(
                            get_checkpoint_state(self.model, self.optimizer, self.epoch, self.state, best_result, best_epoch),
                            ckpt_name)

                if self.tester is not None: # todo: rank0 save
                    print("Test Epoch {}".format(self.epoch))
                    self.tester.inference(step=self.state.global_step)
                    cur_result = self.tester.evaluate()

                    # Handle different result formats from different datasets
                    # - KITTI returns scalar
                    # - INDY returns (3d_mAP, 2d_mAP)
                    # - Multiple datasets return dict {'dataset_name': (3d_mAP, 2d_mAP)}
                    if isinstance(cur_result, dict):
                        # Multiple datasets: average the 3D mAP across all datasets
                        all_3d_maps = []
                        all_2d_maps = []
                        for ds_name, ds_result in cur_result.items():
                            if isinstance(ds_result, tuple):
                                all_3d_maps.append(ds_result[0])
                                all_2d_maps.append(ds_result[1])
                            else:
                                all_3d_maps.append(ds_result)
                        metric_value = sum(all_3d_maps) / len(all_3d_maps) if all_3d_maps else 0.0
                        cur_result_3d = metric_value
                        cur_result_2d = sum(all_2d_maps) / len(all_2d_maps) if all_2d_maps else None
                    elif isinstance(cur_result, tuple):
                        cur_result_3d, cur_result_2d = cur_result
                        metric_value = cur_result_3d  # Use 3D mAP as primary metric
                    else:
                        metric_value = cur_result
                        cur_result_3d = cur_result
                        cur_result_2d = None

                    if self.rank_zero and metric_value > best_result:
                        best_result = metric_value
                        best_epoch = self.epoch
                        ckpt_name = self.checkpoint_dir / 'checkpoint_best'
                        save_checkpoint(
                            get_checkpoint_state(self.model, self.optimizer, self.epoch, self.state, best_result, best_epoch),
                            ckpt_name)
                        self.fabric.log(f"test/3DmAP_best", cur_result_3d, step=self.state.global_step)  # log best

                    self.fabric.log(f"test/3DmAP", cur_result_3d, step=self.state.global_step)  # always log metric!
                    if cur_result_2d is not None:
                        self.fabric.log(f"test/2DmAP", cur_result_2d, step=self.state.global_step)  # log 2D mAP if available

                    print("Best Result:{}, epoch:{}".format(best_result, best_epoch))

            progress_bar.update()

        print("Best Result:{}, epoch:{}".format(best_result, best_epoch))

        return None

    def train_one_epoch(self, epoch):
        torch.set_grad_enabled(True)
        self.model.train()
        self.fabric.log("epoch", epoch, step=self.state.global_step)

        progress_bar = tqdm.tqdm(total=len(self.train_loader), leave=(self.epoch+1 == self.cfg.trainer.max_epoch), desc='iters', disable=not self.rank_zero)

        # Per-class loss tracking - accumulate over epoch
        per_class_losses = {}
        per_class_counts = {}
        num_classes = self.cfg.model.num_classes

        for batch_idx, (inputs, calibs, targets, info) in enumerate(self.train_loader):

            # compute next iteration and accumulation step (avoid off-by-one)
            next_iter = self.state.global_iteration + 1
            is_step = (next_iter) % self.cfg.trainer.accum_iter == 0
            if is_step:
                self.state.global_step += 1
                self.fabric.log("global_step", self.state.global_step)

            # update the stored iteration
            self.state.global_iteration = next_iter

            img_sizes = targets['img_size']
            targets = self.prepare_targets(targets, inputs.shape[0])

            # Denoising args
            dn_args = None
            if self.cfg.trainer.use_dn:
                dn_args=(targets, self.cfg.trainer.scalar, self.cfg.trainer.label_noise_scale, self.cfg.trainer.box_noise_scale, self.cfg.trainer.num_patterns)

            # train one batch
            with self.fabric.no_backward_sync(self.model, enabled=not is_step):
                with self.fabric.autocast():
                    outputs = self.model(inputs, calibs, targets, img_sizes, dn_args=dn_args)
                    mask_dict = None
                    detr_losses_dict = self.detr_loss(outputs, targets, mask_dict, self.fabric)

                    weight_dict = self.detr_loss.weight_dict
                    detr_losses_dict_weighted = [
                        detr_losses_dict[k] * weight_dict[k]
                        for k in detr_losses_dict.keys()
                        if k in weight_dict
                    ]
                    detr_losses = sum(detr_losses_dict_weighted)

                    if is_step:
                        # Reduce across nodes
                        detr_losses_dict = self.fabric.all_reduce(detr_losses_dict, reduce_op="mean")

                        detr_losses_dict_log = {}
                        detr_losses_log = 0
                        for k in detr_losses_dict.keys():
                            if k in weight_dict:
                                detr_losses_dict_log[k] = (detr_losses_dict[k] * weight_dict[k]).item()
                                detr_losses_log += detr_losses_dict_log[k]
                        detr_losses_dict_log["loss_detr"] = detr_losses_log

                        # Log to Fabric
                        self.log("train/loss_detr", detr_losses_log, step=self.state.global_step)

                        for k, v in detr_losses_dict_log.items():
                            if k != "loss_detr":
                                self.log(f"train/{k}", v, step=self.state.global_step)

                        # Compute per-class losses
                        self._update_per_class_losses(outputs, targets, per_class_losses, per_class_counts)

                    detr_losses /= self.cfg.trainer.accum_iter
                    self.fabric.backward(detr_losses)

            if is_step:
                # update optimizer
                self.optimizer.step()
                self.optimizer.zero_grad()

                # update learning rate
                self.lr_scheduler.step()

                # Update lr_scale
                for lr, group in zip(self.lr_scheduler.get_last_lr(), self.lr_scheduler.optimizer.param_groups):
                    if "lr_scale" in group:
                        group["lr"] = lr * group["lr_scale"]
                    else:
                        group["lr"] = lr
                    self.fabric.log(f"lr/{group['name']}", group["lr"], step=self.state.global_step)

                # update progressbar
                progress_bar.set_postfix({"train_loss": detr_losses.item()})

            progress_bar.update()
            # if batch_idx>10:
            #     break
        progress_bar.close()

        # Log per-class losses at end of epoch
        self._log_per_class_losses(per_class_losses, per_class_counts)

    def prepare_targets(self, targets, batch_size):
        targets_list = []
        mask = targets['mask_2d']

        key_list = ['labels', 'boxes', 'calibs', 'depth', 'size_3d', 'heading_bin', 'heading_res', 'boxes_3d']
        for bz in range(batch_size):
            target_dict = {}
            for key, val in targets.items():
                if key in key_list:
                    target_dict[key] = val[bz][mask[bz]]
                if key == 'depth_map':
                    target_dict[key] = val[bz]
                if key == 'obj_region':
                    target_dict[key] = val[bz]
            targets_list.append(target_dict)
        return targets_list

    def _update_per_class_losses(self, outputs, targets, per_class_losses, per_class_counts):
        """Update per-class loss tracking for current batch"""
        import torch.nn.functional as F
        from lib.losses.focal_loss import sigmoid_focal_loss

        # Get predictions from final layer
        src_logits = outputs['pred_logits']  # [batch, num_queries, num_classes]
        pred_depth = outputs['pred_depth'][:, :, 0]  # [batch, num_queries]
        pred_angle = outputs['pred_angle']  # [batch, num_queries, 24]

        # For each batch item, compute losses per ground truth object
        for batch_idx, target in enumerate(targets):
            if len(target['labels']) == 0:
                continue

            # Handle squeeze carefully to avoid 0-d tensors when only 1 object
            gt_labels = target['labels'].view(-1).long()  # [num_objects]
            gt_depth = target['depth'].view(-1)  # [num_objects]
            gt_heading_bin = target['heading_bin']  # [num_objects, 12]
            gt_heading_res = target['heading_res']  # [num_objects, 12]

            # For each ground truth object
            for obj_idx in range(gt_labels.shape[0]):
                class_id = gt_labels[obj_idx]
                class_id = class_id.item()

                # Initialize tracking for this class if needed
                if class_id not in per_class_losses:
                    per_class_losses[class_id] = {
                        'classification': 0.0,
                        'depth': 0.0,
                        'angle': 0.0
                    }
                    per_class_counts[class_id] = 0

                # Find best matching prediction for this GT object (simple approach: use highest scoring prediction)
                # In reality, the matcher assigns predictions, but for logging we'll use a simple heuristic
                class_probs = src_logits[batch_idx, :, class_id].sigmoid()
                best_pred_idx = class_probs.argmax()

                # Classification loss (focal loss contribution for this class)
                pred_logit = src_logits[batch_idx, best_pred_idx, class_id]
                target_onehot = torch.zeros_like(src_logits[batch_idx, best_pred_idx])
                target_onehot[class_id] = 1.0
                cls_loss = sigmoid_focal_loss(
                    src_logits[batch_idx, best_pred_idx].unsqueeze(0).unsqueeze(0),
                    target_onehot.unsqueeze(0).unsqueeze(0),
                    num_boxes=1,
                    alpha=0.25,
                    gamma=2
                ).item()

                # Depth loss (L1 loss)
                depth_pred = pred_depth[batch_idx, best_pred_idx]
                depth_gt = gt_depth[obj_idx]
                depth_loss = F.l1_loss(depth_pred, depth_gt).item()

                # Angle loss (cross entropy for bin + L1 for residual)
                angle_pred = pred_angle[batch_idx, best_pred_idx]
                heading_bin_pred = angle_pred[:12]
                heading_res_pred = angle_pred[12:]

                heading_bin_gt = gt_heading_bin[obj_idx]
                heading_res_gt = gt_heading_res[obj_idx]

                bin_cls_loss = F.cross_entropy(
                    heading_bin_pred.unsqueeze(0),
                    heading_bin_gt.argmax().unsqueeze(0),
                    reduction='mean'
                ).item()

                # Residual loss only for the correct bin
                bin_idx = heading_bin_gt.argmax()
                res_loss = F.l1_loss(
                    heading_res_pred[bin_idx].unsqueeze(0),
                    heading_res_gt[bin_idx].unsqueeze(0)
                ).item()

                angle_loss = bin_cls_loss + res_loss

                # Accumulate losses
                per_class_losses[class_id]['classification'] += cls_loss
                per_class_losses[class_id]['depth'] += depth_loss
                per_class_losses[class_id]['angle'] += angle_loss
                per_class_counts[class_id] += 1

    def _log_per_class_losses(self, per_class_losses, per_class_counts):
        """Log averaged per-class losses at end of epoch"""
        if not per_class_losses or not self.rank_zero:
            return

        # Get class names from dataset config
        if hasattr(self.cfg.dataset, 'class_name'):
            class_names = self.cfg.dataset.class_name
        else:
            # Fallback to KITTI class names
            class_names = ['Car', 'Pedestrian', 'Cyclist']

        print("\n" + "="*80)
        print(f"Per-Class Loss Summary (Epoch {self.epoch})")
        print("="*80)

        # for class_id in sorted(per_class_losses.keys()):
        #     count = per_class_counts[class_id]
        #     if count == 0:
        #         continue

        #     class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"

        #     avg_cls_loss = per_class_losses[class_id]['classification'] / count
        #     avg_depth_loss = per_class_losses[class_id]['depth'] / count
        #     avg_angle_loss = per_class_losses[class_id]['angle'] / count

        #     print(f"{class_name:15s} (n={count:4d}): "
        #           f"cls={avg_cls_loss:.4f}, depth={avg_depth_loss:.4f}, angle={avg_angle_loss:.4f}")

        #     # Log to tensorboard/wandb
        #     self.fabric.log(f"train_per_class/{class_name}/classification", avg_cls_loss, step=self.state.global_step)
        #     self.fabric.log(f"train_per_class/{class_name}/depth", avg_depth_loss, step=self.state.global_step)
        #     self.fabric.log(f"train_per_class/{class_name}/angle", avg_angle_loss, step=self.state.global_step)
        #     self.fabric.log(f"train_per_class/{class_name}/count", count, step=self.state.global_step)

        print("="*80 + "\n")
