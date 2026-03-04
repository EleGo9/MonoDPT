# MonoDPT

**Monocular 3D Object Detection with Depth Priors**

MonoDPT is a transformer-based framework for 3D object detection from a single camera image. It combines DINOv2 Vision Transformer backbones with a depth-aware detection head to predict 3D bounding boxes, object dimensions, orientation, and depth from a single RGB image — without LiDAR or stereo.

---

## Architecture Overview

```
Input Image
    │
    ▼
DINOv2 ViT Backbone (ViT-S/B/L) + DPT Depth Head
    │
    ├─► Multi-Scale Feature Projection
    │
    ├─► Depth Predictor (LID — Log-Interval Discretization)
    │
    ├─► 2D Detection Transformer (Encoder–Decoder)
    │
    └─► 3D Detection Transformer (Encoder–Decoder)
             │
             ▼
     3D Detection output:
       class, 3D bounding box (center, dimensions, yaw), depth
```

---

## Repository Structure

```
MonoDPT/
├── tools/
│   ├── train_val.py          # Training + validation entry point
│   ├── demo.py               # Inference and ONNX export entry point
│   └── plot_validation.py    # Validation visualization
├── configs/
│   ├── monodpt_kitti.yaml    # KITTI dataset config
│   ├── monodpt_indy.yaml     # INDY dataset config
│   └── monodpt_custom.yaml   # Custom dataset config
├── lib/
│   ├── models/monodpt/       # Model architecture
│   ├── datasets/             # KITTI / INDY / custom dataset loaders
│   ├── helpers/              # Config, model, dataloader, trainer, tester
│   └── losses/               # Focal, uncertainty, dim-aware losses
└── utils/                    # Misc utilities, box ops, DINOv2 helpers
```

---

## Installation

### 1. Create the conda environment

```bash
conda create -n monodpt_cu128 python=3.10
conda activate monodpt_cu128
```

### 2. Install PyTorch

```bash
pip3 install torch torchvision
```

> The default `pip3 install torch torchvision` command installs the latest stable release with CUDA support. If you need a specific CUDA version, follow the [PyTorch installation guide](https://pytorch.org/get-started/locally/).

### 3. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### Weights & Biases

Create a `seecrets.env` file in the project root with your W&B API key:

```env
WANDB_API_KEY=your_wandb_api_key_here
```

### Depth Anything V2 Pretrained Weights

Download the [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) checkpoint and set its path in your config under `trainer.depthany_model`.

---

## Configuration

All settings are controlled by YAML config files in `configs/`. Key sections:

| Section | Key parameters |
|---|---|
| **Global** | `fp16`, `world_size`, `gpus_per_node`, `logdir` |
| **Dataset** | `type` (`KITTI`/`INDY`/`custom`), `root_dir`, `train_split`, `test_split`, `batch_size` |
| **Model** | `backbone`, `num_classes`, `num_queries`, `enc_layers`, `dec_layers`, `hidden_dim` |
| **Depth** | `mode` (`LID`), `num_depth_bins`, `depth_min`, `depth_max` |
| **Optimizer** | `type` (`adamw`), `lr`, `weight_decay` |
| **Scheduler** | `warmup`, `decay`, `warmup_epochs`, `min_decay_lr` |
| **Trainer** | `max_epoch`, `gpu_ids`, `accum_iter`, `save_frequency` |
| **Tester** | `mode` (`single`/`all`), `checkpoint`, `threshold`, `topk` |

---

## Training

```bash
python tools/train_val.py --config configs/monodpt_kitti.yaml
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--config` | `configs/monodpt.yaml` | Path to YAML config file |
| `-e` / `--evaluate_only` | `False` | Run evaluation only, skip training |

**Evaluate only:**

```bash
python tools/train_val.py --config configs/monodpt_kitti.yaml -e
```

**Multi-GPU training** is controlled via the config file:

```yaml
world_size: 1       # number of nodes
gpus_per_node: 2    # GPUs per node
trainer:
  gpu_ids: '0,1'
  accum_iter: 4     # gradient accumulation steps
```

Checkpoints and outputs are saved under `logdir/<run-name>/checkpoints/`.

---

## ONNX Export

Use `demo.py` to load a trained checkpoint and export to ONNX format for deployment:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/demo.py \
    --config configs/monodpt_kitti.yaml \
    --ckpt /path/to/checkpoint.pth \
    -onnx
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--config` | — | Path to YAML config file (required) |
| `--ckpt` | — | Path to trained checkpoint `.pth` file |
| `-onnx` / `--onnx_export` | `False` | Export model to ONNX |

**ONNX export details:**
- Opset version: **16**
- Constant folding enabled
- Legacy exporter (stable for complex deformable attention models)
- Model is verified with `onnx.checker` and tested with ONNX Runtime

**Exported model inputs/outputs:**

| Name | Type | Description |
|---|---|---|
| `images` | input | RGB image tensor |
| `calibs` | input | Camera calibration matrix |
| `img_sizes` | input | Original image dimensions |
| `pred_logits` | output | Class scores per detection query |
| `pred_boxes` | output | 2D projected box coordinates |
| `pred_3d_dim` | output | 3D object dimensions (H, W, L) |
| `pred_angle` | output | Yaw angle |
| `pred_depth` | output | Per-object depth estimate |
| `pred_depth_map_logits` | output | Dense depth map logits (auxiliary) |
| `pred_region_prob` | output | Region probability (auxiliary) |

The primary 3D detection result is obtained by combining `pred_logits`, `pred_boxes`, `pred_3d_dim`, `pred_angle`, and `pred_depth` into 3D bounding boxes in camera coordinates.

---

## Results

### KITTI Benchmark — Car category (AP-R40)

> **Bold** = best &nbsp;|&nbsp; *Italic* = second best

| Method | Extra Data | Ref | Test AP<sub>BEV</sub> E/M/H | Test AP<sub>3D</sub> E/M/H | Val AP<sub>BEV</sub> E/M/H | Val AP<sub>3D</sub> E/M/H |
|---|---|---|---|---|---|---|
| CaDDN | LiDAR | CVPR'21 | 27.94 / 18.91 / 17.19 | 19.17 / 13.41 / 11.46 | — | 23.57 / 16.31 / 13.84 |
| MonoDTR | LiDAR | CVPR'22 | 28.59 / 20.38 / 17.14 | 21.99 / 15.39 / 12.73 | 33.33 / 25.35 / 21.68 | 24.52 / 18.57 / 15.51 |
| OccupancyM3D | LiDAR | CVPR'24 | *35.38* / 24.18 / 21.37 | 25.55 / 17.02 / 14.79 | 35.72 / 26.60 / 23.68 | 26.87 / 19.96 / 17.15 |
| OPA-3D | Depth | RAL'23 | 33.54 / 22.53 / 19.22 | 24.60 / 17.05 / 14.25 | 33.80 / 25.51 / 22.13 | 24.97 / 19.40 / 16.59 |
| GUPNet | None | ICCV'21 | — | 20.11 / 14.20 / 11.77 | 31.07 / 22.94 / 19.75 | 22.76 / 16.46 / 13.72 |
| DEVIANT | None | ECCV'22 | 29.65 / 20.44 / 17.43 | 21.88 / 14.46 / 11.89 | 32.60 / 23.04 / 19.99 | 24.63 / 16.54 / 14.52 |
| MonoDDE | None | CVPR'22 | 33.58 / 23.46 / 20.37 | 24.93 / 17.14 / 15.10 | 35.51 / 26.48 / 23.07 | 26.66 / 19.75 / 16.72 |
| MonoUNI | None | NeurIPS'23 | — | 24.75 / 16.73 / 13.49 | — | 24.51 / 17.18 / 14.01 |
| MonoDETR | None | ICCV'23 | 33.60 / 22.11 / 18.60 | 25.00 / 16.47 / 13.58 | 37.86 / 26.95 / 22.80 | 28.84 / 20.61 / 16.38 |
| MonoDinoDETR | None | IEEE IV'25 | — | — | 37.65 / 26.70 / 21.79 | 26.72 / 19.19 / 15.92 |
| MonoDinoDETR+DAB | None | IEEE IV'25 | — | — | 38.51 / 26.15 / 22.00 | 27.93 / 19.39 / 15.97 |
| MonoCD | None | CVPR'24 | 33.41 / 22.81 / 19.57 | 25.53 / 16.59 / 14.53 | 34.60 / 24.96 / 21.51 | 26.45 / 19.37 / 16.38 |
| FD3D | None | AAAI'24 | 34.20 / 23.72 / 20.76 | 25.38 / 17.12 / 14.50 | 36.98 / 26.77 / 23.16 | 28.22 / 20.23 / 17.04 |
| MonoDGP | None | CVPR'25 | 35.24 / *25.23* / *22.02* | *26.35* / *18.72* / *15.97* | *39.40* / *28.20* / *24.42* | *30.76* / *22.34* / *19.02* |
| **MonoDPT (ours)** | None | — | **38.27 / 26.57 / 23.34** | **29.63 / 20.80 / 17.93** | **42.93 / 31.22 / 27.00** | **33.83 / 24.23 / 20.49** |
| Improvement over prev. best | | | +2.89 / +1.34 / +1.32 | +3.28 / +2.08 / +1.96 | +3.53 / +3.02 / +2.58 | +3.07 / +1.89 / +1.47 |

---

### MonoRace3D Benchmark — Car category

> **Bold** = best &nbsp;|&nbsp; *Italic* = second best

| Method | FC AP<sub>BEV</sub> | FC AP<sub>3D</sub> | FC Av. D. err (m) | FC Av. D. err (%) | FC Rot. err (°) | FR AP<sub>BEV</sub> | FR AP<sub>3D</sub> | FR Av. D. err (m) | FR Av. D. err (%) | FR Rot. err (°) |
|---|---|---|---|---|---|---|---|---|---|---|
| MonoDETR | 19.95 | 13.19 | 4.17 | 10.58% | *3.06°* | 25.75 | 15.88 | 1.32 | *2.42%* | *3.35°* |
| MonoDGP | *28.14* | *15.17* | **1.25** | **2.79%** | 3.52° | *31.13* | *16.80* | *1.25* | 2.62% | 3.66° |
| **MonoDPT (ours)** | **29.51** | **22.02** | *2.25* | *4.26%* | **2.68°** | **46.47** | **34.24** | **0.95** | **1.55%** | **2.38°** |

FC = Frontal Central camera &nbsp;|&nbsp; FR = Frontal Right camera

---

## Supported Datasets

| Dataset | Config | Notes |
|---|---|---|
| **KITTI** | `monodpt_kitti.yaml` | Classes: Car, Pedestrian, Cyclist |
| **INDY** | `monodpt_indy.yaml` | Separate train/test dirs, distortion support |
| **Custom** | `monodpt_custom.yaml` | Multiple root dirs, configurable class names |

---

## Experiment Tracking

Training is logged to [Weights & Biases](https://wandb.ai). Model parameters, loss curves, and validation metrics are tracked automatically. To disable W&B logging, set:

```bash
WANDB_MODE=disabled python tools/train_val.py --config ...
```
