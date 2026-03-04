import os
from pathlib import Path

from enum import Enum
from typing import List, Optional, Literal, Union
from pydantic import BaseModel, Field, root_validator
import yaml
import torch

class BackboneType(str, Enum):
    resnet50 = "resnet50"
    depthany_small = "dinov2_vits" # todo: specificare configurazione vit a parte
    depthany_base = "dinov2_vitb"
    depthany_large = "dinov2_vitl"

class DatasetConfig(BaseModel):
    type: str = Field(..., alias="type")
    root_dir: Union[Path, List[Path], List[str], str]
    # Optional separate train/test directories (used by INDY and CUSTOM datasets)
    root_dir_train: Optional[Union[List[Path], List[str]]] = None
    root_dir_test: Optional[Union[List[Path], List[str]]] = None
    train_split: str
    test_split: str
    batch_size: int
    use_3d_center: bool = True
    class_merging: bool = False
    use_dontcare: bool = False
    bbox2d_type: str = 'anno'
    meanshape: bool = False
    writelist: List[str] = ['Car']
    clip_2d: bool = False

    aug_pd: bool = False
    aug_crop: bool = False
    aug_calib: bool = False

    random_flip: float = 0.5
    random_crop: float = 0.5
    random_mixup3d: float = 0.5
    scale: float = 0.4
    shift: float = 0.1

    depth_scale: str = 'normal'

    # Custom_Dataset configurable parameters (optional, with defaults)
    # These are ignored by KITTI_Dataset and INDY_Dataset unless explicitly used
    class_name: Optional[List[str]] = None  # e.g., ['Pedestrian', 'Car', 'Cyclist']
    cls2id: Optional[dict] = None  # e.g., {'Pedestrian': 0, 'Car': 1, 'Cyclist': 2}
    resolution: Optional[List[int]] = None  # e.g., [1920, 1080] (W, H)
    max_objs: Optional[int] = None  # e.g., 50
    cls_mean_size: Optional[List[List[float]]] = None  # e.g., [[H, W, L], [H, W, L], ...]
    filename_format: Optional[List] = None  # e.g., '%06d' for 000000.png, '%08d' for 00000000.png
    original_resolution: Optional[List[int]] = None  # e.g., [1920, 1080]
    kitti_official_eval: Optional[bool] = None

    class Config:
        allow_population_by_field_name = True
        extra = "allow"

class ModelConfig(BaseModel):
    num_classes: int
    return_intermediate_dec: bool
    device: str

    # Backbone
    backbone: BackboneType
    train_backbone: bool
    num_feature_levels: int
    dilation: bool
    position_embedding: str
    masks: bool

    # Depth predictor
    mode: str
    num_depth_bins: int
    depth_min: float
    depth_max: float

    # DPT
    use_dpt_head: bool
    discretization_mode: Literal["LID", "focused_LID"] = "LID"
    focused_max: float = 0.8
    overflow_value: float = 0.5
    soft_discretization: bool = True
    discretization_temp: int = 10.0

    # Transformer
    with_box_refine: bool
    use_dn: bool
    init_box: bool
    enc_layers: int
    dec_layers: int
    hidden_dim: int
    dim_feedforward: int
    dropout: float
    nheads: int
    num_queries: int
    group_num: int
    enc_n_points: int
    dec_n_points: int

    # DN
    scalar: int
    label_noise_scale: float
    box_noise_scale: float
    num_patterns: int

    # Loss
    aux_loss: bool

    # Loss coefficients
    cls_loss_coef: float
    focal_alpha: float

    bbox_loss_coef: float
    giou_loss_coef: float
    loss_3dcenter: float = Field(..., alias="3dcenter_loss_coef")
    dim_loss_coef: float
    angle_loss_coef: float
    depth_loss_coef: float
    depth_map_loss_coef: float
    region_loss_coef: float

    # Matcher
    set_cost_class: float
    set_cost_bbox: float
    set_cost_giou: float
    set_cost_3dcenter: float

    class Config:
        allow_population_by_field_name = True
        extra = "allow"

class OptimizerConfig(BaseModel):
    type: Literal["sgd", "adam", "adamw", "adamw_depthany"]
    lr: float
    weight_decay: float

    class Config:
        allow_population_by_field_name = True
        extra = "allow"

class LRSchedulerConfig(BaseModel):
    warmup: Union[Literal['linear', 'cos'], None]
    decay: Union[Literal['cos', 'poly', 'step'], None]
    warmup_epochs: Optional[int] = None
    warmup_init_lr: Optional[float] = None
    decay_rate: Optional[float] = None
    decay_list: Optional[List[int]] = None
    min_decay_lr: Optional[float] = None

    class Config:
        allow_population_by_field_name = True
        extra = "allow"

    @root_validator(pre=True)
    def validate_scheduler_config(cls, values):
        warmup = values.get("warmup")
        decay = values.get("decay")

        # --- Warmup checks ---
        if warmup is not None:
            if values.get("warmup_epochs") is None:
                raise ValueError("`warmup_epochs` is required when warmup is not None.")
            if values.get("warmup_init_lr") is None:
                raise ValueError("`warmup_init_lr` is required when warmup is not None.")

        # --- Decay checks ---
        if decay == "step":
            if values.get("decay_rate") is None:
                raise ValueError("`decay_rate` is required for step decay.")
            if values.get("decay_list") is None:
                raise ValueError("`decay_list` is required for step decay.")
        elif decay == "cos":
            if values.get("min_decay_lr") is None:
                raise ValueError("`min_decay_lr` is required for cosine decay.")

        return values

class TrainerConfig(BaseModel):
    max_epoch: int
    gpu_ids: str
    save_frequency: int
    save_all: bool
    use_dn: bool
    scalar: int
    label_noise_scale: float
    box_noise_scale: float
    num_patterns: int
    accum_iter: int
    resume_model: Optional[str] = None
    pretrain_model: Optional[Path] = None
    depthany_model: Optional[Path] = None

    class Config:
        allow_population_by_field_name = True
        extra = "allow"



class TesterConfig(BaseModel):
    type: str
    mode: str # ['single', 'all']
    checkpoint: int
    threshold: float = 0.2
    topk: int

    class Config:
        allow_population_by_field_name = True
        extra = "allow"

class Config(BaseModel):
    random_seed: int
    logdir: Path
    fp16: bool
    world_size: int
    gpus_per_node: int
    float32_matmul_precision: str = 'high'

    dataset: DatasetConfig
    model_name: str
    model_type: str = "MonoDPT"  # Model architecture type (MonoDPT, etc.)
    model: ModelConfig
    optimizer: OptimizerConfig
    lr_scheduler: LRSchedulerConfig
    trainer: TrainerConfig
    tester: TesterConfig

    """@root_validator(pre=True)
    def set_default_gpus(cls, values):
        if "gpus_per_node" not in values or values["gpus_per_node"] is None:
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                cuda_visible = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
                # ignore empty entries (can happen with trailing commas)
                cuda_visible = [d for d in cuda_visible if d.strip()]
                values["gpus_per_node"] = len(cuda_visible)
            elif torch.cuda.is_available():
                values["gpus_per_node"] = torch.cuda.device_count()
            else:
                values["gpus_per_node"] = 0
        return values"""

    @classmethod
    def load_from_yaml(cls, path: str) -> "Config":
        """
        Load and validate configuration from a YAML file.
        """
        with open(path, "r") as f:
            cfg_dict = yaml.safe_load(f)
        # pydantic will honor aliases; allow population by field name is enabled in submodels
        return cls.parse_obj(cfg_dict)

    class Config:
        allow_population_by_field_name = True
        extra = "allow"