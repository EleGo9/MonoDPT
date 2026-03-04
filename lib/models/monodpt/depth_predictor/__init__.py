from .depth_predictor import DepthPredictor
from .depth_predictor_dpt import DepthPredictorWithDPT

from lib.helpers.config_helper import Config

# Factory function to choose the right depth predictor
def build_depth_predictor(cfg: Config):
    if cfg.model.backbone in ["dinov2_vits", "dinov2_vitb", "dinov2_vitl"]:
        predictor = DepthPredictorWithDPT(cfg)
        return predictor
    else:
        return DepthPredictor(cfg)

"""# Load DPT head weights from Depth Anything V2 checkpoint (same as backbone)
checkpoint_path = cfg.get('backbone_checkpoint_path', None)
if checkpoint_path is not None:
    predictor.load_dpt_head_weights(checkpoint_path)
else:
    print("INFO: No backbone checkpoint path provided. DPT head using random initialization.")"""