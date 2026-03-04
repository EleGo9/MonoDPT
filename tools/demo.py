import warnings
warnings.filterwarnings("ignore")

import os
import sys
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import yaml
import argparse
import datetime

from lib.helpers.model_helper import build_model
from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.optimizer_helper import build_optimizer
from lib.helpers.scheduler_helper import build_lr_scheduler
from lib.helpers.trainer_helper import Trainer
from lib.helpers.tester_helper import Tester
from lib.helpers.utils_helper import create_logger
from lib.helpers.utils_helper import set_random_seed
import time
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.config_helper import Config

from lib.helpers.decode_helper import extract_dets_from_outputs
from lib.helpers.decode_helper import decode_detections

import onnx
from omegaconf import OmegaConf
import wandb
from datetime import date
import lightning as L
from lightning.fabric.strategies import DDPStrategy  # type: ignore[reportPrivateImportUsage]
from wandb.integration.lightning.fabric import WandbLogger

from lib.models.monodpt.ops.functions import ms_deform_attn_func
import os
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
parser = argparse.ArgumentParser(description='MonoDPT Monocular 3D Object Detection')
parser.add_argument('--config', dest='config', help='settings of detection in yaml format')
parser.add_argument('--ckpt', help='dir with weights', default='/home/elenagovi/repos/multigpu/MonoDGP/logs/twisted-seance-326/checkpoints/checkpoint_best.pth' )
parser.add_argument('-onnx', '--onnx_export', action='store_true', default=False, help='onnx exportation')
args = parser.parse_args()


def main():
    import os
    assert (os.path.exists(args.config))
    config_file = OmegaConf.load(args.config)
    cfg = Config(**config_file)
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    torch.set_float32_matmul_precision(cfg.float32_matmul_precision)
    precision = "16-mixed" if cfg.fp16 else 32
    strategy = DDPStrategy(
        precision=precision,  # type: ignore
        find_unused_parameters=True,
    )

    logger = WandbLogger(
        project="monodgp_debug", # ;WANDB_MODE=disabled
        log_model=False,
        save_dir=cfg.logdir
    )

    fabric = L.Fabric(
        accelerator="gpu",
        devices=cfg.gpus_per_node, # gpu per nodo
        num_nodes=cfg.world_size,
        strategy=strategy,
        precision=precision,
        loggers=[logger],
    )

    fabric.launch()
    L.seed_everything(42)

    # cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    # print(cfg)
    def onnx_compatible_forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):
        return ms_deform_attn_func.ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights)

    ms_deform_attn_func.MSDeformAttnFunction.forward = onnx_compatible_forward

    model_name = cfg.model_name

    today = date.today()
    outputs_path = os.path.join('./' + cfg.trainer.save_path, model_name+str(today))
    os.makedirs(outputs_path, exist_ok=True)
    outputs_dir = fabric.broadcast(outputs_path, src=0)


    log_file = os.path.join(outputs_path, 'train.log.%s' % datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    logger = create_logger(log_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_ids = list(map(int, cfg.trainer.gpu_ids.split(',')))
    # build dataloader
    _, test_loader = build_dataloader(fabric, cfg)
    


    # build model
    model, loss = build_model(cfg)
    checkpoint_path = args.ckpt
    print(checkpoint_path)
    assert os.path.exists(checkpoint_path)
    checkpoint_dir = fabric.broadcast(checkpoint_path, src=0)

    # Load checkpoint with correct signature (uses fabric, not map_location)
    print("==> Loading checkpoint weights...")
    load_checkpoint(fabric=fabric,
                    model=model,
                    optimizer=None,
                    filename=checkpoint_path,
                    logger=logger)
    print("✓ Checkpoint loaded successfully")
    
    if args.onnx_export:
        for batch_idx, (inputs, calibs, targets, info) in enumerate(test_loader):
                inputs = inputs.to(device)
                print("inputs shape:", inputs.shape)
                calibs = calibs.to(device)
                print("calibs shape:", calibs.shape)
                img_sizes = info['img_size'].to(device)
                print("img_sizes shape:", img_sizes.shape)
                target = None
                dn_args = 0
                break

    if len(gpu_ids) == 1:
        model = model.to(device)
    else:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids).to(device)

    torch.set_grad_enabled(False)
    model.eval()

    # start_time = time.time()
    ###dn
    if args.onnx_export:
        # Create a wrapper to only return main outputs (no aux_outputs)
        # For inference, we don't need targets (training-only)
        class ModelWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, images, calibs, img_sizes):
                # During inference, targets=None and dn_args=None
                outputs = self.model(images, calibs, None, img_sizes, None)
                # Only return main outputs, not aux_outputs
                return (
                    outputs['pred_logits'],
                    outputs['pred_boxes'],
                    outputs['pred_3d_dim'],
                    outputs['pred_angle'],
                    outputs['pred_depth'],
                    outputs['pred_depth_map_logits'],
                    outputs['pred_region_prob']
                )

        wrapped_model = ModelWrapper(model)
        wrapped_model.eval()

        onnx_path = outputs_dir + ".onnx"
        print(f"\n{'='*60}")
        print(f"STARTING ONNX EXPORT")
        print(f"{'='*60}")
        print(f"Output path: {onnx_path}")
        print(f"Export mode: Inference only (no targets/dn_args)")
        print(f"Note: This may take several minutes and produce verbose output...")
        print(f"{'='*60}\n")

        # Use dynamo=False to use the legacy (more stable) ONNX exporter
        # Suppress verbose ONNX graph output
        import sys
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

        try:
            torch.onnx.export(
                wrapped_model,
                (inputs, calibs, img_sizes),  # Only essential inference inputs
                onnx_path,
                export_params=True,
                do_constant_folding=True,
                input_names=["images", "calibs", "img_sizes"],
                output_names=["pred_logits",
                                "pred_boxes",
                                "pred_3d_dim",
                                "pred_angle",
                                "pred_depth",
                                "pred_depth_map_logits",
                                "pred_region_prob"],
                # dynamic_axes={
                #     "images": {0: "batch_size", 2: "height", 3: "width"},
                #     "calibs": {0: "batch_size"},
                #     "img_sizes": {0: "batch_size"},
                #     "pred_logits": {0: "batch_size"},
                #     "pred_boxes": {0: "batch_size"},
                #     "pred_3d_dim": {0: "batch_size"},
                #     "pred_angle": {0: "batch_size"},
                #     "pred_depth": {0: "batch_size"},
                #     "pred_depth_map_logits": {0: "batch_size"},
                #     "pred_region_prob": {0: "batch_size"}
                # },
                opset_version=16,
                dynamo=False  # Use legacy exporter (more stable for complex models)
            )
        finally:
            # Restore stdout
            sys.stdout.close()
            sys.stdout = original_stdout

            print(f"\n{'='*60}")
            print(f"✅ ONNX EXPORT COMPLETED SUCCESSFULLY!")
            print(f"{'='*60}")
            print(f"Model saved to: {onnx_path}")

            # Check file size
            import os
            if os.path.exists(onnx_path):
                file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
                print(f"File size: {file_size_mb:.2f} MB")
            else:
                print(f"⚠️  WARNING: File not found at {onnx_path}")

        # except Exception as e:
        #     print(f"\n{'='*60}")
        #     print(f"❌ ONNX EXPORT FAILED!")
        #     print(f"{'='*60}")
        #     print(f"Error: {str(e)}")
        #     import traceback
        #     traceback.print_exc()
        #     return

        # 4️⃣ Verify the ONNX model
        onnx_model = onnx.load(onnx_path)  # Load the ONNX model
        onnx.checker.check_model(onnx_model)  # Check if the model is valid
        print("✅ ONNX model check passed!")

        # Test the exported model
        print("\nTesting exported model with sample input...")
        import onnxruntime as ort
        session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        print(f"Model has {len(session.get_outputs())} outputs:")
        for out in session.get_outputs():
            print(f"  - {out.name}: {out.shape}")
    
    logger.info('###################  Inference Only  ##################')
    tester = Tester(fabric=fabric,
                    cfg=cfg,
                    model=model,
                    dataloader=test_loader,
                    model_name=model_name,
                    checkpoint_dir=checkpoint_dir,
                    outputs_dir=outputs_dir)

    # tester.inference()
    return


if __name__ == '__main__':
    main()


