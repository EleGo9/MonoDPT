import warnings
warnings.filterwarnings("ignore")

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import torch
import argparse
from omegaconf import OmegaConf
from pathlib import Path
from dotenv import load_dotenv
import wandb
import lightning as L
from lightning.fabric.strategies import DDPStrategy  # type: ignore[reportPrivateImportUsage]
from wandb.integration.lightning.fabric import WandbLogger

from lib.helpers.config_helper import Config
from lib.helpers.model_helper import build_model
from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.optimizer_helper import build_optimizer
from lib.helpers.scheduler_helper import build_lr_scheduler
from lib.helpers.trainer_helper import Trainer
from lib.helpers.tester_helper import Tester
from utils.misc import printTrainingParams, printModelParamCounts
#from lib.helpers.utils_helper import set_random_seed

load_dotenv("secrets.env")
os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")

parser = argparse.ArgumentParser(description='Monocular 3D Object Detection with Decoupled-Query and Geometry-Error Priors')
parser.add_argument('--config', dest='config', help='settings of detection in yaml format', default="configs/monodpt.yaml")
parser.add_argument('-e', '--evaluate_only', action='store_true', default=False, help='evaluation only')
parser.add_argument('--exp_name', default=None, help='evaluation only')
args = parser.parse_args()

def main():

    # Build Configuration Object
    config_file = OmegaConf.load(args.config)
    cfg = Config(**config_file)

    wandb.login(key=os.getenv("WANDB_API_KEY"))
    model_name = cfg.model_name

    # Load fabric
    torch.set_float32_matmul_precision(cfg.float32_matmul_precision)
    major, minor = torch.cuda.get_device_capability()
    if cfg.fp16:
        precision = "bf16-mixed" if major >= 8 else "16-mixed"
    else:
        precision = 32
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

    # build dataloader
    train_loader, test_loader = build_dataloader(fabric, cfg)

    # build model and optimizer
    model, loss = build_model(cfg)
    optimizer = build_optimizer(cfg, model)

    model, optimizer = fabric.setup(model, optimizer)
    logger.watch(model)

    if fabric.is_global_zero:
        if args.evaluate_only:
            assert args.exp_name is not None
            checkpoint_dir = cfg.logdir / args.exp_name / "checkpoints"
            outputs_dir = cfg.logdir / args.exp_name / "outputs"
        else:
            checkpoint_dir = cfg.logdir / logger.experiment.name / "checkpoints"
            outputs_dir = cfg.logdir / logger.experiment.name / "outputs"

        logger.experiment.config.update(cfg.model_dump())
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        printModelParamCounts(model, logger.experiment.name)

    else:
        checkpoint_dir = None
        outputs_dir = None

    checkpoint_dir = fabric.broadcast(checkpoint_dir, src=0)
    outputs_dir = fabric.broadcast(outputs_dir, src=0)

    # build scheduler
    iterations_per_epoch = len(train_loader)
    steps_per_epoch = iterations_per_epoch // cfg.trainer.accum_iter
    lr_scheduler = build_lr_scheduler(cfg, optimizer, max_epochs=cfg.trainer.max_epoch, last_epoch=-1,
                                      steps_per_epoch=steps_per_epoch)

    if args.evaluate_only:
        print('###################  Evaluation Only  ##################')
        tester = Tester(
                        fabric=fabric,
                        cfg=cfg,
                        model=model,
                        dataloader=test_loader,
                        model_name=model_name,
                        checkpoint_dir=checkpoint_dir,
                        outputs_dir=outputs_dir)
        tester.test()
        return

    trainer = Trainer(
                fabric=fabric,
                cfg=cfg,
                model=model,
                optimizer=optimizer,
                train_loader=train_loader,
                test_loader=test_loader,
                lr_scheduler=lr_scheduler,
                loss=loss,
                model_name=model_name,
                checkpoint_dir=checkpoint_dir)

    tester = Tester(
                    fabric=fabric,
                    cfg=cfg,
                    model=trainer.model,
                    dataloader=test_loader,
                    model_name=model_name,
                    checkpoint_dir=checkpoint_dir,
                    outputs_dir=outputs_dir)

    if cfg.dataset.test_split != 'test':
        trainer.tester = tester

    if fabric.is_global_zero:
        printTrainingParams(cfg, steps_per_epoch, iterations_per_epoch, precision)

    print('###################  Training  ##################')
    print('Batch Size: %d' % (cfg.dataset.batch_size))
    print('Learning Rate: %f' % (cfg.optimizer.lr))

    trainer.train()

    if cfg.dataset.test_split == 'test':
        return

    print('###################  Testing  ##################')
    print('Batch Size: %d' % (cfg.dataset.batch_size))
    print('Split: %s' % (cfg.dataset.test_split))

    tester.test()




if __name__ == '__main__':
    main()