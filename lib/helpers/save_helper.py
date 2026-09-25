import os
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
from lightning.fabric import Fabric
import os

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def get_checkpoint_state(
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: Optional[int] = None,
    trainer_state: Optional[dataclass] = None,
    best_result: Optional[int] = None,
    best_epoch: Optional[int] = None,
):

    optim_state = optimizer.state_dict() if optimizer is not None else None
    scheduler_state = scheduler.state_dict() if scheduler is not None else None
    if model is not None:
        if isinstance(model, torch.nn.DataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    return {
        "epoch": epoch,
        "trainer_state": trainer_state,
        "model_state": model_state,
        "optimizer_state": optim_state,
        "scheduler_state": scheduler_state,
        "best_result": best_result,
        "best_epoch": best_epoch,
    }


def save_checkpoint(state: dict, filename: Path):
    filename = "{}.pth".format(filename)
    torch.save(state, filename)


def load_checkpoint(
    fabric: Fabric,
    model: Optional[nn.Module],
    optimizer: Optional[torch.optim.Optimizer],
    filename: Path,
    logger: Optional[Any] = None,
    scheduler: Optional[Any] = None,
):
    if os.path.isfile(filename):
        print("==> Loading from checkpoint '{}'".format(str(filename)))
        checkpoint = fabric.load(filename)
        epoch = checkpoint.get("epoch", -1)
        best_result = checkpoint.get("best_result", 0.0)
        best_epoch = checkpoint.get("best_epoch", 0.0)
        if model is not None and checkpoint["model_state"] is not None:
            model.load_state_dict(checkpoint["model_state"])
        if optimizer is not None and checkpoint["optimizer_state"] is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if scheduler is not None and checkpoint.get("scheduler_state") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        trainer_state = checkpoint["trainer_state"]
        print("==> Done")
    else:
        raise FileNotFoundError

    return epoch, best_result, best_epoch, trainer_state


def load_depthany_checkpoint(
    fabric: Fabric, model: nn.Module, filename: Path, key_mapping: dict
):
    """
    Load Depth Anything V2 checkpoint with custom key mapping.

    Args:
        fabric: Lightning Fabric instance
        model: The model to load weights into
        filename: Path to checkpoint file
        key_mapping: Dict mapping checkpoint prefixes to model prefixes
                    e.g., {'pretrained': 'backbone.backbone',
                           'depth_head': 'depth_predictor.dpt_head'}
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Checkpoint not found: {filename}")

    print(f"\n{'='*80}")
    print(f"Loading Depth Anything V2 checkpoint: {filename}")
    print(f"{'='*80}\n")

    # Load checkpoint using fabric
    checkpoint = fabric.load(filename)

    # The checkpoint is a flat dict with keys like 'pretrained.xxx' and 'depth_head.xxx'
    # We need to remap these to our model structure

    model_state_dict = {}
    stats = {prefix: 0 for prefix in key_mapping.keys()}

    for ckpt_key, ckpt_value in checkpoint.items():
        # Remove 'module.' prefix if present (from DDP)
        ckpt_key = ckpt_key.replace("module.", "")

        # Check which mapping applies
        for ckpt_prefix, model_prefix in key_mapping.items():
            if ckpt_key.startswith(ckpt_prefix):
                # Remap: 'pretrained.blocks.0.weight' -> 'backbone.backbone.blocks.0.weight'
                new_key = ckpt_key.replace(ckpt_prefix, model_prefix, 1)
                model_state_dict[new_key] = ckpt_value
                stats[ckpt_prefix] += 1
                break

    print("Checkpoint key mapping statistics:")
    for prefix, count in stats.items():
        target = key_mapping[prefix]
        print(f"  '{prefix}.*' -> '{target}.*': {count} keys")

    total_params = sum(v.numel() for v in model_state_dict.values())
    print(f"\nTotal keys to load: {len(model_state_dict)}")
    print(f"Total parameters to load: {total_params:,}")

    # Load into model
    missing, unexpected = model.load_state_dict(model_state_dict, strict=False)

    print(f"\nLoading results:")
    print(f"  - Loaded keys: {len(model_state_dict)}")
    print(f"  - Missing keys: {len(missing)}")
    if missing and len(missing) <= 10:
        for key in missing:
            print(f"      {key}")
    elif missing:
        for key in missing[:5]:
            print(f"      {key}")
        print(f"      ... and {len(missing) - 5} more")

    assert len(unexpected) == 0  # make sure we loaded all DAV2 weights
    print(f"  - Unexpected keys: {len(unexpected)}")
    if unexpected and len(unexpected) <= 10:
        for key in unexpected:
            print(f"      {key}")
    elif unexpected:
        for key in unexpected[:5]:
            print(f"      {key}")
        print(f"      ... and {len(unexpected) - 5} more")

    print(f"\n{'='*80}")
    print("Depth Anything V2 weights loaded successfully!")
    print(f"{'='*80}\n")

    return model
