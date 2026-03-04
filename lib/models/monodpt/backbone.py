# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from utils.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding
from lib.models.depth_anything_v2 import DINOv2, DPTHead

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, images):
        xs = self.body(images)
        out = {}
        for name, x in xs.items():
            m = torch.zeros(x.shape[0], x.shape[2], x.shape[3]).to(torch.bool).to(x.device)
            out[name] = NestedTensor(x, m)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        norm_layer = FrozenBatchNorm2d
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=norm_layer)
        assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"
        super().__init__(backbone, train_backbone, return_interm_layers)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2

# -- ele --
class DINOv2Backbone(nn.Module):
    """DINOv2 backbone for multi-scale feature extraction"""

    def __init__(self, model_name='vitl', train_backbone=True, return_interm_layers=True):
        super().__init__()

        # Load DINOv2 model
        self.backbone = DINOv2(model_name)
        self.model_name = model_name

        intermediate_layer_idx = {
            'vits': ([2, 5, 8, 11], 384),
            'vitb': ([2, 5, 8, 11], 768),
            'vitl': ([4, 11, 17, 23], 1024),
        }
        self.intermediate_layer_idx, self.embed_dim, = intermediate_layer_idx[model_name]
        self.patch_size = (14, 14)

        # Load DPT
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        }

        dpt_features = model_configs[model_name]['features']
        out_channels = model_configs[model_name]['out_channels']

        # Initialize DPT head with correct out_channels
        self.dpt_head = DPTHead(
            in_channels=self.embed_dim,
            features=dpt_features,
            use_bn=False,
            out_channels=out_channels,
            use_clstoken=False,
        )

        # Define output channels and strides to match ResNet interface
        if return_interm_layers:
            self.num_channels = out_channels
            self.strides = [8, 16, 32, 64]  # Approximate feature map strides
        else:
            self.num_channels = [out_channels[-1]] # wip
            self.strides = [32]

        # Freeze backbone if needed
        if train_backbone is False:
            """for param in self.backbone.parameters():
                param.requires_grad = False"""

            for name_1, param in self.backbone.named_parameters():
                # if not 'norm.' in name_1 and 'blocks.8' not in name_1 and 'blocks.11' not in name_1 and not 'cls_token' in name_1 and not 'pos_embed' in name_1 and not 'mask_token' in name_1:
                param.requires_grad_(False)

            for name_2, param in self.dpt_head.named_parameters():
                param.requires_grad_(False)

        self.model_name = model_name


    def forward(self, images):
        """
        Args:
            images: Tensor of shape (B, 3, H, W)
        Returns:
            Dict of NestedTensor features at different scales
        """

        # Padding
        pad = False
        B, C, H, W = images.shape

        if pad:
            if W % self.patch_size[1] != 0:
                images = F.pad(images, (0, self.patch_size[1] - W % self.patch_size[1]))
            if H % self.patch_size[0] != 0:
                images = F.pad(images, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
        else:
            H_dino = ((H + 13) // 14) * 14
            W_dino = ((W + 13) // 14) * 14

            images = F.interpolate(
                images,
                size=(H_dino, W_dino),
                mode='bilinear',
                align_corners=False
            )

        Hh, Ww = images.shape[-2], images.shape[-1]
        patch_h, patch_w = Hh // self.patch_size[0], Ww // self.patch_size[1]

        """# interpolate images to target size
        target_size = 518
        images = F.interpolate(images, size=(target_size, target_size), mode='bilinear', align_corners=False)
        patch_h, patch_w = target_size // 14, target_size // 14"""

        raw_features = self.backbone.get_intermediate_layers(images, self.intermediate_layer_idx, return_class_token=True) # todo: masks?
        depth = self.dpt_head(raw_features, patch_h, patch_w)

        # Original
        output = {}
        output["raw_features"] = depth

        for i, (feat, idx) in enumerate(zip(raw_features[-3:], self.intermediate_layer_idx[-3:])):
            # feat = features[2]
            x, cls = feat  # Patch tokens: shape [B, num_patches, embed_dim]

            # Reshape patch tokens to 2D feature maps
            x = x.permute(0, 2, 1).reshape(B, -1, patch_h, patch_w).contiguous()  # Shape: [B, embed_dim, H', W']

            # Create a binary mask (all zeros since we have no padding)
            mask = torch.zeros(B, patch_h * (2 ** (2 - i)), patch_w * (2 ** (2 - i)), dtype=torch.bool,
                               device=x.device)  # [B, H', W']

            # Store the feature map in the output dictionary
            output[str(idx)] = NestedTensor(x, mask)

        return output

# -- /ele --

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, samples: NestedTensor):
        xs = self[0](samples)

        out_feature_maps = []
        pos = []
        extra_outputs = {}

        for key, value in xs.items():
            if isinstance(value, NestedTensor):
                out_feature_maps.append(value)
                pos.append(self[1](value).to(value.tensors.dtype))
            else:
                # Keep additional outputs untouched
                extra_outputs[key] = value

        # If there are no extra outputs, return the original DETR-style result
        if not extra_outputs:
            return out_feature_maps, pos

        # Otherwise return everything clearly separated
        return out_feature_maps, pos, extra_outputs

def build_backbone(cfg):

    position_embedding = build_position_encoding(cfg)
    return_interm_layers = cfg.model.masks or cfg.model.num_feature_levels > 1

    # original
    # backbone = Backbone(cfg.model.backbone, cfg.model.train_backbone, return_interm_layers, cfg.model.dilation)

    #--- ele ---
    # Check if using DINOv2 or ResNet
    backbone_name = cfg.model.backbone

    if backbone_name.startswith('dinov2'):
        # Extract model size: 'dinov2_vitl' -> 'vitl'
        model_name = backbone_name.split('_')[1] if '_' in backbone_name else 'vitl'
        backbone = DINOv2Backbone(
            model_name=model_name,
            train_backbone=cfg.model.train_backbone,
            return_interm_layers=return_interm_layers
        )

    else:
        # Original ResNet backbone
        backbone = Backbone(cfg.model.backbone,
                            cfg.model.train_backbone,
                            return_interm_layers,
                            cfg.model.dilation)
    #--- /ele ---

    model = Joiner(backbone, position_embedding)
    return model
