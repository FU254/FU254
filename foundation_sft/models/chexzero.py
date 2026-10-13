"""Local CheXzero medical CLIP ViT-B/32 image encoder, without a text tower.

Architecture adapted from OpenAI/CLIP (MIT), Copyright (c) 2021 OpenAI:
https://github.com/openai/CLIP/blob/main/clip/model.py
The reviewed CheXzero ``pretrained=True`` release uses this 224-pixel encoder:
https://github.com/rajpurkarlab/CheXzero/blob/main/zero_shot.py

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions: The above copyright notice and this
permission notice shall be included in all copies or substantial portions of
the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""

from collections import OrderedDict
from pathlib import Path
import re

import torch
from torch import nn
from torch.nn import functional as F

from .common import (BackboneAdapter, BackboneFeatures, PreprocessSpec,
                     load_checkpoint, sha256_file)


SOURCE = "https://github.com/rajpurkarlab/CheXzero"
WEIGHTS_SOURCE = "https://drive.google.com/drive/folders/1makFLiEMbSleYltaRxw81aBhEDMpVwno"


class CLIPLayerNorm(nn.LayerNorm):
    """Preserve the original CLIP float32 normalization in mixed precision."""

    def forward(self, inputs):
        return super().forward(inputs.float()).to(inputs.dtype)


class QuickGELU(nn.Module):
    def forward(self, inputs):
        return inputs * torch.sigmoid(1.702 * inputs)


class ProjectedSelfAttention(nn.Module):
    """Original MHA computation with its fused projection exposed to LoRA.

    ``qkv.weight`` also works after ``inject_lora`` replaces the linear module.
    Accessing that property preserves gradients through the merged LoRA update;
    the original fused Q/K/V order is unchanged.
    """

    def __init__(self, width, heads):
        super().__init__()
        if width <= 0 or heads <= 0 or width % heads:
            raise ValueError("Attention width must be divisible by its positive head count")
        self.width, self.heads = width, heads
        self.qkv = nn.Linear(width, 3 * width)
        self.out_proj = nn.Linear(width, width)

    def forward(self, inputs):
        return F.multi_head_attention_forward(
            inputs, inputs, inputs, embed_dim_to_check=self.width, num_heads=self.heads,
            in_proj_weight=self.qkv.weight, in_proj_bias=self.qkv.bias,
            bias_k=None, bias_v=None, add_zero_attn=False, dropout_p=0.0,
            out_proj_weight=self.out_proj.weight, out_proj_bias=self.out_proj.bias,
            training=self.training, key_padding_mask=None, need_weights=False,
            attn_mask=None,
        )[0]


class ResidualAttentionBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.attn = ProjectedSelfAttention(width, heads)
        self.ln_1 = CLIPLayerNorm(width)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(width, width * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(width * 4, width)),
        ]))
        self.ln_2 = CLIPLayerNorm(width)

    def forward(self, inputs):
        inputs = inputs + self.attn(self.ln_1(inputs))
        return inputs + self.mlp(self.ln_2(inputs))


class CheXzeroVisionTransformer(nn.Module):
    """Fixed release defaults; smaller constructors support independent checks."""

    def __init__(self, input_resolution=224, patch_size=32, width=768,
                 layers=12, heads=12, output_dim=512):
        super().__init__()
        dimensions = (input_resolution, patch_size, width, layers, heads, output_dim)
        if any(type(value) is not int or value <= 0 for value in dimensions):
            raise ValueError("Vision dimensions must be positive integers")
        if input_resolution % patch_size or width % heads:
            raise ValueError("Image size and attention width must divide into patches and heads")
        self.input_resolution, self.patch_size = input_resolution, patch_size
        self.width, self.output_dim = width, output_dim
        self.conv1 = nn.Conv2d(3, width, patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = CLIPLayerNorm(width)
        self.blocks = nn.ModuleList([ResidualAttentionBlock(width, heads) for _ in range(layers)])
        self.ln_post = CLIPLayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward_features(self, images):
        if (images.ndim != 4 or images.shape[1] != 3
                or tuple(images.shape[-2:]) != (self.input_resolution, self.input_resolution)):
            raise ValueError(f"CheXzero expects three-channel {self.input_resolution}x{self.input_resolution} images")
        tokens = self.conv1(images.to(self.conv1.weight.dtype)).flatten(2).transpose(1, 2)
        cls = self.class_embedding.to(tokens.dtype).expand(tokens.shape[0], 1, -1)
        tokens = torch.cat((cls, tokens), dim=1)
        tokens = self.ln_pre(tokens + self.positional_embedding.to(tokens.dtype))
        tokens = tokens.transpose(0, 1)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = tokens.transpose(0, 1)
        # CLS follows the exact official image path. Applying the same frozen
        # output norm/projection to patches is an explicit spatial extension.
        pooled = self.ln_post(tokens[:, 0, :]) @ self.proj
        spatial = self.ln_post(tokens[:, 1:, :]) @ self.proj
        edge = self.input_resolution // self.patch_size
        return BackboneFeatures(pooled, spatial, (edge, edge)).validate()

    def forward(self, images):
        return self.forward_features(images).global_features


class CheXzeroAdapter(BackboneAdapter):
    def __init__(self, backbone, provenance):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = backbone.output_dim
        self.preprocess = PreprocessSpec(
            image_size=backbone.input_resolution,
            mean=(101.48761 / 255.0,) * 3, std=(83.43944 / 255.0,) * 3,
            resize_mode="chexzero_pad320", interpolation="bicubic")
        self.provenance = {
            **provenance,
            "pooling": "projected CLS without external L2 normalization",
            "local_features": "spatial extension: patch tokens through pretrained ln_post and projection",
            "preprocessing": "shared-DICOM adaptation: uint8 grayscale, LANCZOS long-side floor resize "
                             "to 320, centered zero padding, tensor bicubic to 224 (antialias=False), "
                             "common [0,1] clipping, then medical mean/std divided by 255",
            "preprocessing_scope": "official geometry and medical statistics; shared DICOM "
                                   "window/percentile rendering and final clipping differ from original PNG inputs",
        }

    def forward_features(self, images):
        return self.backbone.forward_features(images)

    def lora_target_names(self):
        return [f"backbone.blocks.{index}.attn.qkv" for index in range(len(self.backbone.blocks))]


def _visual_state(full_state):
    """Extract only the official visual tower and rename fused projections."""
    visual = {}
    for key, value in full_state.items():
        if not isinstance(key, str) or not key.startswith("visual."):
            continue
        name = key[len("visual."):]
        if name.startswith("transformer.resblocks."):
            name = "blocks." + name[len("transformer.resblocks."):]
        name = name.replace(".attn.in_proj_weight", ".attn.qkv.weight")
        name = name.replace(".attn.in_proj_bias", ".attn.qkv.bias")
        if name in visual:
            raise ValueError("Checkpoint has duplicate CheXzero visual parameter mappings")
        visual[name] = value
    if not visual:
        raise ValueError("CheXzero requires its official full CLIP checkpoint containing visual.* tensors")
    return visual


def _load_visual_weights(backbone, full_state):
    state = _visual_state(full_state)
    expected = backbone.state_dict()
    if set(state) != set(expected):
        raise ValueError("CheXzero visual weights mismatch: "
                         f"missing={sorted(set(expected) - set(state))[:12]}, "
                         f"unexpected={sorted(set(state) - set(expected))[:12]}")
    for name, value in state.items():
        if (not isinstance(value, torch.Tensor) or value.shape != expected[name].shape
                or not value.is_floating_point()):
            raise ValueError(f"CheXzero visual parameter shape/type mismatch: {name}")
        if not torch.isfinite(value).all().item():
            raise ValueError(f"CheXzero visual parameter is non-finite: {name}")
    # Validate all keys/shapes before copying anything; no random missing layers.
    backbone.load_state_dict(state, strict=True)


def _validate_release(state):
    visual = _visual_state(state)
    required_shapes = {"conv1.weight": (768, 3, 32, 32), "class_embedding": (768,),
                       "positional_embedding": (50, 768), "proj": (768, 512)}
    if any(key not in visual or tuple(visual[key].shape) != shape
           for key, shape in required_shapes.items()):
        raise ValueError("CheXzero supports the official CLIP-pretrained ViT-B/32 at 224, "
                         "with 768 channels and a 512-dimensional projection")
    indices = {int(match.group(1)) for name in visual
               if (match := re.match(r"blocks\.(\d+)\.", name))}
    if indices != set(range(12)):
        raise ValueError("The CheXzero ViT-B/32 release must contain exactly 12 visual transformer blocks")


def build_chexzero(config):
    if config.get("image_size", 224) != 224:
        raise ValueError("CheXzero's official CLIP-pretrained release requires image_size=224")
    if config.get("architecture", "ViT-B/32") != "ViT-B/32":
        raise ValueError("Only the official CheXzero CLIP-pretrained ViT-B/32 release is supported")
    path = Path(config.get("checkpoint", "")).expanduser()
    if not path.is_file():
        raise FileNotFoundError("Set checkpoint to one local official CheXzero full CLIP .pt file")
    fingerprint = sha256_file(path)
    expected_fingerprint = config.get("checkpoint_sha256")
    if expected_fingerprint is not None:
        if (not isinstance(expected_fingerprint, str)
                or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_fingerprint)
                or expected_fingerprint.lower() != fingerprint):
            raise ValueError("CheXzero checkpoint SHA256 differs from checkpoint_sha256")
    state = load_checkpoint(path, config.get("state_key"))
    _validate_release(state)
    backbone = CheXzeroVisionTransformer()
    _load_visual_weights(backbone, state)
    if sha256_file(path) != fingerprint:
        raise ValueError("CheXzero checkpoint changed while loading")
    return CheXzeroAdapter(backbone, {
        "model": "chexzero", "architecture": "CLIP ViT-B/32", "image_size": 224,
        "visual_parameters": sum(parameter.numel() for parameter in backbone.parameters()),
        "checkpoint": str(path.resolve()), "sha256": fingerprint,
        "checkpoint_sha256_verified": expected_fingerprint is not None,
        "source": SOURCE, "weights_source": config.get("checkpoint_source", WEIGHTS_SOURCE),
        "release": "official MIMIC-CXR medical CLIP-pretrained branch; single visual encoder",
        "ignored_checkpoint_keys": sorted(key for key in state if not key.startswith("visual.")),
        "attention_loading": "original in_proj Q/K/V weights preserved in LoRA-compatible fused linear",
    })
