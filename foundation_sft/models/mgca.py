"""Offline MGCA ViT-B/16: the published query image encoder, without a text tower.

Reviewed source: HKU-MedAI/MGCA at 3a2b4f25d47efc62e7d01d9204632a23d453afae,
mgca/models/backbones/{encoder,vits}.py and datasets/{transforms,utils}.py.
The author downstream encoder returns normalized CLS and patch tokens, before
the separate contrastive projection heads. No DeiT/ImageNet download is made.
"""

from functools import partial

import torch
from torch import nn

from .common import BackboneAdapter, BackboneFeatures, PreprocessSpec, load_checkpoint
from .extra_vlm_common import local_asset, strict_tensors


SOURCE_REVISION = "3a2b4f25d47efc62e7d01d9204632a23d453afae"
SOURCE = "HKU-MedAI/MGCA:vit_base"


def create_mgca_encoder(**test_geometry):
    from timm.models.vision_transformer import VisionTransformer

    settings = dict(img_size=224, patch_size=16, embed_dim=768, depth=12,
                    num_heads=12, mlp_ratio=4, qkv_bias=True, num_classes=0,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=0.0)
    settings.update(test_geometry)
    return VisionTransformer(**settings)


class MGCAAdapter(BackboneAdapter):
    def __init__(self, encoder, preprocess, provenance):
        super().__init__()
        self.encoder = encoder
        self.feature_dim = encoder.num_features
        self.preprocess, self.provenance = preprocess, provenance

    def lora_target_names(self):
        return [f"encoder.blocks.{i}.attn.qkv" for i in range(len(self.encoder.blocks))]

    def forward_features(self, images):
        if images.ndim != 4 or images.shape[1:] != (3, self.preprocess.image_size, self.preprocess.image_size):
            raise ValueError("MGCA expects recorded square RGB inputs")
        model = self.encoder
        tokens = model.patch_embed(images)
        tokens = torch.cat((model.cls_token.expand(tokens.shape[0], -1, -1), tokens), dim=1)
        tokens = model.pos_drop(tokens + model.pos_embed)
        for block in model.blocks:
            tokens = block(tokens)
        tokens = model.norm(tokens)
        grid = tuple(int(x) for x in model.patch_embed.grid_size)
        return BackboneFeatures(tokens[:, 0], tokens[:, 1:], grid).validate()


def build_mgca(config):
    if config.get("checkpoint_source") != SOURCE:
        raise ValueError(f"MGCA requires checkpoint_source={SOURCE!r}; generic ViT weights are not a substitute")
    path, checksum = local_asset(config, "checkpoint")
    state = load_checkpoint(path, state_key="state_dict")
    prefix = "img_encoder_q.model."
    # These tensors identify a full MGCA checkpoint rather than a generic ViT
    # export. The projection heads are deliberately unused by the official LP.
    for name in ("img_encoder_q.global_embed.head.0.weight", "img_encoder_q.local_embed.head.0.weight"):
        if name not in state:
            raise ValueError("Expected the official full MGCA Lightning checkpoint, including image projection heads")
    backbone = {name[len(prefix):]: tensor for name, tensor in state.items() if name.startswith(prefix)}
    encoder = create_mgca_encoder()
    strict_tensors(encoder, backbone, "MGCA query ViT-B/16")
    preprocess = PreprocessSpec(224, mean=(0.5,) * 3, std=(0.5,) * 3,
                                resize_mode="mgca_pad256_center_crop", interpolation="bilinear")
    return MGCAAdapter(encoder, preprocess, {
        "model": "mgca", "official_repository": "https://github.com/HKU-MedAI/MGCA",
        "verified_source_revision": SOURCE_REVISION, "checkpoint_source": SOURCE,
        "checkpoint": str(path), "checkpoint_sha256": checksum,
        "state_key": "state_dict.img_encoder_q.model", "architecture": "ViT-B/16, 224, 12 blocks, 768 channels",
        "global_features": "Final normalized CLS token, as in the author downstream image encoder",
        "local_features": "Final normalized spatial patch tokens, excluding CLS",
        "unused_pretraining_modules": "Image contrastive projection heads and text encoder; author LP uses raw visual features",
        "preprocessing_implementation": "shared_dicom_display_then_mgca_opencv_area_pad256_crop224_v1",
        "preprocessing_scope": "Author evaluation geometry/normalization after this study's shared DICOM rendering; shared augmentation",
    })
