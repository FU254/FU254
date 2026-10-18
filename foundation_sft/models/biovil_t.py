"""BioViL-T's official single-current-image inference path, loaded offline.

Source: microsoft/hi-ml b67c1d27c6b17d8e8ff01f8c507f3cabdb307388,
health_multimodal/image/model/{encoder,model,modules,resnet,pretrained}.py.
All checkpoint modules, including the inactive temporal pooler, are retained.
"""

import hashlib

import torch
from torch import nn

from .common import BackboneAdapter, BackboneFeatures, PreprocessSpec, load_checkpoint
from .extra_vlm_common import local_asset, require_linear_probe, strict_tensors


SOURCE_REVISION = "b67c1d27c6b17d8e8ff01f8c507f3cabdb307388"
SOURCE = "microsoft/BiomedVLP-BioViL-T:v1.0"
OFFICIAL_MD5 = "a83080e2f23aa584a4f2b24c39b1bb64"


def _resnet_map(model, images):
    x = model.maxpool(model.relu(model.bn1(model.conv1(images))))
    return model.layer4(model.layer3(model.layer2(model.layer1(x))))


class BioViLTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import resnet50
        from .biovil_temporal import VisionTransformerPooler

        # Exact hierarchy of the released MultiImageEncoder. Its unused fc is
        # still part of the official state dict and must load strictly.
        self.encoder = resnet50(weights=None)
        self.backbone_to_vit = nn.Conv2d(2048, 256, 1, bias=False)
        self.vit_pooler = VisionTransformerPooler(input_dim=256, grid_shape=(14, 14))
        self.missing_previous_emb = nn.Parameter(torch.zeros(1, 256, 1, 1))

    def forward(self, images):
        static = self.backbone_to_vit(_resnet_map(self.encoder, images))
        missing = self.missing_previous_emb.expand(static.shape[0], -1, *static.shape[-2:])
        return torch.cat((static, missing), dim=1)


class BioViLTImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = BioViLTEncoder()
        self.projector = nn.Module()
        self.projector.model = nn.Sequential(nn.Conv2d(512, 128, 1, bias=False), nn.BatchNorm2d(128),
                                             nn.ReLU(inplace=True), nn.Conv2d(128, 128, 1, bias=True))

    def forward(self, images):
        return self.projector.model(self.encoder(images))


class BioViLTAdapter(BackboneAdapter):
    def __init__(self, image_model, preprocess, provenance):
        super().__init__()
        self.image_model = image_model
        self.feature_dim = 128
        self.preprocess, self.provenance = preprocess, provenance

    def lora_target_names(self):
        raise ValueError("BioViL-T single-image inference bypasses temporal attention; use linear_probe")

    def forward_features(self, images):
        if images.ndim != 4 or images.shape[1:] != (3, self.preprocess.image_size, self.preprocess.image_size):
            raise ValueError("BioViL-T expects recorded square RGB inputs")
        spatial = self.image_model(images)
        return BackboneFeatures(spatial.mean((2, 3)), spatial.flatten(2).transpose(1, 2),
                                tuple(int(x) for x in spatial.shape[-2:])).validate()


def build_biovil_t(config):
    require_linear_probe(config)
    if config.get("checkpoint_source") != SOURCE:
        raise ValueError(f"BioViL-T requires checkpoint_source={SOURCE!r}")
    path, checksum = local_asset(config, "checkpoint")
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != OFFICIAL_MD5:
        raise ValueError("BioViL-T weights differ from the MD5 published by the official loader (v1.0)")
    model = BioViLTImageModel()
    strict_tensors(model, load_checkpoint(path), "BioViL-T complete image model")
    preprocess = PreprocessSpec(448, mean=(0.0,) * 3, std=(1.0,) * 3,
                                resize_mode="biovil_center_crop", resize_shortest_edge=512,
                                interpolation="bilinear")
    return BioViLTAdapter(model, preprocess, {
        "model": "biovil_t", "official_repository": "https://github.com/microsoft/hi-ml",
        "verified_source_revision": SOURCE_REVISION, "checkpoint_source": SOURCE,
        "checkpoint": str(path), "checkpoint_sha256": checksum, "official_md5": OFFICIAL_MD5,
        "architecture": "ResNet50 + static 256 channels + missing-prior 256 channels + 128-channel projector",
        "inference_branch": "Official previous_image=None; pretrained missing_previous_emb retained",
        "global_features": "Mean of the official 128-dimensional projected patch map, before optional L2 normalization",
        "local_features": "Official 128-dimensional projected spatial patch map; no disease-query tokens",
        "inactive_modules": "Temporal ViT pooler loaded strictly but unused without a previous image; no temporal LoRA",
        "preprocessing_implementation": "shared_dicom_display_then_biovil_resize512_crop448_bilinear_no_normalize_v1",
    })
