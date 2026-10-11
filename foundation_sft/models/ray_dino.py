"""Native RayDINO / XRay-DINO from the authors' DINOv2 release.

The verified hub factory is xray_dino_vitl16: ViT-L/16, 512 pretraining
resolution, no registers, four block chunks. We construct it without pretrained
downloads and load the user's local medical checkpoint strictly.
"""

from contextlib import contextmanager
import importlib
import re
from pathlib import Path
import subprocess
import sys

from torch import nn

from .common import (BackboneAdapter, BackboneFeatures, PreprocessSpec,
                     load_backbone_weights, load_checkpoint, sha256_file)

OFFICIAL_REPOSITORY = "https://github.com/facebookresearch/dinov2"
OFFICIAL_WEIGHT_REQUEST = "https://ai.meta.com/resources/models-and-libraries/raydino-downloads/"


@contextmanager
def _official_source(directory):
    saved = {name: module for name, module in list(sys.modules.items())
             if name == "dinov2" or name.startswith("dinov2.")}
    for name in saved:
        del sys.modules[name]
    old_path = list(sys.path)
    sys.path.insert(0, str(directory))
    try:
        yield importlib.import_module("dinov2.hub.xray_dino.backbones")
    finally:
        sys.path[:] = old_path
        for name in list(sys.modules):
            if name == "dinov2" or name.startswith("dinov2."):
                del sys.modules[name]
        sys.modules.update(saved)


class RayDINOAdapter(BackboneAdapter):
    def __init__(self, backbone, preprocess, provenance):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = int(backbone.embed_dim)
        self.patch_size = int(backbone.patch_size)
        self.preprocess = preprocess
        self.provenance = provenance
        self._targets = [name for name, module in self.named_modules()
                         if isinstance(module, nn.Linear) and re.fullmatch(
                             r"backbone\.blocks\.\d+(?:\.\d+)?\.attn\.qkv", name)]
        if len(self._targets) != int(backbone.n_blocks):
            raise ValueError("Native RayDINO must expose one fused QKV projection per real block")

    def lora_target_names(self):
        return list(self._targets)

    def forward_features(self, images):
        if images.shape[-2] % self.patch_size or images.shape[-1] % self.patch_size:
            raise ValueError("RayDINO image dimensions must be divisible by its patch size")
        output = self.backbone.forward_features(images)
        if not isinstance(output, dict) or not {"x_norm_clstoken", "x_norm_patchtokens"}.issubset(output):
            raise ValueError("Native RayDINO encoder returned an unsupported feature layout")
        return BackboneFeatures(output["x_norm_clstoken"], output["x_norm_patchtokens"],
                                (images.shape[-2] // self.patch_size,
                                 images.shape[-1] // self.patch_size)).validate()


def build_ray_dino(config):
    if not config.get("repo_dir") or not config.get("checkpoint"):
        raise ValueError("RayDINO needs local official DINOv2 repo_dir and its released medical checkpoint")
    directory = Path(config["repo_dir"]).expanduser().resolve()
    source_file = directory / "dinov2/hub/xray_dino/backbones.py"
    checkpoint = Path(config["checkpoint"]).expanduser().resolve()
    if not source_file.is_file() or not checkpoint.is_file():
        raise FileNotFoundError("RayDINO source or checkpoint is absent; use the official XRay-DINO release")
    values = config.get("preprocessing", {})
    if values.get("mean") is None or values.get("std") is None:
        raise ValueError("Record RayDINO's release normalization in preprocessing.mean/std; do not guess it")
    size = values.get("image_size", 512)
    if not isinstance(size, int) or size <= 0 or size % 16:
        raise ValueError("RayDINO image_size must be a positive multiple of 16")
    preprocess = PreprocessSpec(size, tuple(values["mean"]), tuple(values["std"]),
                                resize_mode=values.get("resize_mode", "shortest_edge_center_crop"),
                                resize_shortest_edge=values.get("resize_shortest_edge", size),
                                interpolation=values.get("interpolation", "bicubic"))
    with _official_source(directory) as source:
        if Path(source.__file__).resolve() != source_file:
            raise RuntimeError("RayDINO constructor was not imported from the configured source")
        backbone = source.xray_dino_vitl16(pretrained=False)
    if (backbone.embed_dim != 1024 or backbone.n_blocks != 24
            or backbone.patch_size != 16 or backbone.num_register_tokens != 0):
        raise ValueError("The local XRay-DINO factory differs from the verified ViT-L/16 release")
    load_backbone_weights(backbone, load_checkpoint(checkpoint, config.get("state_key")), ignored_prefixes=())
    revision = subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"],
                              capture_output=True, text=True)
    provenance = {"model": "ray_dino", "factory": "xray_dino_vitl16",
                  "official_repository": OFFICIAL_REPOSITORY,
                  "repo_dir": str(directory), "repo_revision": revision.stdout.strip() if revision.returncode == 0 else None,
                  "source_sha256": sha256_file(source_file),
                  "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
                  "preprocessing_source": "explicit release settings supplied in configuration"}
    return RayDINOAdapter(backbone, preprocess, provenance)
