"""Offline, source-specific CNN encoders for frozen linear-head experiments.

No downloads occur here. The medical ResNet18 must contain the author's MoCo
query encoder, rather than silently substituting torchvision ImageNet weights.
"""

from pathlib import Path
import re

import torch
from torch import nn
from torch.nn import functional as F

from ..cnn_sources import CNN_SOURCES
from .common import BackboneAdapter, BackboneFeatures, PreprocessSpec, load_checkpoint, sha256_file


class CNNAdapter(BackboneAdapter):
    def __init__(self, encoder, architecture, channels, preprocess, provenance):
        super().__init__()
        self.encoder = encoder
        self.architecture = architecture
        self.input_channels = channels
        self.feature_dim = 1024 if architecture == "densenet121" else 512
        self.preprocess = preprocess
        self.provenance = provenance

    def lora_target_names(self):
        raise ValueError("These CNN baselines support linear_probe only; no attention LoRA is configured")

    def forward_features(self, images):
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("The shared CNN input must have shape [B,3,H,W]")
        if self.input_channels == 1:
            # The shared DICOM transform repeats one grayscale image. Taking
            # that channel preserves float32 bits, unlike averaging duplicates.
            images = images[:, :1]
        if self.architecture == "densenet121":
            feature_map = F.relu(self.encoder.features(images), inplace=False)
        else:
            x = self.encoder.maxpool(self.encoder.relu(self.encoder.bn1(self.encoder.conv1(images))))
            x = self.encoder.layer1(x)
            x = self.encoder.layer2(x)
            x = self.encoder.layer3(x)
            feature_map = self.encoder.layer4(x)
        pooled = F.adaptive_avg_pool2d(feature_map, 1).flatten(1)
        local = feature_map.flatten(2).transpose(1, 2)
        return BackboneFeatures(pooled, local, tuple(feature_map.shape[-2:])).validate()


def _dense_keys(state):
    # Official old torchvision ImageNet weights predate norm1/conv1 key names.
    pattern = re.compile(r"^(.*denselayer\d+\.(?:norm|relu|conv))\.([12])\.(weight|bias|running_mean|running_var)$")
    cleaned = {}
    for key, value in state.items():
        match = pattern.match(key)
        mapped = match.group(1) + match.group(2) + "." + match.group(3) if match else key
        if mapped in cleaned:
            raise ValueError("Duplicate DenseNet key after legacy-name conversion")
        cleaned[mapped] = value
    return cleaned


def _moco_encoder(state):
    prefixes = [prefix for prefix in ("module.encoder_q.", "encoder_q.") if any(key.startswith(prefix) for key in state)]
    if len(prefixes) != 1:
        raise ValueError("CXR ResNet18 requires the official MoCo encoder_q checkpoint; ImageNet weights are not a replacement")
    prefix = prefixes[0]
    encoder = {key[len(prefix):]: value for key, value in state.items()
               if key.startswith(prefix) and not key.startswith(prefix + "fc.")}
    if not encoder:
        raise ValueError("MoCo checkpoint contains no query-encoder backbone tensors")
    return encoder


def _strict_features(encoder, state, ignored=(), allow_legacy_bn_counts=False):
    state = {key: value for key, value in state.items() if key not in ignored}
    expected = encoder.state_dict()
    # Old ImageNet exports do not store this integer counter. It has no effect
    # in the frozen eval-mode encoder; running_mean/var must still be present.
    allowed_missing = ({key for key in expected if key.endswith("num_batches_tracked")}
                       if allow_legacy_bn_counts else set())
    missing = set(expected).difference(state).difference(allowed_missing)
    unexpected = set(state).difference(expected)
    if missing or unexpected:
        raise ValueError("CNN backbone mismatch: missing={}, unexpected={}".format(sorted(missing)[:12], sorted(unexpected)[:12]))
    for key, value in state.items():
        if value.shape != expected[key].shape or value.dtype != expected[key].dtype:
            raise ValueError("CNN tensor shape/dtype mismatch: " + key)
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
            raise ValueError("Non-finite pretrained CNN tensor: " + key)
    encoder.load_state_dict(state, strict=False)


def build_cnn(config):
    name = config["model"]
    if name not in CNN_SOURCES:
        raise ValueError("Unknown CNN source")
    if config.get("mode", "linear_probe") != "linear_probe":
        raise ValueError("These CNN baselines only train one linear classification head; choose mode=linear_probe")
    source = CNN_SOURCES[name]
    checkpoint = Path(config["checkpoint"]).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError("CNN checkpoint not found: {}. Download it before running the experiment.".format(checkpoint))
    checksum = sha256_file(checkpoint)
    expected_hash = source.get("sha256")
    expected_prefix = source.get("sha256_prefix")
    if ((expected_hash and checksum != expected_hash) or (expected_prefix and not checksum.startswith(expected_prefix))):
        raise ValueError("CNN checkpoint checksum differs from its official pretrained source")
    from torchvision.models import densenet121, resnet18
    if source["architecture"] == "densenet121":
        encoder = densenet121(weights=None)
        if source["input_channels"] == 1:
            encoder.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            from .cnn_checkpoint import load_xrv_densenet_state
            state = load_xrv_densenet_state(checkpoint)
        else:
            state = _dense_keys(load_checkpoint(checkpoint))
        encoder.classifier = nn.Identity()
        _strict_features(encoder, state, ignored=("classifier.weight", "classifier.bias", "op_threshs"),
                         allow_legacy_bn_counts=name == "densenet121_imagenet")
    else:
        encoder = resnet18(weights=None)
        encoder.fc = nn.Identity()
        state = _moco_encoder(load_checkpoint(checkpoint, state_key="state_dict"))
        _strict_features(encoder, state)
    provenance = {
        "model": name, "architecture": source["architecture"], "input_channels": source["input_channels"],
        "official_repository": source["repository"], "official_checkpoint_url": source["url"],
        "checkpoint": str(checkpoint), "checkpoint_sha256": checksum, "pretraining": source["pretraining"],
        "features": "ReLU DenseNet final map or ResNet layer4; adaptive global mean, no pretrained task head",
        "preprocessing_note": "Shared DICOM display processing plus source-specific normalization and declared geometry; not pixel-exact reproduction of author RGB/skimage pipeline",
    }
    return CNNAdapter(encoder, source["architecture"], source["input_channels"],
                      PreprocessSpec(**source["preprocessing"]), provenance)
