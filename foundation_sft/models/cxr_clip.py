"""The official CXR-CLIP Swin-Tiny MCC visual encoder, loaded locally.

The released ``.tar`` is a PyTorch checkpoint, not a tar archive. Its visual
state is loaded strictly without constructing the text encoder or loading Hub
weights. CXR-CLIP's released code uses the first final spatial token as its
global feature. Swin has no CLS token; this adapter preserves that choice.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn

from .common import BackboneAdapter, BackboneFeatures, PreprocessSpec, sha256_file


OFFICIAL_REPOSITORY = "https://github.com/Soombit-ai/cxr-clip"
VERIFIED_SOURCE_REVISION = "31188857cc5e892c22c731b176080eb5d4484cf2"
OFFICIAL_IMAGE_MODEL_ID = "microsoft/swin-tiny-patch4-window7-224"
OFFICIAL_CHECKPOINT_NAME = "swint_mcc.tar"
OFFICIAL_CHECKPOINT_URL = (
    "https://twg.kakaocdn.net/brainrepo/models/cxr-clip/"
    "a25ce760064591c899f67c97ed7790de/swint_mcc.tar"
)
OFFICIAL_TRAINING_DATASETS = {"mimic_cxr", "chexpert", "chest14"}
OFFICIAL_ARCHITECTURE = {
    "image_size": 224,
    "patch_size": 4,
    "num_channels": 3,
    "embed_dim": 96,
    "depths": [2, 2, 6, 2],
    "num_heads": [3, 6, 12, 24],
    "window_size": 7,
    "mlp_ratio": 4.0,
    "qkv_bias": True,
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "drop_path_rate": 0.1,
    "hidden_act": "gelu",
    "use_absolute_embeddings": False,
    "layer_norm_eps": 1e-5,
    "initializer_range": 0.02,
    "encoder_stride": 32,
}


def _pair(value):
    return (value, value) if isinstance(value, int) else tuple(value)


class CXRClipAdapter(BackboneAdapter):
    """Expose all Swin spatial features and the official first-token summary."""

    def __init__(self, encoder, preprocess, provenance=None):
        super().__init__()
        self.encoder = encoder
        self.preprocess = preprocess
        self.provenance = dict(provenance or {})
        config = encoder.config
        if config.model_type != "swin" or int(config.num_channels) != 3:
            raise ValueError("CXR-CLIP requires a three-channel Hugging Face Swin encoder")
        self.feature_dim = int(config.hidden_size)
        self.patch_size = _pair(config.patch_size)
        if len(self.patch_size) != 2 or min(self.patch_size) < 1:
            raise ValueError("Invalid Swin patch size")
        depths = list(config.depths)
        stages = encoder.encoder.layers
        if len(stages) != len(depths) or not depths or any(depth < 1 for depth in depths):
            raise ValueError("Swin stages differ from the recorded architecture")
        self.stage_count = len(stages)
        self._targets = []
        for stage_index, (stage, depth) in enumerate(zip(stages, depths)):
            if len(stage.blocks) != depth:
                raise ValueError("Swin block count differs from the recorded architecture")
            width = int(config.embed_dim) * 2 ** stage_index
            for block_index, block in enumerate(stage.blocks):
                attention = block.attention.self
                for projection_name in ("query", "key", "value"):
                    projection = getattr(attention, projection_name, None)
                    if (not isinstance(projection, nn.Linear) or projection.in_features != width
                            or projection.out_features != width):
                        raise ValueError("Swin attention does not expose the verified Q/K/V projections")
                    self._targets.append(
                        f"encoder.encoder.layers.{stage_index}.blocks.{block_index}."
                        f"attention.self.{projection_name}"
                    )
        if len(self._targets) != 3 * sum(depths) or self.feature_dim != int(config.embed_dim) * 2 ** (self.stage_count - 1):
            raise ValueError("Swin attention or output dimensions differ from the recorded architecture")

    def lora_target_names(self):
        return list(self._targets)

    def forward_features(self, images):
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected images shaped [batch, 3, height, width]")
        if images.shape[-2] < self.patch_size[0] or images.shape[-1] < self.patch_size[1]:
            raise ValueError("Swin inputs must contain at least one image patch")
        grid = [(images.shape[-2 + axis] + patch - 1) // patch for axis, patch in enumerate(self.patch_size)]
        for _ in range(self.stage_count - 1):
            grid = [(dimension + 1) // 2 for dimension in grid]
        tokens = self.encoder(pixel_values=images, return_dict=True).last_hidden_state
        if tuple(tokens.shape) != (images.shape[0], grid[0] * grid[1], self.feature_dim):
            raise RuntimeError("Swin spatial tokens differ from the expected final patch grid")
        return BackboneFeatures(tokens[:, 0], tokens, tuple(grid))


def _training_config(payload):
    if not isinstance(payload, Mapping) or not isinstance(payload.get("config"), Mapping):
        raise ValueError("CXR-CLIP requires its released checkpoint training configuration")
    config = payload["config"]
    model = config.get("model")
    if not isinstance(model, Mapping) or model.get("name") != "clip_custom":
        raise ValueError("Checkpoint is not the released CXR-CLIP pretraining model")
    image_encoder = model.get("image_encoder")
    expected = {"source": "huggingface", "name": OFFICIAL_IMAGE_MODEL_ID, "model_type": "swin"}
    if not isinstance(image_encoder, Mapping) or any(image_encoder.get(key) != value for key, value in expected.items()):
        raise ValueError("Checkpoint does not identify the official CXR-CLIP Swin-Tiny visual architecture")
    if image_encoder.get("pretrained", True) is not True:
        raise ValueError("CXR-CLIP Swin-Tiny checkpoint records an unsupported initialization")
    datasets = config.get("data_train")
    if datasets is not None and (not isinstance(datasets, Mapping) or set(datasets) != OFFICIAL_TRAINING_DATASETS):
        raise ValueError("Checkpoint training datasets do not match the fixed CXR-CLIP MCC release")
    return config


def _preprocess(training_config):
    """Validate saved eval transforms or use the reviewed fixed release protocol."""
    base = training_config.get("base", {})
    if not isinstance(base, Mapping) or base.get("image_size", 224) != 224:
        raise ValueError("CXR-CLIP Swin-Tiny MCC requires the released 224-pixel image size")
    transforms = training_config.get("transform")
    recorded = False
    if transforms is not None:
        if not isinstance(transforms, Mapping):
            raise ValueError("Invalid CXR-CLIP checkpoint transform metadata")
        for split in ("valid", "test"):
            transform = transforms.get(split)
            if transform is None:
                continue
            recorded = True
            if not isinstance(transform, Mapping) or list(transform) != ["Resize", "CenterCrop"]:
                raise ValueError("CXR-CLIP evaluation requires Resize then CenterCrop, without CLAHE")
            for operation in ("Resize", "CenterCrop"):
                values = transform[operation]
                if not isinstance(values, Mapping) or values.get("size") not in (224, "${base.image_size}"):
                    raise ValueError("CXR-CLIP evaluation resize and crop must both be 224 pixels")
                allowed = {"size", "interpolation", "antialias"} if operation == "Resize" else {"size"}
                if set(values).difference(allowed):
                    raise ValueError("Unsupported CXR-CLIP evaluation transform setting")
                if operation == "Resize":
                    if values.get("interpolation", "bilinear") not in ("bilinear", 2):
                        raise ValueError("CXR-CLIP evaluation requires bilinear interpolation")
                    if values.get("antialias", True) is not True:
                        raise ValueError("CXR-CLIP evaluation uses antialiased PIL resizing")
    for partition in ("data_train", "data_valid", "data_test"):
        datasets = training_config.get(partition)
        if isinstance(datasets, Mapping):
            for settings in datasets.values():
                if isinstance(settings, Mapping) and settings.get("normalize", "huggingface") != "huggingface":
                    raise ValueError("CXR-CLIP Swin-Tiny requires the released 0.5/0.5 normalization")
    spec = PreprocessSpec(
        image_size=224, mean=(0.5,) * 3, std=(0.5,) * 3,
        resize_mode="shortest_edge_center_crop", resize_shortest_edge=224,
        interpolation="bilinear", do_hist_equalization=False,
    )
    return spec, "checkpoint_eval_transforms_and_verified_release" if recorded else "verified_official_swin_eval_protocol"


def _visual_state(payload):
    state = payload.get("model")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("CXR-CLIP checkpoint is missing its pretrained model state")
    result = {}
    prefix = "image_encoder.image_encoder."
    for name, value in state.items():
        if not isinstance(name, str):
            raise ValueError("CXR-CLIP checkpoint parameter names must be strings")
        if name.startswith("module."):
            raise ValueError("CXR-CLIP release weights must use the official unwrapped parameter layout")
        if not name.startswith("image_encoder."):
            continue
        if not name.startswith(prefix) or not isinstance(value, torch.Tensor):
            raise ValueError("CXR-CLIP visual weights do not match the official Hugging Face wrapper")
        mapped = name[len(prefix):]
        if mapped in result:
            raise ValueError("Duplicate CXR-CLIP visual parameter after normalization")
        result[mapped] = value
    if not result:
        raise ValueError("CXR-CLIP checkpoint contains no visual encoder parameters")
    return result


def build_cxr_clip(config):
    """Build the fixed Swin-Tiny MCC release from a local official checkpoint."""
    if not config.get("checkpoint"):
        raise ValueError("CXR-CLIP requires the local official swint_mcc.tar checkpoint")
    checkpoint = Path(config["checkpoint"]).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError("The configured CXR-CLIP checkpoint is unavailable")
    source = config.get("checkpoint_source", OFFICIAL_CHECKPOINT_URL)
    if source not in {OFFICIAL_CHECKPOINT_URL, "Soombit-ai/cxr-clip/swint_mcc.tar", "swint_mcc.tar"}:
        raise ValueError("CXR-CLIP checkpoint_source must identify the fixed official Swin-Tiny MCC release")
    try:
        from transformers import SwinConfig, SwinModel
    except ImportError as error:
        raise ImportError("CXR-CLIP requires Transformers with built-in SwinModel support") from error
    from .common import load_checkpoint_payload
    checkpoint_sha256 = sha256_file(checkpoint)
    payload = load_checkpoint_payload(checkpoint, allow_omegaconf_metadata=True)
    training_config = _training_config(payload)
    preprocess, preprocess_source = _preprocess(training_config)
    encoder = SwinModel(SwinConfig(**OFFICIAL_ARCHITECTURE), add_pooling_layer=True)
    state = _visual_state(payload)
    expected_state = encoder.state_dict()
    for name, tensor in state.items():
        expected = expected_state.get(name)
        if expected is not None and expected.is_floating_point():
            if not tensor.is_floating_point() or not torch.isfinite(tensor).all().item():
                raise ValueError("CXR-CLIP visual parameters must contain finite floating-point weights")
        elif expected is not None and tensor.dtype != expected.dtype:
            raise ValueError("CXR-CLIP visual buffer dtype differs from the released architecture")
    try:
        encoder.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError("Refusing incomplete or incompatible CXR-CLIP Swin-Tiny visual weights") from error
    if sha256_file(checkpoint) != checkpoint_sha256:
        raise ValueError("CXR-CLIP checkpoint changed during loading; retry with a stable local release file")
    provenance = {
        "model": "cxr_clip", "variant": "swin_tiny_mcc",
        "official_repository": OFFICIAL_REPOSITORY,
        "verified_source_revision": VERIFIED_SOURCE_REVISION,
        "checkpoint_source": OFFICIAL_CHECKPOINT_URL,
        "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha256,
        "image_encoder_source": OFFICIAL_IMAGE_MODEL_ID,
        "architecture": dict(OFFICIAL_ARCHITECTURE),
        "checkpoint_visual_prefix": "image_encoder.image_encoder.",
        "preprocessing_source": preprocess_source,
        "preprocessing_implementation": "shared_dicom_display_then_cxr_clip_swin_eval_v1",
        "global_pooling": "official_first_final_spatial_token",
        "has_cls_token": False,
        "global_pooling_note": "The official CXR-CLIP Swin code uses tokens[:, 0]; Swin has no CLS token.",
        "attention_implementation": "transformers_swin_eager",
        "pretraining_dataset_variant": "MIMIC-CXR+CheXpert+ChestX-ray14",
        "checkpoint_training_datasets_recorded": "data_train" in training_config,
        "checkpoint_identity_note": "Local weights must be obtained from the official Swin-Tiny MCC release.",
    }
    return CXRClipAdapter(encoder, preprocess, provenance)
