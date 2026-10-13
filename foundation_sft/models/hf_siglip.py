"""Offline XraySigLIP vision tower from the actual Transformers release.

The published safetensors contains ``vision_model.*`` and ``text_model.*``.
Only the vision tensors are read, preserving the released attention pool.
Local official model/processor JSONs are required; no text tower, downloads,
remote code, inferred activation, or generic pretrained substitute is used.
"""

from dataclasses import asdict
import copy
import importlib.metadata
import json
import math
from pathlib import Path
import re

import torch
from torch import nn
from torch.nn import functional as F

from .common import BackboneAdapter, BackboneFeatures, sha256_file
from .hf_dino import _processor_spec


SOURCE = "StanfordAIMI/XraySigLIP__vit-b-16-siglip-512__webli"
TRANSFORMERS_VERSION = "4.48.3"
_RELEASE_SHAPE = {
    "image_size": 512, "patch_size": 16, "hidden_size": 768,
    "intermediate_size": 3072, "num_hidden_layers": 12,
    "num_attention_heads": 12, "num_channels": 3,
}


def _json_object(path):
    if not path.is_file():
        raise FileNotFoundError(f"Local official XraySigLIP {path.name} is missing: {path}; "
                                "preserve the existing model.safetensors and add this small JSON "
                                "from the same official release")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path.name}")
    return payload


def _vision_config(payload):
    from transformers import SiglipVisionConfig

    if payload.get("model_type") == "siglip":
        raw = payload.get("vision_config")
    elif payload.get("model_type") == "siglip_vision_model":
        raw = payload
    else:
        raise ValueError("XraySigLIP requires the official Transformers SigLIP config.json; "
                         "an OpenCLIP config or unrelated model is not interchangeable")
    if not isinstance(raw, dict):
        raise ValueError("Official SigLIP config.json must contain a vision_config object")
    if raw.get("model_type", "siglip_vision_model") != "siglip_vision_model":
        raise ValueError("Unsupported SigLIP vision model_type")
    # PretrainedConfig.from_dict adds attention metadata to its input in this
    # Transformers release. Keep the author's raw JSON untouched in provenance.
    vision = SiglipVisionConfig.from_dict(copy.deepcopy(raw))
    differences = {name: (getattr(vision, name), wanted)
                   for name, wanted in _RELEASE_SHAPE.items()
                   if isinstance(getattr(vision, name), bool) or getattr(vision, name) != wanted}
    if differences:
        raise ValueError(f"XraySigLIP config differs from the B/16 512 release: {differences}")
    if getattr(vision, "vision_use_head", True) is not True:
        raise ValueError("XraySigLIP requires the released multihead attention pooling head")
    if not math.isfinite(vision.layer_norm_eps) or vision.layer_norm_eps <= 0:
        raise ValueError("SigLIP layer_norm_eps must be positive and finite")
    if not math.isfinite(vision.attention_dropout) or not 0 <= vision.attention_dropout < 1:
        raise ValueError("SigLIP attention_dropout must be finite and in [0,1)")
    # Specify the concrete supported implementation, never a checkpoint-supplied
    # flash/remote backend. This is also compatible with PyTorch 2.0.1.
    vision._attn_implementation = "eager"
    defaults = SiglipVisionConfig().to_dict()
    inherited = {name: value for name, value in defaults.items()
                 if name not in raw and not name.startswith("_")}
    return vision, raw, inherited


def _preprocess(payload, vision):
    from transformers import SiglipImageProcessor

    processor_type = payload.get("image_processor_type")
    if processor_type not in {None, "SiglipImageProcessor"}:
        raise ValueError("XraySigLIP requires SiglipImageProcessor metadata")
    # Metadata cannot request additional crops or padding that the fixed shared
    # dataset does not implement. Defaults otherwise come from the pinned class.
    for option in ("do_center_crop", "do_pad", "do_hist_equalization"):
        if payload.get(option, False) is not False:
            raise ValueError(f"Unsupported SigLIP preprocessing option {option}")
    for option in ("do_resize", "do_rescale", "do_normalize", "do_convert_rgb"):
        if option in payload and payload[option] is not None and not isinstance(payload[option], bool):
            raise ValueError(f"SigLIP {option} must be a boolean")
    processor = SiglipImageProcessor.from_dict(payload)
    spec = _processor_spec(processor, vision)
    if spec.image_size != vision.image_size or spec.resize_mode != "resize":
        raise ValueError("SigLIP processor must resize directly to the released 512 square input")
    defaults = SiglipImageProcessor().to_dict()
    inherited = {name: value for name, value in defaults.items() if name not in payload}
    return spec, {
        "published": payload, "effective_processor": processor.to_dict(),
        "inherited_library_defaults": inherited,
        "defaults_library_version": TRANSFORMERS_VERSION,
        "shared_preprocess": asdict(spec),
        "defaults_source": "https://github.com/huggingface/transformers/blob/v4.48.3/"
                           "src/transformers/models/siglip/image_processing_siglip.py",
        "pixel_path": "shared_float32_DICOM_display_resize_then_repeat_RGB",
        "pixel_path_note": "Official processor geometry and normalization are retained. "
                           "Shared float DICOM resizing differs from native uint8 RGB resizing; "
                           "[0,1] DICOM pixels are not rescaled by 1/255 again.",
    }


def _checkpoint_path(directory, config):
    path = Path(config.get("checkpoint") or "model.safetensors").expanduser()
    path = path if path.is_absolute() else directory / path
    if not path.is_file():
        raise FileNotFoundError(f"Local XraySigLIP checkpoint is missing: {path}; "
                                "set checkpoint to model.safetensors in xray_siglip.json")
    if path.suffix != ".safetensors":
        raise ValueError("This XraySigLIP release requires the local model.safetensors checkpoint")
    return path.resolve()


def _load_visual_weights(encoder, path):
    from safetensors import safe_open

    expected = encoder.state_dict()
    state, text_keys, scalars = {}, [], []
    with safe_open(str(path), framework="pt", device="cpu") as archive:
        keys = set(archive.keys())
        visual_keys = {key for key in keys if key.startswith("vision_model.")}
        missing, unexpected = set(expected) - visual_keys, visual_keys - set(expected)
        if missing or unexpected:
            raise ValueError(f"SigLIP visual weights mismatch: missing={sorted(missing)[:20]}, "
                             f"unexpected={sorted(unexpected)[:20]}")
        for name in sorted(keys - visual_keys):
            if name.startswith("text_model."):
                text_keys.append(name)
            elif name in {"logit_scale", "logit_bias"} and math.prod(
                    archive.get_slice(name).get_shape()) == 1:
                scalars.append(name)
            else:
                raise ValueError(f"Unrecognized key outside the SigLIP vision/text towers: {name}")
        for name in sorted(visual_keys):
            if tuple(archive.get_slice(name).get_shape()) != tuple(expected[name].shape):
                raise ValueError(f"SigLIP visual weight shape mismatch: {name}")
            tensor = archive.get_tensor(name)
            if not tensor.is_floating_point():
                raise ValueError(f"Non-floating SigLIP visual weight: {name}")
            if not torch.isfinite(tensor).all().item():
                raise ValueError(f"Non-finite SigLIP visual weight: {name}")
            state[name] = tensor
    encoder.load_state_dict(state, strict=True)
    return {"loaded_visual_key_count": len(state), "ignored_text_key_count": len(text_keys),
            "ignored_contrastive_scalars": scalars,
            "source_visual_dtypes": sorted({str(tensor.dtype) for tensor in state.values()})}


class HFSiglipAdapter(BackboneAdapter):
    def __init__(self, encoder, preprocess, provenance, normalize=False):
        super().__init__()
        if not isinstance(normalize, bool):
            raise ValueError("normalize_image_embedding must be a boolean")
        self.encoder = encoder
        self.preprocess = preprocess
        self.provenance = dict(provenance)
        self.feature_dim = int(encoder.config.hidden_size)
        self.patch_size = int(encoder.config.patch_size)
        self.normalize = normalize
        if preprocess.image_size != encoder.config.image_size:
            raise ValueError("SigLIP input resolution differs from its local vision config")
        if not getattr(encoder.vision_model, "use_head", False):
            raise ValueError("SigLIP must retain the pretrained attention pool")

    def forward_features(self, images):
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected SigLIP images shaped [B,3,H,W]")
        size = self.preprocess.image_size
        if tuple(images.shape[-2:]) != (size, size):
            raise ValueError("SigLIP input must use its published resolution")
        outputs = self.encoder(pixel_values=images, return_dict=True,
                               interpolate_pos_encoding=False)
        tokens, global_features = outputs.last_hidden_state, outputs.pooler_output
        grid = (size // self.patch_size,) * 2
        if tokens.ndim != 3 or tokens.shape[1] != math.prod(grid):
            raise RuntimeError("SigLIP spatial token count differs from its patch grid")
        if self.normalize:
            global_features = F.normalize(global_features, dim=-1)
            tokens = F.normalize(tokens, dim=-1)
        return BackboneFeatures(global_features, tokens, grid).validate()

    def lora_target_names(self):
        pattern = re.compile(r"^encoder\.vision_model\.encoder\.layers\.\d+\.self_attn\.(q|k|v)_proj$")
        names = [name for name, module in self.named_modules()
                 if isinstance(module, nn.Linear) and pattern.fullmatch(name)]
        if len(names) != self.encoder.config.num_hidden_layers * 3:
            raise RuntimeError("Could not identify every SigLIP visual Q/K/V projection")
        return names


def build_hf_siglip(config):
    if config.get("model") != "xray_siglip":
        raise ValueError("HF SigLIP adapter is for the official XraySigLIP release")
    if config.get("checkpoint_source", SOURCE) != SOURCE:
        raise ValueError(f"Use the actual official medical release {SOURCE!r}")
    if not config.get("model_dir"):
        raise ValueError("XraySigLIP requires its local model_dir snapshot")
    directory = Path(config["model_dir"]).expanduser().resolve()
    config_path, processor_path = directory / "config.json", directory / "preprocessor_config.json"
    payload, processor_payload = _json_object(config_path), _json_object(processor_path)
    if importlib.metadata.version("transformers") != TRANSFORMERS_VERSION:
        raise ValueError("XraySigLIP requires the verified transformers==4.48.3 defaults")
    from transformers import SiglipVisionModel

    vision, raw_vision, inherited = _vision_config(payload)
    preprocess, preprocessing = _preprocess(processor_payload, vision)
    normalize = config.get("normalize_image_embedding", False)
    if not isinstance(normalize, bool):
        raise ValueError("normalize_image_embedding must be a boolean")
    path = _checkpoint_path(directory, config)
    digest = sha256_file(path)
    expected = config.get("expected_checkpoint_sha256")
    if expected is not None:
        if not isinstance(expected, str) or re.fullmatch(r"[0-9a-fA-F]{64}", expected) is None:
            raise ValueError("expected_checkpoint_sha256 must be a 64-character hexadecimal SHA-256")
        if expected.lower() != digest:
            raise ValueError("Local XraySigLIP checkpoint does not match expected SHA-256")
    encoder = SiglipVisionModel(vision)
    loading = _load_visual_weights(encoder, path)
    provenance = {
        "model": "xray_siglip", "checkpoint_source": SOURCE, "model_dir": str(directory),
        "checkpoint": str(path), "checkpoint_sha256": digest,
        "config_sha256": sha256_file(config_path),
        "preprocessor_config_sha256": sha256_file(processor_path),
        "checkpoint_format": "Transformers SigLIP vision_model.* / text_model.*",
        "vision_cfg": vision.to_dict(), "raw_vision_cfg": raw_vision,
        "inherited_vision_defaults": inherited,
        "vision_defaults_library_version": TRANSFORMERS_VERSION,
        "vision_defaults_source": "https://github.com/huggingface/transformers/blob/v4.48.3/"
                                  "src/transformers/models/siglip/configuration_siglip.py",
        "image_embedding_dim": vision.hidden_size, "preprocessing": preprocessing,
        "embedding_normalization": "l2_global_and_local" if normalize else "none",
        "global_features": "released SigLIP multihead attention pooler_output",
        "local_features": "post_layernorm spatial patch sequence, no CLS",
        "lora_attention_scope": "visual encoder Q/K/V; pretrained attention pooling remains frozen",
        "text_tower_constructed": False, "text_weights_loaded": False,
        "attention_implementation": "eager", "library_versions": {"transformers": TRANSFORMERS_VERSION},
        "checkpoint_identity_note": "Source ID declares the medical release; local SHA-256 identifies "
                                    "the artifact. No generic pretrained weights are substituted.",
        **loading,
    }
    return HFSiglipAdapter(encoder, preprocess, provenance, normalize=normalize)
