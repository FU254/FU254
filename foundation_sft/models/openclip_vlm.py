"""Local visual-only adapter for the released OpenCLIP-format BiomedCLIP.

Constructing OpenCLIP's complete model would also construct/download a text
encoder for BiomedCLIP. This module builds only its TimmModel visual tower and
requires a local release config plus a complete, matching visual state dict.
"""

from dataclasses import asdict
import importlib.metadata
import json
import math
from pathlib import Path
import re

import torch
from torch import nn
from torch.nn import functional as F

from .common import (BackboneAdapter, BackboneFeatures, PreprocessSpec,
                     load_checkpoint, sha256_file)


RELEASES = {
    "biomedclip": {
        "source": "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "image_size": 224,
        "embed_dim": 512,
        "timm_model_name": "vit_base_patch16_224",
        "timm_pool": "token",
        "timm_proj": "linear",
        "prefix_tokens": 1,
    },
}

# Fields accepted by the pinned OpenCLIP CLIPVisionCfg. Its timm branch
# forwards only the timm options plus image_size/patch_dropout; native CLIP
# architecture fields are intentionally superseded by timm_model_name.
_NATIVE_CLIP_FIELDS = {
    "layers", "width", "head_width", "mlp_ratio", "patch_size", "ls_init_value",
    "attentional_pool", "attn_pooler_queries", "attn_pooler_heads", "no_ln_pre",
    "pos_embed_type", "final_ln_after_pool", "pool_type", "output_tokens",
    "act_kwargs", "norm_kwargs",
}
_TIMM_FIELDS = {"image_size", "timm_model_name", "timm_model_pretrained", "timm_pool", "timm_proj",
                "timm_proj_bias", "timm_drop", "timm_drop_path", "patch_dropout"}


def _json_object(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path.name}")
    return data


def _square_size(value):
    if isinstance(value, (list, tuple)):
        if len(value) == 3 and value[0] == 3:
            value = value[1:]
        if len(value) != 2 or value[0] != value[1]:
            raise ValueError("OpenCLIP visual input size must be square")
        value = value[0]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("OpenCLIP image_size must be a positive integer")
    return value


def _release_config(config, payload):
    model_name = config.get("model")
    if model_name not in RELEASES:
        raise ValueError("OpenCLIP adapter model must be biomedclip; XraySigLIP uses its native HF adapter")
    release = RELEASES[model_name]
    source = config.get("checkpoint_source") or release["source"]
    if source != release["source"]:
        raise ValueError(f"Use the actual official medical release {release['source']!r}")
    model = payload.get("model_cfg")
    vision = model.get("vision_cfg") if isinstance(model, dict) else None
    if not isinstance(vision, dict):
        raise ValueError("Local open_clip_config.json must contain model_cfg.vision_cfg")
    if set(vision) - (_NATIVE_CLIP_FIELDS | _TIMM_FIELDS):
        raise ValueError(f"Unsupported OpenCLIP visual config fields: "
                         f"{sorted(set(vision) - (_NATIVE_CLIP_FIELDS | _TIMM_FIELDS))}")
    try:
        version = importlib.metadata.version("open_clip_torch")
        if version != "2.23.0":
            raise ValueError("OpenCLIP visual config requires the verified open_clip_torch==2.23.0 defaults")
        from open_clip.model import CLIPVisionCfg
    except importlib.metadata.PackageNotFoundError as error:
        raise ImportError("Use the isolated open_clip_torch==2.23.0 / timm==0.9.12 environment") from error
    # The official factory first instantiates CLIPVisionCfg. BiomedCLIP can
    # omit image_size (224) or timm_proj (linear), and an empty timm_pool keeps
    # the native ViT's token pooling instead of replacing it with another pool.
    vision = asdict(CLIPVisionCfg(**vision))
    expected = {key: release[key] for key in ("timm_model_name", "timm_proj")}
    differences = {key: (vision.get(key), wanted) for key, wanted in expected.items()
                   if vision.get(key) != wanted}
    allowed_pool = {"", "token"}
    if not isinstance(vision["timm_pool"], str) or vision["timm_pool"] not in allowed_pool:
        differences["timm_pool"] = (vision["timm_pool"], sorted(allowed_pool))
    if _square_size(vision.get("image_size")) != release["image_size"]:
        differences["image_size"] = (vision.get("image_size"), release["image_size"])
    if model.get("embed_dim") != release["embed_dim"]:
        differences["embed_dim"] = (model.get("embed_dim"), release["embed_dim"])
    if differences:
        raise ValueError(f"OpenCLIP config differs from the released visual architecture: {differences}")
    if vision.get("timm_model_pretrained", False) is not False:
        raise ValueError("Loading timm's generic pretrained weights is prohibited; use the local medical weights")
    if vision.get("patch_dropout", 0.0) != 0:
        raise ValueError("Nonzero patch_dropout is incompatible with the complete spatial patch-token contract")
    return release, vision


def _preprocess(payload, release, config):
    published = payload.get("preprocess_cfg")
    if not isinstance(published, dict):
        raise ValueError("Local official snapshot must contain a preprocess_cfg JSON object")
    try:
        version = importlib.metadata.version("open_clip_torch")
        if version != "2.23.0":
            raise ValueError("OpenCLIP preprocessing requires the verified open_clip_torch==2.23.0 defaults")
        from open_clip.transform import PreprocessCfg, merge_preprocess_dict
    except importlib.metadata.PackageNotFoundError as error:
        raise ImportError("Use the isolated open_clip_torch==2.23.0 / timm==0.9.12 environment") from error
    defaults = asdict(PreprocessCfg())
    # Match the official HF factory: library defaults, published overlay, then
    # the visual tower's image size. This does not construct the text encoder.
    settings = merge_preprocess_dict(defaults, published)
    override = config.get("preprocess_override", {})
    if not isinstance(override, dict):
        raise ValueError("preprocess_override must be a JSON object")
    allowed = {"mean", "std", "resize_mode", "interpolation"}
    if set(override) - allowed:
        raise ValueError(f"Unsupported preprocess override fields: {sorted(set(override) - allowed)}")
    if override and not str(config.get("preprocess_override_note") or "").strip():
        raise ValueError("Missing published metadata requires an explicit preprocess_override_note research choice")
    for key, value in override.items():
        if published.get(key) is not None and published[key] != value:
            raise ValueError(f"Do not override published OpenCLIP preprocessing field {key}")
        settings[key] = value
    for key in ("size", "image_size"):
        if published.get(key) is not None and _square_size(published[key]) != release["image_size"]:
            raise ValueError("Published preprocessing and visual encoder sizes differ")
    settings["size"] = release["image_size"]
    if settings["mode"] != "RGB":
        raise ValueError("The shared OpenCLIP image pipeline requires published RGB input")
    mode = settings["resize_mode"]
    if mode not in {"squash", "shortest"}:
        raise ValueError("Supported OpenCLIP resize modes are squash and shortest; longest/padding needs a separate adapter")
    interpolation = settings["interpolation"]
    if interpolation not in {"bicubic", "bilinear", "nearest"}:
        raise ValueError("Unsupported OpenCLIP image interpolation")
    mean, std = settings["mean"], settings["std"]
    if not isinstance(mean, (list, tuple)) or not isinstance(std, (list, tuple)):
        raise ValueError("OpenCLIP mean/std must contain three numeric channels")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           for value in tuple(mean) + tuple(std)):
        raise ValueError("OpenCLIP mean/std must contain finite numeric channels")
    spec = PreprocessSpec(
        image_size=release["image_size"], mean=tuple(mean), std=tuple(std),
        resize_mode="resize" if mode == "squash" else "shortest_edge_center_crop",
        resize_shortest_edge=release["image_size"] if mode == "shortest" else None,
        interpolation=interpolation,
    )
    inherited = {key: value for key, value in defaults.items()
                 if published.get(key) is None and key not in override and key != "size"}
    return spec, {"published": published, "explicit_missing_metadata": override,
                  "research_choice_note": config.get("preprocess_override_note"),
                  "inherited_library_defaults": inherited, "defaults_library_version": version,
                  "defaults_source": "https://github.com/mlfoundations/open_clip/blob/v2.23.0/"
                                     "src/open_clip/factory.py",
                  "effective_openclip_preprocess": settings,
                  "ignored_published_fields": sorted(set(published) - set(defaults)),
                  "shared_preprocess": asdict(spec),
                  "pixel_path": "shared_float32_DICOM_display_resize_then_repeat_RGB",
                  "pixel_path_note": "Official statistics and geometry; shared float DICOM resizing "
                                     "is not bit-identical to OpenCLIP PIL uint8 RGB preprocessing."}


def _create_visual_tower(vision, embed_dim):
    try:
        from open_clip.timm_model import TimmModel
    except ImportError as error:
        raise ImportError("Use a separate environment with open_clip_torch==2.23.0 and timm==0.9.12; "
                          "keep the working EVA-X environment unchanged") from error
    options = {
        "model_name": vision["timm_model_name"], "embed_dim": embed_dim,
        "image_size": _square_size(vision["image_size"]),
        "pool": vision["timm_pool"], "proj": vision["timm_proj"],
        "proj_bias": vision.get("timm_proj_bias", False),
        "drop": vision.get("timm_drop", 0.0),
        "drop_path": vision.get("timm_drop_path"),
        "patch_drop": vision.get("patch_dropout") or None,
        "pretrained": False,
    }
    if not isinstance(options["proj_bias"], bool):
        raise ValueError("timm_proj_bias must be a boolean")
    for key in ("drop", "drop_path", "patch_drop"):
        value = options[key]
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not math.isfinite(value) or not 0 <= value < 1):
            raise ValueError(f"Invalid OpenCLIP visual {key}")
    return TimmModel(**options)


def _weights_path(directory, config):
    explicit = config.get("checkpoint")
    if explicit:
        path = Path(explicit).expanduser()
        path = path if path.is_absolute() else directory / path
    else:
        candidates = [directory / name for name in ("model.safetensors", "open_clip_pytorch_model.bin")
                      if (directory / name).is_file()]
        if len(candidates) != 1:
            raise ValueError("Provide exactly one official model.safetensors/open_clip_pytorch_model.bin "
                             "or explicitly select the local checkpoint")
        path = candidates[0]
    if not path.is_file():
        raise FileNotFoundError("Local OpenCLIP medical checkpoint is missing")
    if path.suffix not in {".safetensors", ".bin", ".pt", ".pth"}:
        raise ValueError("Unsupported local OpenCLIP checkpoint format")
    return path.resolve()


def _load_visual_weights(visual, path):
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(str(path), device="cpu")
    else:
        state = load_checkpoint(path)
    visual_state, ignored_text, ignored_scalars = {}, [], []
    for name, value in state.items():
        if name.startswith("visual."):
            visual_state[name[len("visual."):]] = value
        elif name.startswith("text."):
            ignored_text.append(name)
        elif name in {"logit_scale", "logit_bias"} and value.numel() == 1:
            ignored_scalars.append(name)
        else:
            raise ValueError(f"Unrecognized key outside the released visual/text towers: {name}")
    expected = visual.state_dict()
    missing, unexpected = set(expected) - set(visual_state), set(visual_state) - set(expected)
    if missing or unexpected:
        raise ValueError(f"OpenCLIP visual weights mismatch: missing={sorted(missing)[:20]}, "
                         f"unexpected={sorted(unexpected)[:20]}")
    for name, tensor in visual_state.items():
        if tensor.shape != expected[name].shape:
            raise ValueError(f"OpenCLIP visual weight shape mismatch: {name}")
        if expected[name].is_floating_point() and not tensor.is_floating_point():
            raise ValueError(f"Non-floating OpenCLIP visual weight: {name}")
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"Non-finite OpenCLIP visual weight: {name}")
    visual.load_state_dict(visual_state, strict=True)
    return {"ignored_text_key_count": len(ignored_text), "ignored_contrastive_scalars": sorted(ignored_scalars),
            "source_visual_dtypes": sorted({str(tensor.dtype) for tensor in visual_state.values()})}


class OpenCLIPVisionAdapter(BackboneAdapter):
    def __init__(self, visual, preprocess, provenance, *, embed_dim, prefix_tokens, normalize=False):
        super().__init__()
        self.visual = visual
        self.preprocess = preprocess
        self.provenance = dict(provenance)
        self.feature_dim = embed_dim
        self.prefix_tokens = prefix_tokens
        self.normalize = normalize
        if prefix_tokens != 1:
            raise ValueError("BiomedCLIP requires its released single CLS prefix token")
        actual_prefix = getattr(visual.trunk, "num_prefix_tokens", None)
        if actual_prefix != prefix_tokens:
            raise ValueError("OpenCLIP visual prefix tokens differ from the verified release")
        if getattr(visual.trunk, "global_pool", None) != "token":
            raise ValueError("OpenCLIP native visual pooling differs from the verified release")
        patch = visual.trunk.patch_embed.patch_size
        self.patch_size = (patch, patch) if isinstance(patch, int) else tuple(patch)
        if len(self.patch_size) != 2 or min(self.patch_size) < 1:
            raise ValueError("OpenCLIP patch size must be a positive pair")

    def forward_features(self, images):
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected OpenCLIP images shaped [B,3,H,W]")
        if tuple(images.shape[-2:]) != (self.preprocess.image_size,) * 2:
            raise ValueError("OpenCLIP images must use the verified published resolution")
        grid = tuple(images.shape[-2 + axis] // self.patch_size[axis] for axis in (0, 1))
        tokens = self.visual.trunk.forward_features(images)
        if tokens.ndim != 3 or tokens.shape[1] != self.prefix_tokens + math.prod(grid):
            raise RuntimeError("OpenCLIP patch token count differs; patch dropout must be disabled for the shared spatial contract")
        pooled = self.visual.trunk.forward_head(tokens)
        global_features = self.visual.head(pooled)
        local_features = tokens[:, self.prefix_tokens:]
        projection = getattr(self.visual.head, "proj", None)
        if projection is not None:
            # Retain BiomedCLIP's released projection for both CLS/global and
            # local features instead of conflating its 512-D image embedding
            # with the 768-D unprojected patch-token representation.
            local_features = projection(local_features)
        if global_features.shape[-1] != self.feature_dim:
            raise RuntimeError("OpenCLIP global embedding width differs from the release config")
        if self.normalize:
            global_features = F.normalize(global_features, dim=-1)
            local_features = F.normalize(local_features, dim=-1)
        return BackboneFeatures(global_features, local_features, grid).validate()

    def lora_target_names(self):
        pattern = re.compile(r"^visual\.trunk\.blocks\.\d+\.attn\.qkv$")
        names = [name for name, module in self.named_modules()
                 if isinstance(module, nn.Linear) and pattern.fullmatch(name)]
        if len(names) != len(self.visual.trunk.blocks):
            raise RuntimeError("Could not identify every OpenCLIP visual block's fused QKV projection")
        return names


def build_openclip_vlm(config):
    if config.get("model") not in RELEASES:
        raise ValueError("OpenCLIP adapter model must be biomedclip; XraySigLIP uses its native HF adapter")
    if not config.get("model_dir"):
        raise ValueError("OpenCLIP medical models require a complete local model_dir snapshot")
    directory = Path(config["model_dir"]).expanduser().resolve()
    config_path = directory / "open_clip_config.json"
    if not config_path.is_file():
        raise FileNotFoundError("Local official open_clip_config.json is missing; no network downloads are attempted")
    payload = _json_object(config_path)
    release, vision = _release_config(config, payload)
    preprocess, preprocessing = _preprocess(payload, release, config)
    normalize = config.get("normalize_image_embedding", False)
    if not isinstance(normalize, bool):
        raise ValueError("normalize_image_embedding must be a boolean")
    path = _weights_path(directory, config)
    digest = sha256_file(path)
    expected_digest = config.get("expected_checkpoint_sha256")
    if expected_digest is not None:
        if not isinstance(expected_digest, str) or re.fullmatch(r"[0-9a-fA-F]{64}", expected_digest) is None:
            raise ValueError("expected_checkpoint_sha256 must be a 64-character hexadecimal SHA-256")
        if expected_digest.lower() != digest:
            raise ValueError("Local OpenCLIP checkpoint does not match the expected SHA-256")
    visual = _create_visual_tower(vision, release["embed_dim"])
    loading = _load_visual_weights(visual, path)
    versions = {}
    for distribution in ("open_clip_torch", "timm"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    provenance = {
        "model": config["model"], "checkpoint_source": release["source"],
        "model_dir": str(directory), "checkpoint": str(path), "checkpoint_sha256": digest,
        "config_sha256": sha256_file(config_path), "vision_cfg": vision,
        "raw_vision_cfg": payload["model_cfg"]["vision_cfg"],
        "inherited_vision_defaults": {key: value for key, value in vision.items()
                                      if key not in payload["model_cfg"]["vision_cfg"]},
        "vision_defaults_library_version": versions["open_clip_torch"],
        "vision_defaults_source": "https://github.com/mlfoundations/open_clip/blob/v2.23.0/"
                                  "src/open_clip/model.py",
        "native_clip_fields_ignored_by_official_timm_branch": {
            key: vision[key] for key in sorted(_NATIVE_CLIP_FIELDS & set(vision))},
        "image_embedding_dim": release["embed_dim"], "preprocessing": preprocessing,
        "embedding_normalization": "l2_global_and_local" if normalize else "none",
        "global_features": "released visual pooling and projection",
        "local_features": "spatial patch tokens with the same retained visual linear projection; no CLS",
        "lora_attention_scope": "visual transformer block fused QKV; released image projection remains frozen",
        "text_tower_constructed": False, "library_versions": versions, **loading,
        "checkpoint_identity_note": "Source ID declares the official medical release; local SHA-256 identifies "
                                    "the artifact. Do not substitute generic timm/OpenAI pretrained weights.",
    }
    return OpenCLIPVisionAdapter(visual, preprocess, provenance, embed_dim=release["embed_dim"],
                                prefix_tokens=release["prefix_tokens"], normalize=normalize)
