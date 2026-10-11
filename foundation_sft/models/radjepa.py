"""Local, strictly loaded RadJEPA ViT-B/14 encoders.

The authors publish a Hugging Face custom-code model, rather than a Python
implementation in their GitHub repository.  ``hf`` therefore loads a reviewed
local snapshot using their documented AutoModel API.  ``ijepa`` supports an
encoder-only export in the official I-JEPA state-dict format; it does not assume
that the Hugging Face weights have that format.

Sources: https://github.com/aidelab-iitbombay/RadJEPA (5f490ab58295a403ae6f29aa304db43a2aded423),
https://huggingface.co/AIDELab-IITBombay/RadJEPA, and
https://github.com/facebookresearch/ijepa (52c1ae95d05f743e000e8f10a1f3a79b10cff048).
"""

from contextlib import contextmanager
import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import torch
from torch import nn

from .common import (BackboneAdapter, BackboneFeatures, PreprocessSpec,
                     load_checkpoint, load_backbone_weights, sha256_file)


OFFICIAL_MODEL_ID = "AIDELab-IITBombay/RadJEPA"
OFFICIAL_REPOSITORY = "https://github.com/aidelab-iitbombay/RadJEPA"
IJEPA_REPOSITORY = "https://github.com/facebookresearch/ijepa"


def _pair(value):
    if isinstance(value, int):
        return (value, value)
    return tuple(value)


def _preprocessing(config):
    """Require recorded settings instead of guessing unverified normalization."""
    values = config.get("preprocessing", config)
    missing = [key for key in ("image_size", "mean", "std") if values.get(key) is None]
    if missing:
        checkpoint = Path(config.get("checkpoint") or "").expanduser().resolve()
        processor_file = checkpoint / "preprocessor_config.json"
        if config.get("backend", "hf") != "hf" or not processor_file.is_file():
            raise ValueError("RadJEPA requires verified preprocessing settings: " + ", ".join(missing)
                             + ". Supply a complete local preprocessor_config.json or record the "
                             "verified model-card settings in preprocessing.")
        processor = json.loads(processor_file.read_text(encoding="utf-8"))
        if not isinstance(processor, dict):
            raise ValueError("RadJEPA preprocessor_config.json must be a JSON object")
        required = {"do_resize", "size", "do_normalize", "do_rescale", "rescale_factor", "resample"}
        if processor.get("do_normalize"):
            required.update(("image_mean", "image_std"))
        if processor.get("do_center_crop"):
            required.add("crop_size")
        if absent := required.difference(processor):
            raise ValueError("RadJEPA local processor metadata is incomplete: " + ", ".join(sorted(absent))
                             + ". Record verified preprocessing explicitly; no model defaults are guessed.")
        # Parse saved metadata without importing or executing custom processor code.
        from .hf_dino import _processor_spec
        spec = _processor_spec(SimpleNamespace(**processor), SimpleNamespace(patch_size=14))
        for key in ("image_size", "mean", "std"):
            if values.get(key) is not None:
                value = tuple(values[key]) if key in {"mean", "std"} else int(values[key])
                if value != getattr(spec, key):
                    raise ValueError("Partial RadJEPA preprocessing conflicts with the local official "
                                     "processor; supply a complete explicit preprocessing override")
        if spec.image_size % 14:
            raise ValueError("RadJEPA ViT-B/14 processor image size must be divisible by 14")
        return spec
    size = int(values["image_size"])
    if size % 14:
        raise ValueError("RadJEPA ViT-B/14 image_size must be divisible by 14")
    return PreprocessSpec(size, tuple(values["mean"]), tuple(values["std"]),
                          resize_mode=values.get("resize_mode", "resize"),
                          resize_shortest_edge=values.get("resize_shortest_edge"),
                          interpolation=values.get("interpolation", "bicubic"))


def _is_encoder(module):
    patch = getattr(module, "patch_embed", None)
    projection = getattr(patch, "proj", None)
    blocks = getattr(module, "blocks", None)
    return (isinstance(projection, nn.Conv2d)
            and _pair(projection.kernel_size) == (14, 14)
            and projection.out_channels == 768
            and isinstance(blocks, (nn.ModuleList, nn.Sequential))
            and len(blocks) == 12)


def _select_encoder(model, encoder_path=None):
    if encoder_path is not None:
        encoder = model.get_submodule(encoder_path) if encoder_path else model
        if not _is_encoder(encoder):
            raise ValueError(f"Configured encoder_path {encoder_path!r} is not a RadJEPA ViT-B/14 encoder")
        return encoder, encoder_path
    matches = [(name, module) for name, module in model.named_modules() if _is_encoder(module)]
    if len(matches) != 1:
        raise ValueError("Expected exactly one ViT-B/14 encoder in the local RadJEPA model; "
                         "set encoder_path after reviewing the source. Predictor/teacher selection "
                         "is never inferred from checkpoint names.")
    return matches[0][1], matches[0][0]


def _prefix_count(encoder):
    value = getattr(encoder, "num_prefix_tokens", None)
    if value is not None:
        return int(value)
    count = 0
    for name in ("cls_token", "reg_token", "register_tokens"):
        token = getattr(encoder, name, None)
        if isinstance(token, torch.Tensor):
            count += token.shape[1]
    return count


class RadJEPAAdapter(BackboneAdapter):
    """Use the encoder's patch sequence; global features are mean patch pooling."""

    def __init__(self, encoder, preprocess, provenance=None):
        super().__init__()
        if not _is_encoder(encoder):
            raise ValueError("RadJEPA requires a ViT-B/14 encoder with 12 blocks and 768 channels")
        self.backbone = encoder
        self.feature_dim = 768
        self.preprocess = preprocess
        self.provenance = dict(provenance or {})
        self.num_prefix_tokens = _prefix_count(encoder)
        if self.num_prefix_tokens < 0:
            raise ValueError("Encoder prefix-token count must be nonnegative")
        self._targets = []
        for index, block in enumerate(encoder.blocks):
            projection = getattr(getattr(block, "attn", None), "qkv", None)
            if (not isinstance(projection, nn.Linear) or projection.in_features != 768
                    or projection.out_features != 3 * 768):
                raise ValueError(f"RadJEPA block {index} does not expose the verified fused QKV projection")
            self._targets.append(f"backbone.blocks.{index}.attn.qkv")

    def lora_target_names(self):
        return list(self._targets)

    def forward_features(self, images):
        if (images.ndim != 4 or images.shape[1] != 3
                or tuple(images.shape[-2:]) != (self.preprocess.image_size,) * 2):
            raise ValueError("RadJEPA input must be [B,3,image_size,image_size] using recorded preprocessing")
        # timm encoders expose forward_features; the official I-JEPA encoder's
        # forward directly returns normalized patch tokens and takes no CLS token.
        method = getattr(self.backbone, "forward_features", None)
        tokens = method(images) if callable(method) else self.backbone(images)
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
            raise ValueError("Reviewed RadJEPA encoder must return [B,N,768] patch-token features; "
                             "pooled classifier/predictor outputs cannot be used as spatial features")
        grid = (images.shape[-2] // 14, images.shape[-1] // 14)
        expected = grid[0] * grid[1]
        if tokens.shape != (images.shape[0], expected + self.num_prefix_tokens, self.feature_dim):
            raise ValueError(f"RadJEPA encoder returned {tuple(tokens.shape)}; expected "
                             f"[B,{expected + self.num_prefix_tokens},768]. Check preprocessing and encoder_path.")
        local = tokens[:, self.num_prefix_tokens:]
        return BackboneFeatures(local.mean(dim=1), local, grid).validate()


def _require_reviewed_code(config):
    if config.get("allow_remote_code") is not True:
        raise ValueError("RadJEPA uses the authors' custom Python code. Review the local source, then "
                         "set allow_remote_code=true. Only local files are loaded; nothing is downloaded.")


def _source_provenance(directory, expected_commit=None):
    sources = sorted(directory.rglob("*.py"))
    if not sources:
        raise ValueError(f"No reviewed Python source found in {directory}")
    result = {"source_sha256": {str(path.relative_to(directory)): sha256_file(path) for path in sources}}
    if (directory / ".git").exists():
        commit = subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"],
                                check=True, capture_output=True, text=True).stdout.strip()
        result["source_commit"] = commit
        if expected_commit is not None and commit != expected_commit:
            raise ValueError(f"Local source commit {commit} differs from expected {expected_commit}")
    elif expected_commit is not None:
        raise ValueError("expected_repo_commit was provided but local source has no Git metadata")
    return result


def _hf_weights(directory):
    config_file = directory / "config.json"
    if not config_file.is_file():
        raise FileNotFoundError(f"Local RadJEPA snapshot needs {config_file}")
    config = json.loads(config_file.read_text(encoding="utf-8"))
    auto_map = config.get("auto_map", {})
    if "AutoModel" not in auto_map:
        raise ValueError("RadJEPA snapshot lacks the authors' custom AutoModel mapping")
    for key in ("AutoConfig", "AutoModel"):
        reference = auto_map.get(key)
        if reference is None:
            continue
        if not isinstance(reference, str) or "--" in reference:
            raise ValueError("RadJEPA custom-code references must point inside the reviewed local snapshot")
        module_name, _, class_name = reference.rpartition(".")
        parts = module_name.split(".")
        if not class_name.isidentifier() or not all(part.isidentifier() for part in parts):
            raise ValueError(f"Invalid local custom-code reference for {key}: {reference!r}")
        module_file = directory.joinpath(*parts).with_suffix(".py").resolve()
        if not module_file.is_relative_to(directory) or not module_file.is_file():
            raise FileNotFoundError(f"Local custom-code module for {key} not found: {module_file}")
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = directory / name
        if index.is_file():
            mapping = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
            names = sorted(set(mapping.values()))
            if not names:
                raise ValueError(f"Empty weight map in {index}")
            break
    else:
        names = [name for name in ("model.safetensors", "pytorch_model.bin")
                 if (directory / name).is_file()]
        if len(names) > 1:
            raise ValueError("Snapshot contains two weight formats; keep the verified format to avoid ambiguity")
    if not names:
        raise FileNotFoundError("RadJEPA snapshot has no pretrained weights")
    weights = []
    for name in names:
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise ValueError(f"Missing or invalid local checkpoint shard {name!r}")
        weights.append(path)
    return weights


def _load_hf(config, preprocess):
    checkpoint = Path(config["checkpoint"]).expanduser().resolve()
    if not checkpoint.is_dir():
        raise ValueError("RadJEPA hf checkpoint must be a complete local Hugging Face snapshot directory")
    weights = _hf_weights(checkpoint)
    source = _source_provenance(checkpoint, config.get("expected_repo_commit"))
    try:
        from transformers import AutoModel
    except ImportError as error:
        raise ImportError("RadJEPA hf backend requires transformers and the authors' custom-code dependencies. "
                          "The official model card recommends timm==1.0.24; use an isolated environment "
                          "rather than changing EVA-X's timm==0.4.12.") from error
    loaded = AutoModel.from_pretrained(str(checkpoint), local_files_only=True,
                                      trust_remote_code=True, output_loading_info=True,
                                      ignore_mismatched_sizes=False)
    if not isinstance(loaded, tuple) or len(loaded) != 2:
        raise ValueError("Custom RadJEPA loader did not return loading information; strict pretrained verification failed")
    model, info = loaded
    bad = {key: info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
           if info.get(key)}
    if bad:
        raise ValueError(f"RadJEPA pretrained loading mismatch: {bad}")
    encoder, path = _select_encoder(model, config.get("encoder_path"))
    provenance = {"model_id": OFFICIAL_MODEL_ID, "repository": OFFICIAL_REPOSITORY,
                  "backend": "hf", "checkpoint": str(checkpoint), "encoder_path": path,
                  "weights_sha256": {file.name: sha256_file(file) for file in weights},
                  "pooling": "mean_patch_tokens", **source}
    processor_file = checkpoint / "preprocessor_config.json"
    if processor_file.is_file():
        provenance["preprocessor_sha256"] = sha256_file(processor_file)
    return RadJEPAAdapter(encoder, preprocess, provenance)


@contextmanager
def _ijepa_import_path(directory):
    """Avoid collisions with other repositories that also use a top-level src."""
    saved = {name: module for name, module in list(sys.modules.items())
             if name == "src" or name.startswith("src.")}
    for name in saved:
        del sys.modules[name]
    old_path = list(sys.path)
    sys.path.insert(0, str(directory))
    try:
        yield
    finally:
        sys.path[:] = old_path
        for name in list(sys.modules):
            if name == "src" or name.startswith("src."):
                del sys.modules[name]
        sys.modules.update(saved)


def _load_ijepa(config, preprocess):
    repo_value = config.get("local_repo_dir", config.get("repo_dir", config.get("localrepo_dir")))
    if not repo_value:
        raise ValueError("RadJEPA ijepa backend needs local_repo_dir pointing to official facebookresearch/ijepa source")
    directory = Path(repo_value).expanduser().resolve()
    module_file = directory / "src/models/vision_transformer.py"
    if not module_file.is_file():
        raise FileNotFoundError(f"Official I-JEPA encoder source not found: {module_file}")
    source = _source_provenance(directory, config.get("expected_repo_commit"))
    state_key = config.get("state_key")
    if not state_key:
        raise ValueError("RadJEPA ijepa backend requires explicit state_key='encoder' or 'target_encoder' "
                         "for a verified export; Hugging Face weights are not assumed to use this format")
    if state_key not in {"encoder", "target_encoder"}:
        raise ValueError("RadJEPA ijepa state_key must be encoder or target_encoder; predictor is not an image encoder")
    checkpoint = Path(config["checkpoint"]).expanduser().resolve()
    state = load_checkpoint(checkpoint, state_key=state_key)
    with _ijepa_import_path(directory):
        vit = importlib.import_module("src.models.vision_transformer")
        encoder = vit.vit_base(patch_size=14, img_size=[preprocess.image_size])
    # Exact loading also catches a wrong image size through pos_embed dimensions.
    load_backbone_weights(encoder, state, ignored_prefixes=())
    return RadJEPAAdapter(encoder, preprocess,
                          {"model_id": OFFICIAL_MODEL_ID, "repository": OFFICIAL_REPOSITORY,
                           "architecture_source": IJEPA_REPOSITORY, "backend": "ijepa",
                           "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
                           "state_key": state_key, "pooling": "mean_patch_tokens", **source})


def build_radjepa(config):
    if not config.get("checkpoint"):
        raise ValueError("RadJEPA requires a local pretrained checkpoint; random initialization is disabled")
    _require_reviewed_code(config)
    preprocess = _preprocessing(config)
    backend = config.get("backend", "hf")
    if backend == "hf":
        return _load_hf(config, preprocess)
    if backend == "ijepa":
        return _load_ijepa(config, preprocess)
    raise ValueError("RadJEPA backend must be hf (official local snapshot) or ijepa (verified encoder export)")
