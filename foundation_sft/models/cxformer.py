"""Official CXformer-small/base Hugging Face register-token encoders.

The released model identifiers are documented in m42-health/CXformer/Readme.md
and hf_predict.py. Load local snapshots; never execute checkpoint-provided code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common import BackboneAdapter, sha256_file
from .hf_dino import HFDinoAdapter, _processor_spec


OFFICIAL_REPOSITORY = "https://github.com/m42-health/CXformer"
VERIFIED_SOURCE_REVISION = "da740db15a6d16cc06a26b82bd78527f2616dd7e"
OFFICIAL_RELEASES = {
    "m42-health/CXformer-small": {"hidden_size": 384, "num_attention_heads": 6},
    "m42-health/CXformer-base": {"hidden_size": 768, "num_attention_heads": 12},
}


def build_cxformer(config: dict[str, Any]) -> BackboneAdapter:
    if not config.get("model_dir"):
        raise ValueError(
            "CXformer requires model_dir containing its official local HF snapshot "
            "(config, processor and weights); automatic downloading is disabled"
        )
    model_dir = Path(config["model_dir"]).expanduser().resolve()
    source = config.get("checkpoint_source") or config.get("source")
    if not source:
        variant = config.get("variant", "base")
        if variant not in {"small", "base"}:
            raise ValueError("CXformer variant must be 'small' or 'base'")
        source = f"m42-health/CXformer-{variant}"
    if source not in OFFICIAL_RELEASES:
        raise ValueError(
            "checkpoint_source must identify the official CXformer-small or "
            "CXformer-base release, rather than a generic DINOv2 model"
        )
    for filename in ("config.json", "preprocessor_config.json"):
        if not (model_dir / filename).is_file():
            raise FileNotFoundError(f"Required CXformer snapshot file missing: {model_dir / filename}")
    raw_config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    expected = {
        "model_type": "dinov2_with_registers",
        "patch_size": 14,
        "num_register_tokens": 4,
        "num_hidden_layers": 12,
        **OFFICIAL_RELEASES[source],
    }
    differences = {key: (raw_config.get(key), value) for key, value in expected.items() if raw_config.get(key) != value}
    if differences:
        raise ValueError(f"The supplied CXformer archive differs from its released architecture: {differences}")
    try:
        from transformers import AutoConfig, AutoImageProcessor, Dinov2WithRegistersModel
    except ImportError as error:
        raise ImportError(
            "CXformer requires a compatible Transformers release with "
            "Dinov2WithRegistersModel support; keep working PyTorch/CUDA packages"
        ) from error
    options = {"local_files_only": True, "trust_remote_code": False}
    encoder_config = AutoConfig.from_pretrained(str(model_dir), **options)
    processor = AutoImageProcessor.from_pretrained(str(model_dir), use_fast=False, **options)
    preprocess = _processor_spec(processor, encoder_config)
    # Avoid AutoModel's unrelated timm model registry. The published CXformer
    # snapshots use the built-in register-token DINOv2 architecture explicitly.
    encoder, loading_info = Dinov2WithRegistersModel.from_pretrained(
        str(model_dir),
        config=encoder_config,
        output_loading_info=True,
        attn_implementation="eager",
        weights_only=True,
        **options,
    )
    problems = {
        key: loading_info.get(key, [])
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        if loading_info.get(key)
    }
    if problems:
        raise ValueError(f"Refusing incompletely initialized CXformer weights: {problems}")
    provenance = {
        "model": "cxformer",
        "official_repository": OFFICIAL_REPOSITORY,
        "verified_source_revision": VERIFIED_SOURCE_REVISION,
        "checkpoint_source": source,
        "model_dir": str(model_dir),
        "model_type": raw_config["model_type"],
        "config_sha256": sha256_file(model_dir / "config.json"),
        "preprocessor_sha256": sha256_file(model_dir / "preprocessor_config.json"),
        "weights_sha256": {
            path.name: sha256_file(path)
            for path in sorted(model_dir.glob("*.safetensors"))
            + sorted(model_dir.glob("pytorch_model*.bin"))
        },
        "attention_implementation": "eager",
        "checkpoint_identity_note": "Local weights must be obtained from this official release.",
    }
    return HFDinoAdapter(encoder, preprocess, provenance)
