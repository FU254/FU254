"""Release/config and strict checkpoint checks for the two verified adapters."""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from foundation_sft.models.chexfound import (
    _normalize_teacher_state,
    _read_architecture,
    build_chexfound,
)
from foundation_sft.models.cxformer import build_cxformer


class VendorCheckpointTests(unittest.TestCase):
    def test_teacher_chunk_mapping_drops_only_pretraining_heads(self):
        state = {
            "module.backbone.blocks.2.13.attn.qkv.weight": torch.randn(12, 4),
            "module.backbone.norm.weight": torch.ones(4),
            "module.dino_head.mlp.weight": torch.zeros(4, 4),
            "module.ibot_head.mlp.weight": torch.zeros(4, 4),
        }
        mapped = _normalize_teacher_state(state)
        self.assertEqual(set(mapped), {"blocks.13.attn.qkv.weight", "norm.weight"})
        torch.testing.assert_close(
            mapped["blocks.13.attn.qkv.weight"],
            state["module.backbone.blocks.2.13.attn.qkv.weight"],
        )
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            _normalize_teacher_state({
                "backbone.blocks.0.1.attn.qkv.weight": torch.zeros(3, 1),
                "backbone.blocks.1.attn.qkv.weight": torch.zeros(3, 1),
            })

    def test_chexfound_has_no_random_weight_fallback(self):
        with self.assertRaisesRegex(ValueError, "official checkout"):
            build_chexfound({})
        with self.assertRaisesRegex(FileNotFoundError, "checkpoint"):
            build_chexfound({"repo_dir": "/not/a/repository", "checkpoint": "/not/a/checkpoint"})

    def test_cxformer_rejects_generic_source_and_wrong_architecture(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with self.assertRaisesRegex(ValueError, "generic DINOv2"):
                build_cxformer({"model_dir": directory, "checkpoint_source": "facebook/dinov2-base"})
            (path / "config.json").write_text(json.dumps({"model_type": "dinov2"}))
            (path / "preprocessor_config.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "released architecture"):
                build_cxformer({"model_dir": directory})


if __name__ == "__main__":
    unittest.main()
