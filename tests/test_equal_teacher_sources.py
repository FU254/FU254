"""Declared source resolution and strict trained-teacher restoration."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from foundation_sft import equal_teacher_sources as sources


HAVE_RUNTIME = bool(importlib.util.find_spec("torch") and importlib.util.find_spec("pandas"))


class DeclaredTeacherSourceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name)
        self.source = deepcopy(sources.DEFAULT_TEACHER_SOURCES[0])
        self.source["suite"] = "declared_suite"
        self.suite = self.project / "declared_suite"
        self.log = self.suite / "logs" / "rad_dino_lora_seed42_train.log"
        self.receipt = self.suite / "receipts" / "rad_dino_lora_seed42_train.json"
        self.checkpoint = self.project / "selected_run" / "best.pt"
        self.checkpoint.parent.mkdir()
        self.checkpoint.write_bytes(b"source-test-only")

    def log_run(self, *directories):
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self.log.write_text("\n".join(f"Run directory: {directory}" for directory in directories),
                            encoding="utf-8")

    def write_receipt(self, **overrides):
        value = {"model": "rad_dino", "mode": "lora", "seed": 42,
                 "action": "train", "run_dir": str(self.checkpoint.parent)}
        value.update(overrides)
        self.receipt.parent.mkdir(parents=True, exist_ok=True)
        self.receipt.write_text(json.dumps(value), encoding="utf-8")

    def test_defaults_are_three_explicit_sources_in_confirmed_protocol(self):
        self.assertEqual(tuple(source["model"] for source in sources.DEFAULT_TEACHER_SOURCES),
                         ("rad_dino", "chexfound", "chexzero"))
        self.assertTrue(all(source["weight"] == 1 / 3 and source["mode"] == "lora"
                            and source["seed"] == 42 for source in sources.DEFAULT_TEACHER_SOURCES))
        self.assertIn("suite_20261013T024024Z_92ac596d", sources.DEFAULT_TEACHER_SOURCES[0]["suite"])
        self.assertIn("suite_20261015T045005Z_2162f5a1", sources.DEFAULT_TEACHER_SOURCES[2]["suite"])

    def test_resolves_only_exact_declared_job(self):
        self.log_run("selected_run", "selected_run")
        other = self.project / "other_suite" / "logs"
        other.mkdir(parents=True)
        (other / self.log.name).write_text("Run directory: newer_better_run", encoding="utf-8")
        self.assertEqual(sources.resolve_teacher_checkpoint(self.source, self.project), self.checkpoint)

    def test_explicit_checkpoint_overrides_suite_without_scanning_it(self):
        self.log_run("ambiguous_one", "ambiguous_two")
        self.source["checkpoint"] = "selected_run/best.pt"
        self.assertEqual(sources.resolve_teacher_checkpoint(self.source, self.project), self.checkpoint)

    def test_receipt_alone_and_agreeing_receipt_are_supported(self):
        self.write_receipt()
        self.assertEqual(sources.resolve_teacher_checkpoint(self.source, self.project), self.checkpoint)
        self.log_run(self.checkpoint.parent)
        self.assertEqual(sources.resolve_teacher_checkpoint(self.source, self.project), self.checkpoint)

    def test_ambiguous_log_is_not_resolved_by_scores_or_existence(self):
        self.log_run(self.checkpoint.parent, "missing_second_run")
        with self.assertRaisesRegex(ValueError, "unambiguous"):
            sources.resolve_teacher_checkpoint(self.source, self.project)

    def test_conflicting_receipt_and_log_require_explicit_source(self):
        self.log_run("another_run")
        self.write_receipt()
        with self.assertRaisesRegex(ValueError, "unambiguous"):
            sources.resolve_teacher_checkpoint(self.source, self.project)

    def test_receipt_cannot_belong_to_other_teacher(self):
        self.write_receipt(model="chexfound")
        with self.assertRaisesRegex(ValueError, "model/mode/seed"):
            sources.resolve_teacher_checkpoint(self.source, self.project)

    def test_receipt_must_be_training_success(self):
        self.write_receipt(action="evaluate")
        with self.assertRaisesRegex(ValueError, "successful training"):
            sources.resolve_teacher_checkpoint(self.source, self.project)

    def test_missing_declared_source_has_no_fallback(self):
        with self.assertRaisesRegex(FileNotFoundError, "no training log or receipt"):
            sources.resolve_teacher_checkpoint(self.source, self.project)
        self.log_run("missing_run")
        with self.assertRaisesRegex(FileNotFoundError, "checkpoint is missing"):
            sources.resolve_teacher_checkpoint(self.source, self.project)

    def test_protocol_does_not_silently_change_teacher_or_seed(self):
        self.source["checkpoint"] = str(self.checkpoint)
        for key, value in (("model", "eva_x"), ("mode", "linear_probe"),
                           ("seed", 43), ("seed", True), ("weight", 0.5)):
            with self.subTest(key=key, value=value):
                invalid = {**self.source, key: value}
                with self.assertRaises(ValueError):
                    sources.resolve_teacher_checkpoint(invalid, self.project)


@unittest.skipUnless(HAVE_RUNTIME, "Torch and pandas are required for restoration checks")
class RestoreSelectedTeacherTests(unittest.TestCase):
    def setUp(self):
        import pandas as pd
        import torch
        from torch import nn
        self.torch = torch
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name)
        self.manifest = self.project / "frozen.csv"
        self.manifest.write_text("frozen test manifest", encoding="utf-8")
        self.checkpoint = self.project / "best.pt"
        self.source = {**deepcopy(sources.DEFAULT_TEACHER_SOURCES[0]),
                       "checkpoint": str(self.checkpoint)}
        self.config = {"model": "rad_dino", "mode": "lora", "seed": 42, "device": "cuda:1",
                       "split_manifest": str(self.manifest), "split_seed": 42,
                       "split_ratios": [0.7, 0.1, 0.2], "data": {"label_column": "obstruction"},
                       "evaluation": {"bootstrap_samples": 100}, "cache_dir": None}
        self.split = pd.DataFrame({"sample_id": ["s1", "s2", "s3"],
                                   "patient_id": ["p1", "p2", "p3"],
                                   "path": ["one", "two", "three"],
                                   "label": [0, 1, 0], "split": ["train", "val", "test"]})
        self.split.attrs["split_metadata"] = {"_manifest_hash": "frozen-cohort",
                                               "_source_hash": "frozen-label-source"}
        self.restored_before_freeze = []

        class SmallTeacher(nn.Module):
            def __init__(inner):
                super().__init__()
                inner.weight = nn.Parameter(torch.zeros(2, dtype=torch.float32))
                inner.register_buffer("frozen_buffer", torch.ones(1))

            def trainable_state_dict(inner):
                return {name: value.detach().cpu() for name, value in inner.named_parameters()
                        if value.requires_grad}

            def load_trainable_state_dict(inner, state):
                inner.load_state_dict(state, strict=False)

            def training_metadata(inner):
                return {"mode": "lora", "trainable_parameters": sum(
                    value.numel() for value in inner.parameters() if value.requires_grad)}

        self.SmallTeacher = SmallTeacher
        self.teacher = SmallTeacher()
        self.payload = {"format_version": 1, "model": {"weight": torch.tensor([0.7, -0.4])},
                        "model_metadata": self.teacher.training_metadata(),
                        "cohort_hash": "frozen-cohort", "config": deepcopy(self.config),
                        "threshold": 0.43, "best_val_patient_roc_auc": 0.81}
        torch.save(self.payload, self.checkpoint)

        def load_checkpoint(path, model, config, cohort_hash):
            state = torch.load(path, map_location="cpu", weights_only=True)
            self.restored_before_freeze.append(all(value.requires_grad for value in model.parameters()))
            if state["cohort_hash"] != cohort_hash or state["model_metadata"] != model.training_metadata():
                raise ValueError("legacy metadata/cohort mismatch")
            model.load_trainable_state_dict(state["model"])
            return state

        self.supervised = SimpleNamespace(
            prepare=Mock(return_value=self.split), _split_hash=lambda split: split.attrs["split_metadata"]["_manifest_hash"],
            _classifier=Mock(return_value=self.teacher), _load_checkpoint=Mock(side_effect=load_checkpoint))
        self.builder = Mock(return_value=object())
        self.runtime_patch = patch.object(sources, "_runtime", return_value=(torch, self.supervised, self.builder))
        self.runtime_patch.start()
        self.addCleanup(self.runtime_patch.stop)

    def save(self):
        self.torch.save(self.payload, self.checkpoint)

    def load(self):
        return sources.load_teacher(self.source, self.project, device="cpu", cache_dir="image_cache")

    def test_restores_trained_parameters_before_freezing_and_fingerprints_checkpoint(self):
        with patch.object(self.torch, "load", wraps=self.torch.load) as safe_load:
            teacher, config, split, payload, info = self.load()
        self.assertTrue(all(call.kwargs.get("weights_only") is True for call in safe_load.call_args_list))
        self.assertEqual(len(safe_load.call_args_list), 2)
        self.torch.testing.assert_close(teacher.weight, self.payload["model"]["weight"], rtol=0, atol=0)
        self.assertEqual(self.restored_before_freeze, [True])
        self.assertFalse(teacher.training)
        self.assertFalse(any(value.requires_grad for value in teacher.parameters()))
        self.assertEqual(info["model_metadata"]["trainable_parameters"], 2)
        self.assertEqual(info["checkpoint_sha256"], sources._sha256_file(self.checkpoint))
        self.assertEqual(info["config"]["device"], "cuda:1")
        self.assertEqual(config["device"], "cpu")
        self.assertEqual(config["cache_dir"], str(self.project / "image_cache"))
        self.assertIs(split, self.split)

    def test_generalizes_to_both_other_selected_teachers(self):
        for model in ("chexfound", "chexzero"):
            with self.subTest(model=model):
                self.source["model"] = model
                self.payload["config"]["model"] = model
                self.teacher = self.SmallTeacher()
                self.supervised._classifier.return_value = self.teacher
                self.save()
                result = self.load()
                self.assertEqual(result[4]["model"], model)

    def test_rejects_wrong_checkpoint_identity(self):
        for key, value in (("model", "chexzero"), ("mode", "linear_probe"), ("seed", 43)):
            with self.subTest(key=key):
                self.payload["config"] = {**self.config, key: value}
                self.save()
                with self.assertRaisesRegex(ValueError, "model/mode/seed"):
                    self.load()
        self.supervised._classifier.assert_not_called()

    def test_requires_completed_validation_selected_checkpoint(self):
        for key, value in (("format_version", 2), ("threshold", None),
                           ("best_val_patient_roc_auc", float("nan")), ("threshold", True)):
            with self.subTest(key=key):
                old = self.payload[key]
                self.payload[key] = value
                self.save()
                with self.assertRaises(ValueError):
                    self.load()
                self.payload[key] = old
        self.supervised._classifier.assert_not_called()

    def test_missing_manifest_never_recreates_split(self):
        self.manifest.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "do not create a new split"):
            self.load()
        self.supervised.prepare.assert_not_called()

    def test_rejects_different_frozen_cohort_before_model_build(self):
        self.payload["cohort_hash"] = "different"
        self.save()
        with self.assertRaisesRegex(ValueError, "different frozen patient split"):
            self.load()
        self.supervised._classifier.assert_not_called()

    def test_rejects_missing_extra_wrong_shape_dtype_and_nonfinite_weights(self):
        torch = self.torch
        candidates = ({}, {"weight": torch.ones(2), "extra": torch.ones(1)},
                      {"weight": torch.ones(3)}, {"weight": torch.ones(2, dtype=torch.float64)},
                      {"weight": torch.tensor([float("nan"), 1.0])})
        for state in candidates:
            with self.subTest(state=state):
                self.payload["model"] = state
                self.save()
                with self.assertRaisesRegex(ValueError, "parameter keys|shape/dtype"):
                    self.load()
        self.supervised._load_checkpoint.assert_not_called()

    def test_uses_legacy_metadata_checks_before_freezing(self):
        self.payload["model_metadata"]["trainable_parameters"] = 99
        self.save()
        with self.assertRaisesRegex(ValueError, "legacy metadata"):
            self.load()
        self.assertTrue(all(value.requires_grad for value in self.teacher.parameters()))

    def test_detects_nonfinite_frozen_source_buffer(self):
        self.teacher.frozen_buffer.fill_(float("inf"))
        with self.assertRaisesRegex(ValueError, "non-finite parameters or buffers"):
            self.load()

    def test_detects_source_replacement_between_safe_loads(self):
        original = self.supervised._load_checkpoint.side_effect

        def replace_after_load(*args):
            restored = original(*args)
            self.checkpoint.write_bytes(b"replaced checkpoint")
            return restored

        self.supervised._load_checkpoint.side_effect = replace_after_load
        with self.assertRaisesRegex(ValueError, "changed while"):
            self.load()

    def cohort_records(self):
        records = []
        for model in sources.TEACHER_MODELS:
            config = {**deepcopy(self.config), "model": model}
            payload = {"cohort_hash": "frozen-cohort"}
            records.append((None, config, self.split.copy(deep=True), payload, {"model": model}))
        return records

    def test_cohort_comparison_accepts_row_order_change_and_returns_anchor(self):
        records = self.cohort_records()
        other = records[1][2].iloc[::-1].copy()
        records[1] = (*records[1][:2], other, *records[1][3:])
        self.assertIs(sources.validate_teacher_cohorts(records), records[0][2])

    def test_cohort_comparison_rejects_mismatches_even_if_manifest_claims_match(self):
        mutations = (
            lambda record: record[2].__setitem__("label", [1, 1, 0]),
            lambda record: record[2].__setitem__("split", ["test", "val", "train"]),
            lambda record: record[3].__setitem__("cohort_hash", "different"),
            lambda record: record[2].attrs["split_metadata"].__setitem__("_source_hash", "different"),
            lambda record: record[1]["data"].__setitem__("label_column", "another-task"),
            lambda record: record[1].__setitem__("label_mapping", {"positive": 0}),
            lambda record: record[1].__setitem__("split_seed", 100),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                records = self.cohort_records()
                mutation(records[1])
                with self.assertRaisesRegex(ValueError, "different"):
                    sources.validate_teacher_cohorts(records)

    def test_cohort_comparison_requires_all_three_once(self):
        records = self.cohort_records()
        for candidate in (records[:2], [records[0], records[0], records[2]]):
            with self.assertRaises(ValueError):
                sources.validate_teacher_cohorts(candidate)


if __name__ == "__main__":
    unittest.main()
