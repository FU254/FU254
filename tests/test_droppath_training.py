"""DropPath training lifecycle on synthetic DICOMs and a real tiny HF teacher.

Native timm stochastic-depth probabilities are covered by the model tests. This
small token student keeps full optimizer, data, checkpoint and selection tests
fast while exercising the same metadata and isolated-stream contract.
"""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import shutil
import unittest
from unittest import mock

import torch

from foundation_sft import distillation as single
from foundation_sft import random_student_distillation as trainer
from foundation_sft import train as supervised
from foundation_sft.models.common import BackboneFeatures
from tests import test_random_student_training as synthetic


class SmallDropPathStudent(synthetic.SmallRandomStudent):
    """A tiny residual branch with a nonpersistent, isolated CPU random stream."""

    def __init__(self, checkpoint=None, *, initialization=None, random_seed=None,
                 drop_path_rate=0.0):
        super().__init__(checkpoint, initialization=initialization, random_seed=random_seed)
        self.drop_path_rate = float(drop_path_rate)
        self.stochastic_seed = self.initialization["seed"] ^ 0x44504154
        self._stream = torch.Generator().manual_seed(self.stochastic_seed).get_state()

    def stochastic_state_dict(self):
        if self.drop_path_rate == 0:
            return None
        return {"format_version": 1, "seed": self.stochastic_seed,
                "cpu": self._stream.clone(), "cuda": None}

    def load_stochastic_state_dict(self, state):
        if (self.drop_path_rate == 0 or not isinstance(state, dict)
                or set(state) != {"format_version", "seed", "cpu", "cuda"}
                or state["format_version"] != 1 or state["seed"] != self.stochastic_seed
                or state["cuda"] is not None):
            raise ValueError("Invalid synthetic DropPath stochastic state")
        cpu = state["cpu"]
        if not isinstance(cpu, torch.Tensor) or cpu.dtype != torch.uint8 or cpu.ndim != 1:
            raise ValueError("Invalid synthetic DropPath CPU state")
        try:
            torch.Generator().set_state(cpu)
        except RuntimeError as error:
            raise ValueError("Invalid synthetic DropPath CPU state") from error
        self._stream = cpu.clone()

    def _forward(self, images, return_features):
        spatial = self.encoder.patch_embed(images)
        local = self.encoder.norm(spatial.flatten(2).transpose(1, 2))
        mask = 1.
        if self.training and self.drop_path_rate:
            keep = 1. - self.drop_path_rate
            mask = (torch.rand((len(images), 1, 1)) < keep).to(local) / keep
        local = local + .1 * local * mask
        features = BackboneFeatures(local.mean(1), local, tuple(spatial.shape[-2:])).validate()
        logits = self.encoder.head(features.global_features).squeeze(-1)
        if return_features:
            return {"logits": logits, "global_features": features.global_features,
                    "local_features": features.local_features, "grid_shape": features.grid_shape}
        return logits

    def forward(self, images, return_features=False):
        if not self.training or self.drop_path_rate == 0:
            return self._forward(images, return_features)
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(self._stream)
            result = self._forward(images, return_features)
            self._stream = torch.get_rng_state().clone()
        return result

    def training_metadata(self):
        metadata = super().training_metadata()
        if self.drop_path_rate:
            metadata["regularization"] = {"drop_path_rate": self.drop_path_rate,
                "schedule": "linear_12_blocks", "rng": "isolated_torch",
                "rng_seed": self.stochastic_seed}
        return metadata


class _RecordBatches:
    def __init__(self, loader, records):
        self.loader, self.dataset, self.records = loader, loader.dataset, records

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            self.records.append((tuple(batch["sample_id"]), batch["image"].clone()))
            yield batch


class DropPathTrainingTests(unittest.TestCase):
    def setUp(self):
        fixture = synthetic.RandomStudentTrainingTests(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture, self.root = fixture, fixture.root
        for module in (trainer, single):
            patch = mock.patch.object(module, "build_vit_tiny_student", side_effect=SmallDropPathStudent)
            patch.start()
            self.addCleanup(patch.stop)
        self.raw = deepcopy(fixture.raw)
        self.raw["seed"] = 42
        self.raw["training"].update(weight_decay=.01, drop_path_rate=.1,
            num_workers=0, horizontal_flip_probability=.5, rotation_degrees=10.)
        with redirect_stdout(io.StringIO()):
            self.config, self.teacher, self.split, self.info = trainer._prepare(self.raw, self.root)

    def assert_stream_equal(self, first, second):
        self.assertEqual(set(first), set(second))
        self.assertEqual(first["format_version"], second["format_version"])
        self.assertEqual(first["seed"], second["seed"])
        torch.testing.assert_close(first["cpu"], second["cpu"], rtol=0, atol=0)
        self.assertIsNone(first["cuda"])
        self.assertIsNone(second["cuda"])

    def run_condition(self, condition, *, rate=.1, epochs=1, label="run"):
        config = deepcopy(self.config)
        config["training"].update(drop_path_rate=rate, epochs=epochs)
        destination = self.root / label
        destination.mkdir()
        with redirect_stdout(io.StringIO()):
            receipt = trainer._run_condition(condition, destination, self.teacher,
                                             self.split, config, self.info)
        return destination / condition, receipt, config

    def test_positive_preflight_is_train_only_backward_and_complete_reload(self):
        original_loader, original_build = single._loader, trainer._build_student
        students = []

        def train_only(split, name, *args, **kwargs):
            self.assertEqual(name, "train")
            return original_loader(split, name, *args, **kwargs)

        def capture(config):
            value = original_build(config)
            students.append(value)
            return value

        with redirect_stdout(io.StringIO()), mock.patch.object(single, "_loader", side_effect=train_only), mock.patch.object(
                trainer, "_build_student", side_effect=capture), mock.patch.object(
                supervised, "predict", side_effect=AssertionError("Preflight must not evaluate held-out pixels")):
            directory = trainer.check(self.raw, self.root, "distillation")
        check = json.loads((directory / "check.json").read_text())
        self.assertTrue(check["check_passed"])
        self.assertTrue(check["student_state_exact"])
        self.assertEqual(check["config"]["training"]["weight_decay"], .01)
        self.assertEqual(check["model_metadata"]["regularization"]["drop_path_rate"], .1)
        self.assertFalse(list(directory.glob("*.pt")))
        self.assertGreater(students[0].encoder.patch_embed.weight.grad.abs().sum().item(), 0)
        self.fixture.assert_teacher_unchanged(self.teacher)

    def test_a_b_preserve_weights_projector_and_cpu_augmentation_stream_against_dp0(self):
        original_epoch = single._train_epoch
        batches, receipts = {}, {}

        def capture(model, teacher, kd, loader, *args):
            condition = "supervised" if kd is None else "distillation"
            key = (model.drop_path_rate, condition)
            return original_epoch(model, teacher, kd, _RecordBatches(loader, batches.setdefault(key, [])), *args)

        with mock.patch.object(single, "_train_epoch", side_effect=capture):
            for rate in (0., .1):
                for condition in ("supervised", "distillation"):
                    run, receipt, config = self.run_condition(condition, rate=rate,
                        label=f"paired_{rate}_{condition}")
                    receipts[(rate, condition)] = receipt
                    history = json.loads((run / "history.json").read_text())
                    self.assertTrue(all("train_eval_loss" in row and "val_loss" in row for row in history))
                    saved = json.loads((run / "config.json").read_text())
                    self.assertEqual(saved["training"]["weight_decay"], .01)
                    self.assertEqual(saved["training"]["drop_path_rate"], rate)
                    payload = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
                    if rate:
                        self.assertEqual(payload["model_metadata"]["regularization"]["drop_path_rate"], rate)
                        self.assertIsInstance(payload["student_stochastic_state"], dict)
                    else:
                        self.assertNotIn("regularization", payload["model_metadata"])
                        self.assertNotIn("student_stochastic_state", payload)
        self.assertEqual(len({r["initial_state_sha256"] for r in receipts.values()}), 1)
        self.assertEqual(receipts[(0., "distillation")]["distillation_initial_state_sha256"],
                         receipts[(.1, "distillation")]["distillation_initial_state_sha256"])
        first = batches[(0., "supervised")]
        for recorded in batches.values():
            self.assertEqual(len(recorded), len(first))
            for (ids, images), (expected_ids, expected_images) in zip(recorded, first):
                self.assertEqual(ids, expected_ids)
                torch.testing.assert_close(images, expected_images, rtol=0, atol=0)
        self.fixture.assert_teacher_unchanged(self.teacher)

    def test_source_free_checkpoint_restores_stream_and_rejects_changed_protocol(self):
        run, receipt, config = self.run_condition("distillation", label="standalone")
        payload = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
        self.fixture.fixture.teacher_checkpoint.unlink()
        self.fixture.fixture.student_checkpoint.unlink()
        shutil.rmtree(config["teacher_config"]["model_dir"])
        with mock.patch.object(single, "load_teacher", side_effect=AssertionError("Student inference is source free")):
            student, _ = single.load_student_checkpoint(run / "best.pt", "cpu")
            restored, _ = single.load_student_checkpoint(run / "best.pt", "cpu")
        images = torch.linspace(-1, 1, 3*3*8*8).reshape(3, 3, 8, 8)
        state = student.stochastic_state_dict()
        self.assert_stream_equal(state, payload["student_stochastic_state"])
        with torch.no_grad():
            expected, actual = student(images), student(images)
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        self.assert_stream_equal(student.stochastic_state_dict(), state)
        student.train()
        restored.train()
        outside = torch.get_rng_state().clone()
        torch.testing.assert_close(student(images), restored(images), rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), outside, rtol=0, atol=0)
        self.assert_stream_equal(student.stochastic_state_dict(), restored.stochastic_state_dict())
        for defect in ("missing_stream", "metadata_rate", "config_rate", "stream_seed"):
            corrupt = deepcopy(payload)
            if defect == "missing_stream":
                corrupt.pop("student_stochastic_state")
            elif defect == "metadata_rate":
                corrupt["model_metadata"]["regularization"]["drop_path_rate"] = .2
            elif defect == "config_rate":
                corrupt["config"]["training"]["drop_path_rate"] = .2
            else:
                corrupt["student_stochastic_state"]["seed"] += 1
            path = self.root / (defect + ".pt")
            torch.save(corrupt, path)
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                single.load_student_checkpoint(path, "cpu")

    def test_best_student_and_random_stream_are_from_same_selected_epoch(self):
        snapshots, streams = [], []
        original_epoch, original_metrics = single._train_epoch, trainer.binary_metrics
        calls = 0

        def capture(model, *args):
            result = original_epoch(model, *args)
            snapshots.append(model.trainable_state_dict())
            streams.append(model.stochastic_state_dict())
            return result

        def force_first_best(*args, **kwargs):
            nonlocal calls
            result = original_metrics(*args, **kwargs)
            calls += 1
            if calls % 2 == 0:
                result["roc_auc"] = .9 if len(snapshots) == 1 else .7
            return result

        with mock.patch.object(single, "_train_epoch", side_effect=capture), mock.patch.object(
                trainer, "binary_metrics", side_effect=force_first_best):
            run, receipt, config = self.run_condition("distillation", epochs=2, label="selected_epoch")
        best = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
        last = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
        self.fixture.assert_state_equal(best["model"], snapshots[0])
        self.fixture.assert_state_equal(last["model"], snapshots[1])
        self.assert_stream_equal(best["student_stochastic_state"], streams[0])
        self.assert_stream_equal(last["student_stochastic_state"], streams[1])
        self.assertFalse(torch.equal(streams[0]["cpu"], streams[1]["cpu"]))
        self.assertEqual(receipt["best_val_patient_roc_auc"], .9)
        self.fixture.assert_teacher_unchanged(self.teacher)


if __name__ == "__main__":
    unittest.main()
