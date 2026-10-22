"""CPU integration of three frozen teachers, legacy controls, and checkpoints.

Synthetic image datasets and small token networks replace only external model
weights and DICOM decoding; production preparation, loss, loader, backward, and
student serialization code run without a network or clinical data.
"""

from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import pandas as pd
import torch
from torch import nn

from foundation_sft import distillation as single
from foundation_sft import equal_teacher_distillation as equal
from foundation_sft.models.common import BackboneFeatures, PreprocessSpec


class TokenNetwork(nn.Module):
    def __init__(self, dimension, image_size, *, checkpoint=None, initialization=None,
                 preprocess=None, name="synthetic_student"):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.patch_embed = nn.Conv2d(3, dimension, kernel_size=2, stride=2)
        self.encoder.norm = nn.LayerNorm(dimension)
        self.encoder.head = nn.Linear(dimension, 1)
        self.feature_dim = dimension
        self.preprocess = preprocess or PreprocessSpec(image_size)
        self.adapter = SimpleNamespace(feature_dim=dimension, preprocess=self.preprocess)
        self.name = name
        self.initialization = deepcopy(initialization) if initialization is not None else {
            "source": "synthetic-token-model", "checkpoint": str(checkpoint),
        }

    def forward(self, images, return_features=False):
        spatial = self.encoder.patch_embed(images)
        local = self.encoder.norm(spatial.flatten(2).transpose(1, 2))
        features = BackboneFeatures(local.mean(1), local, tuple(spatial.shape[-2:])).validate()
        logits = self.encoder.head(features.global_features).squeeze(-1)
        if return_features:
            return {"logits": logits, "global_features": features.global_features,
                    "local_features": features.local_features, "grid_shape": features.grid_shape}
        return logits

    def trainable_state_dict(self):
        return {key: value.detach().cpu().clone() for key, value in self.state_dict().items()}

    def load_trainable_state_dict(self, state):
        self.load_state_dict(state, strict=True)

    def training_metadata(self):
        return {"model": self.name, "mode": "full_finetune", "feature_dim": self.feature_dim,
                "preprocessing": asdict(self.preprocess), "initialization": deepcopy(self.initialization),
                "total_parameters": sum(p.numel() for p in self.parameters()),
                "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad)}


class SyntheticCXRDataset:
    created = []

    def __init__(self, cohort, preprocess, training=False, **kwargs):
        self.cohort = cohort.reset_index(drop=True)
        self.preprocess = preprocess
        self.mean = torch.tensor(preprocess.mean).view(3, 1, 1)
        self.std = torch.tensor(preprocess.std).view(3, 1, 1)
        self.training = training
        self.calls = []
        self.created.append(self)

    def __len__(self):
        return len(self.cohort)

    def __getitem__(self, index):
        self.calls.append(index)
        row = self.cohort.iloc[index]
        size = self.preprocess.image_size
        grid = torch.linspace(0, 1, size)
        gray = (0.6 * grid[:, None] + 0.3 * grid[None, :] + 0.05 * row["label"]).unsqueeze(0)
        if self.training:
            if torch.rand(()) < 0.5:
                gray = gray.flip(-1)
            gray = gray.roll(int(torch.randint(0, 3, ()).item()), dims=-2)
        image = (gray.repeat(3, 1, 1) - self.mean) / self.std
        return {"image": image, "label": torch.tensor(float(row["label"])),
                "sample_id": row["sample_id"], "patient_id": row["patient_id"]}


class FirstBatchCaptured(Exception):
    pass


class EqualTeacherPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="equal-teacher-integration-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.student_checkpoint = self.root / "student.pth"
        self.student_checkpoint.write_bytes(b"synthetic factory source")
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, self.old_threads)
        SyntheticCXRDataset.created = []
        self.split = pd.DataFrame([
            {"sample_id": f"sample-{index}", "patient_id": f"patient-{index}",
             "path": str(self.root / f"synthetic-{index}.dcm"), "label": index % 2,
             "split": "train" if index < 4 else "val" if index < 6 else "test"}
            for index in range(8)
        ])
        self.split.attrs["split_metadata"] = {"_manifest_hash": "synthetic-frozen-cohort"}
        self.config = {
            "seed": 42, "device": "cpu", "student_checkpoint": str(self.student_checkpoint),
            "output_root": str(self.root / "outputs"), "cache_dir": str(self.root / "cache"),
            "student": {"mode": "full_finetune"},
            "distillation": deepcopy(equal.LOSS_SETTINGS),
            "teachers": [{"model": name, "mode": "lora", "seed": 42,
                          "checkpoint": str(self.root / name / "best.pt"), "weight": 1 / 3}
                         for name in equal.TEACHER_NAMES],
            "training": {"epochs": 2, "patience": 2, "batch_size": 3, "accumulation_steps": 2,
                         "num_workers": 0, "backbone_lr": 5e-5, "head_lr": 1e-4,
                         "projection_lr": 1e-4, "amp": False},
        }
        self.base_training = {
            **self.config["training"], "minimum_lr": 1e-7, "weight_decay": 0.01,
            "focal_gamma": 2.0, "focal_scale": 0.8, "gradient_clip": 1.0,
            "horizontal_flip_probability": 0.5, "rotation_degrees": 3.0,
        }
        self.loaded = []
        self.teacher_before = []
        self.students = []
        self.student_initial_states = []

    def fake_teacher(self, source, project, **kwargs):
        name = source["model"]
        index = equal.TEACHER_NAMES.index(name)
        if name == "rad_dino":
            spec = PreprocessSpec(32, mean=(0.4, 0.5, 0.6), std=(0.2, 0.3, 0.4),
                                  resize_mode="shortest_edge_center_crop", resize_shortest_edge=36)
        elif name == "chexfound":
            spec = PreprocessSpec(40, resize_mode="shortest_edge_center_crop", resize_shortest_edge=40)
        else:
            spec = PreprocessSpec(28, mean=(101.48761/255,) * 3, std=(83.43944/255,) * 3,
                                  resize_mode="chexzero_pad320")
        teacher = TokenNetwork(8 + index * 2, spec.image_size, preprocess=spec,
                               name=f"synthetic_{name}").requires_grad_(False).eval()
        config = {
            "model": name, "mode": "lora", "seed": 42, "split_seed": 42,
            "split_ratios": [0.5, 0.25, 0.25], "data": {"task": "synthetic"},
            "training": deepcopy(self.base_training), "evaluation": {"bootstrap_samples": 4},
        }
        metadata = teacher.training_metadata()
        payload = {"format_version": 1, "model_metadata": metadata,
                   "cohort_hash": "synthetic-frozen-cohort", "threshold": 0.5,
                   "best_val_patient_roc_auc": 0.75}
        info = {
            "model": name, "mode": "lora", "seed": 42, "weight": 1 / 3,
            "checkpoint": source["checkpoint"], "checkpoint_sha256": str(index + 1) * 64,
            "model_metadata": deepcopy(metadata), "cohort_hash": payload["cohort_hash"],
            "best_val_patient_roc_auc": payload["best_val_patient_roc_auc"],
        }
        result = teacher, config, self.split.copy(deep=True), payload, info
        self.loaded.append(result)
        self.teacher_before.append(single._state_hash(teacher.state_dict()))
        return result

    def student_factory(self, checkpoint=None, *, initialization=None):
        student = TokenNetwork(6, 28, checkpoint=checkpoint, initialization=initialization)
        self.students.append(student)
        self.student_initial_states.append(single._cpu_state(student))
        return student

    def patches(self):
        stack = ExitStack()
        stack.enter_context(mock.patch.object(equal, "load_teacher", side_effect=self.fake_teacher))
        stack.enter_context(mock.patch.object(single, "build_vit_tiny_student", side_effect=self.student_factory))
        stack.enter_context(mock.patch.object(single, "CXRDataset", SyntheticCXRDataset))
        stack.enter_context(redirect_stdout(io.StringIO()))
        return stack

    def assert_teacher_states_unchanged(self, before):
        for item, state in zip(self.loaded, before):
            teacher = item[0]
            self.assertFalse(teacher.training)
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in teacher.parameters()))
            self.assertEqual(single._state_hash(teacher.state_dict()), state)

    def test_prepare_validates_all_teacher_cohorts_and_keeps_exact_legacy_anchor_schema(self):
        original = deepcopy(self.config)
        with self.patches():
            prepared, teachers, split, infos, anchor = equal._prepare(self.config, self.root)
        self.assertEqual(self.config, original)
        self.assertEqual(tuple(teachers), equal.TEACHER_NAMES)
        self.assertEqual([item["model"] for item in infos], list(equal.TEACHER_NAMES))
        self.assertEqual(prepared["training"], self.base_training)
        self.assertEqual(prepared["teacher_checkpoint"], infos[0]["checkpoint"])
        self.assertEqual(split.attrs, self.split.attrs)
        expected = {key: infos[0][key] for key in (
            "checkpoint", "checkpoint_sha256", "model_metadata", "cohort_hash", "best_val_patient_roc_auc")}
        expected["selection"] = "User-selected RAD-DINO LoRA; checkpoint selected by validation patient ROC-AUC"
        self.assertEqual(anchor, expected)
        self.assertTrue(all(not t.training and not any(p.requires_grad for p in t.parameters())
                            for t in teachers.values()))

    def test_prepare_rejects_changed_protocol_before_loading_large_sources(self):
        for change in ({"student": {"mode": "lora"}}, {"seed": True}, {"seed": -1},
                       {"teachers": list(reversed(self.config["teachers"]))},
                       {"distillation": {**self.config["distillation"], "local_weight": 0.0}}):
            config = {**self.config, **change}
            with self.subTest(change=change), mock.patch.object(equal, "load_teacher") as load:
                with self.assertRaises(ValueError):
                    equal._prepare(config, self.root)
                load.assert_not_called()

    def test_check_backpropagates_student_and_three_projectors_then_reloads_without_test_data(self):
        captured_kd, snapshots = [], []
        build = equal._build_loss
        save = equal.supervised._torch_save

        def record_kd(*args, **kwargs):
            kd = build(*args, **kwargs)
            captured_kd.append(kd)
            return kd

        def record_snapshot(path, payload):
            snapshots.append(deepcopy(payload))
            return save(path, payload)

        with self.patches(), mock.patch.object(equal, "_build_loss", side_effect=record_kd), mock.patch.object(
                equal.supervised, "_torch_save", side_effect=record_snapshot), mock.patch.object(
                equal.supervised, "predict", side_effect=AssertionError("Preflight must not evaluate held-out patients")):
            directory = equal.check(self.config, self.root)
        result = json.loads((directory / "check.json").read_text())
        self.assertTrue(result["check_passed"])
        self.assertTrue(result["student_state_exact"])
        self.assertTrue(result["teacher_unchanged"])
        self.assertFalse(result["training_started"])
        self.assertEqual(result["student_state_sha256"], result["restored_state_sha256"])
        self.assertLessEqual(result["roundtrip_max_abs_error"], result["roundtrip_atol"])
        self.assertFalse((directory / "roundtrip.pt").exists())
        self.assertEqual(len(SyntheticCXRDataset.created), 1)
        self.assertEqual(set(SyntheticCXRDataset.created[0].cohort["split"]), {"train"})
        self.assertEqual(len(SyntheticCXRDataset.created[0].calls), 3)
        for name, parameter in self.students[0].named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertTrue(any(p.grad.abs().sum().item() > 0 for p in self.students[0].encoder.patch_embed.parameters()))
        for name in equal.TEACHER_NAMES:
            weight = captured_kd[0].criteria[name].projector.weight
            self.assertTrue(torch.isfinite(weight.grad).all())
            self.assertGreater(weight.grad.abs().sum().item(), 0)
            self.assertIn(f"{name}_total", result["losses"])
        self.assert_teacher_states_unchanged(self.teacher_before)
        payload = snapshots[0]
        self.assertEqual(payload["experiment"], equal.EXPERIMENT)
        self.assertEqual(payload["condition"], "equal_teacher")
        self.assertEqual(json.loads(json.dumps(payload["teachers"])), result["teachers"])
        self.assertEqual(payload["teacher_views"], result["teacher_views"])
        self.assertEqual(set(payload["model"]), set(self.students[0].state_dict()))

    def test_each_condition_has_identical_student_initialization_and_legacy_control_metadata(self):
        results = {}
        for condition in equal.CONDITIONS:
            with self.patches():
                directory = equal.check(self.config, self.root, condition)
            results[condition] = json.loads((directory / "check.json").read_text())
        baseline = results["supervised"]
        for condition, result in results.items():
            with self.subTest(condition=condition):
                self.assertEqual(result["initial_state_sha256"], baseline["initial_state_sha256"])
                self.assertEqual(result["model_metadata"], baseline["model_metadata"])
                self.assertEqual(result["shared_view"], baseline["shared_view"])
                self.assertEqual(result["teacher"], baseline["teacher"])
                self.assertEqual(result["teacher_views"], baseline["teacher_views"])
        with self.patches():
            prepared, teachers, split, _, anchor = equal._prepare(self.config, self.root)
            with mock.patch.object(single, "_prepare", return_value=(prepared, teachers["rad_dino"], split, anchor)):
                directory = single.check(prepared, self.root, condition="distillation")
        legacy = json.loads((directory / "check.json").read_text())
        for key in ("model_metadata", "initial_state_sha256", "shared_view", "teacher", "effective_training"):
            self.assertEqual(results["single_teacher"][key], legacy[key], key)

    def test_training_arms_share_initial_weights_shuffle_views_and_augmentation_rng(self):
        batches, initial_states, rngs = {}, {}, {}

        def capture(condition):
            def first_epoch(student, teacher_or_teachers, kd, loader, *unused):
                initial_states[condition] = single._cpu_state(student)
                batch = next(iter(loader))
                batches[condition] = {key: deepcopy(batch[key]) for key in
                                      ("image", "teacher_image", "sample_id", "patient_id", "label")
                                      if key in batch}
                rngs[condition] = torch.get_rng_state().clone()
                raise FirstBatchCaptured()
            return first_epoch

        with self.patches():
            config, teachers, split, infos, anchor = equal._prepare(self.config, self.root)
            for condition in equal.CONDITIONS:
                suite = self.root / condition
                suite.mkdir()
                with self.subTest(condition=condition), self.assertRaises(FirstBatchCaptured):
                    if condition == "equal_teacher":
                        with mock.patch.object(equal, "_train_epoch", side_effect=capture(condition)):
                            equal._run_equal(suite, config, teachers, split, infos, anchor)
                    else:
                        mapped = "supervised" if condition == "supervised" else "distillation"
                        with mock.patch.object(single, "_train_epoch", side_effect=capture(condition)):
                            single._run_condition(mapped, suite, teachers["rad_dino"], split, config, anchor)
        for condition in equal.CONDITIONS:
            with self.subTest(condition=condition):
                self.assertEqual(single._state_hash(initial_states[condition]),
                                 single._state_hash(initial_states["supervised"]))
                for key in ("sample_id", "patient_id"):
                    self.assertEqual(batches[condition][key], batches["supervised"][key])
                for key in ("image", "label"):
                    torch.testing.assert_close(batches[condition][key], batches["supervised"][key], rtol=0, atol=0)
                torch.testing.assert_close(rngs[condition], rngs["supervised"], rtol=0, atol=0)
        torch.testing.assert_close(batches["single_teacher"]["teacher_image"],
                                   batches["equal_teacher"]["teacher_image"], rtol=0, atol=0)

    def test_saved_equal_payload_reloads_with_old_loader_without_teacher_or_initial_file(self):
        with self.patches():
            config, teachers, split, infos, anchor = equal._prepare(self.config, self.root)
            equal.supervised.set_seed(config["seed"])
            student = single._build_student(config).eval()
            dataset = equal._loader(split, "train", teachers, student, config).dataset
            image = dataset[0]["image"].unsqueeze(0)
            expected = student(image).detach()
            payload = equal._payload(student, config, "synthetic-frozen-cohort",
                                     single._state_hash(student.state_dict()), anchor,
                                     dataset.metadata(), infos, dataset.teacher_view_metadata())
            checkpoint = self.root / "saved_equal.pt"
            torch.save(payload, checkpoint)
            self.student_checkpoint.unlink()
            with mock.patch.object(equal, "load_teacher", side_effect=AssertionError("Inference must not load teachers")):
                restored, saved = single.load_student_checkpoint(checkpoint, "cpu")
            single._verify_roundtrip_state(student, restored)
            torch.testing.assert_close(restored(image), expected, rtol=0, atol=0)
        self.assertEqual(saved["shared_view"], dataset.shared.metadata())
        self.assertEqual(saved["format_version"], single.FORMAT)
        self.assertFalse(restored.training)

    def test_every_teacher_is_checked_for_training_mode_trainable_or_stale_gradients(self):
        with self.patches():
            _, teachers, _, _, _ = equal._prepare(self.config, self.root)
        for name in equal.TEACHER_NAMES:
            teacher = teachers[name]
            parameter = next(teacher.parameters())
            for defect in ("train_mode", "trainable", "gradient"):
                with self.subTest(teacher=name, defect=defect):
                    if defect == "train_mode":
                        teacher.train()
                    elif defect == "trainable":
                        parameter.requires_grad_(True)
                    else:
                        parameter.grad = torch.ones_like(parameter)
                    with self.assertRaisesRegex(RuntimeError, name):
                        equal._assert_frozen(teachers)
                    teacher.requires_grad_(False).eval()
                    parameter.grad = None
        equal._assert_frozen(teachers)

    def test_equal_teacher_uneven_accumulation_matches_whole_batch_for_all_projectors(self):
        with self.patches():
            config, teachers, split, _, _ = equal._prepare(self.config, self.root)
            student = self.student_factory("synthetic")
            criterion = equal._build_loss(student, teachers, config)
            batch = next(iter(equal._loader(split, "train", teachers, student, config)))
        first, second = deepcopy(student), deepcopy(student)
        kd_first, kd_second = deepcopy(criterion), deepcopy(criterion)

        def section(start, stop):
            return {"image": batch["image"][start:stop], "label": batch["label"][start:stop],
                    "teacher_images": {name: batch["teacher_images"][name][start:stop]
                                       for name in equal.TEACHER_NAMES}}

        before = [single._state_hash(item[0].state_dict()) for item in self.loaded]
        for network, kd, chunks, accumulation in (
            (first, kd_first, [section(0, 2), section(2, 3)], 2),
            (second, kd_second, [section(0, 3)], 1),
        ):
            optimizer = torch.optim.SGD(list(network.parameters()) + list(kd.parameters()), lr=0.01)
            result = equal._train_epoch(network, teachers, kd, chunks, optimizer,
                nn.BCEWithLogitsLoss(), torch.device("cpu"),
                {"accumulation_steps": accumulation, "gradient_clip": 100.0})
            self.assertEqual(result["optimizer_steps"], 1)
        for before_module, after_module in ((first, second), (kd_first, kd_second)):
            for name, tensor in before_module.state_dict().items():
                torch.testing.assert_close(tensor, after_module.state_dict()[name], rtol=1e-5, atol=1e-7)
        self.assert_teacher_states_unchanged(before)


if __name__ == "__main__":
    unittest.main()
