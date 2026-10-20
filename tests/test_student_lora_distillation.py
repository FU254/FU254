"""Controlled LoRA-student supervised/KD experiments with a tiny real HF teacher."""

from contextlib import redirect_stdout
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import unittest
from unittest import mock

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_single_teacher_distillation as integration_fixture

from foundation_sft.models.common import BackboneFeatures, LoRALinear


class SmallLoRATokenStudent(integration_fixture.SmallTokenStudent):
    """Complete toy student with production-compatible attention LoRA names."""

    def __init__(self, checkpoint=None, *, initialization=None,
                 mode="full_finetune", rank=8, alpha=16.0):
        super().__init__(checkpoint, initialization=initialization)
        if mode not in {"full_finetune", "lora"}:
            raise ValueError("Unsupported synthetic student training mode")
        block = nn.Module()
        block.attn = nn.Module()
        block.attn.qkv = nn.Linear(6, 6)
        self.encoder.blocks = nn.ModuleList([block])
        self.training_mode = mode
        self.lora = None
        if mode == "lora":
            self.encoder.requires_grad_(False)
            block.attn.qkv = LoRALinear(block.attn.qkv, rank=rank, alpha=alpha)
            self.encoder.head.requires_grad_(True)
            self.lora = {"rank": rank, "alpha": float(alpha),
                         "targets": ["encoder.blocks.0.attn.qkv"]}

    def forward(self, images, return_features=False):
        spatial = self.encoder.patch_embed(images)
        tokens = spatial.flatten(2).transpose(1, 2)
        local = self.encoder.norm(tokens + self.encoder.blocks[0].attn.qkv(tokens))
        features = BackboneFeatures(local.mean(1), local, tuple(spatial.shape[-2:])).validate()
        logits = self.encoder.head(features.global_features).squeeze(-1)
        if return_features:
            return {"logits": logits, "global_features": features.global_features,
                    "local_features": features.local_features, "grid_shape": features.grid_shape}
        return logits

    def training_metadata(self):
        metadata = super().training_metadata()
        metadata["mode"] = self.training_mode
        if self.lora is not None:
            metadata["lora"] = deepcopy(self.lora)
        return metadata


@unittest.skipUnless(importlib.util.find_spec("transformers"), "Tiny HF teacher needs pinned Transformers")
class StudentLoRADistillationTests(unittest.TestCase):
    def setUp(self):
        fixture = integration_fixture.SingleTeacherDistillationTests(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        for name in ("kdtrain", "root", "teacher_config", "teacher_checkpoint",
                     "student_checkpoint", "split"):
            setattr(self, name, getattr(fixture, name))
        self.config = deepcopy(fixture.config)
        self.config["student"] = {"mode": "lora", "rank": 2, "alpha": 4.0}
        self.config["training"]["lora_lr"] = self.config["training"]["backbone_lr"]

    def assert_state_equal(self, expected, actual):
        self.assertEqual(set(expected), set(actual))
        for name, value in expected.items():
            torch.testing.assert_close(value, actual[name], rtol=0, atol=0, msg=name)

    def test_options_default_legacy_factory_and_lora_arguments(self):
        legacy = deepcopy(self.config)
        legacy.pop("student")
        self.assertEqual(self.kdtrain._student_options(legacy), {"mode": "full_finetune"})
        self.assertEqual(self.kdtrain._student_options(self.config), self.config["student"])
        with mock.patch.object(self.kdtrain, "build_vit_tiny_student") as factory:
            self.kdtrain._build_student(legacy)
            self.assertEqual(factory.call_args.args, (str(self.student_checkpoint),))
            self.assertEqual(factory.call_args.kwargs, {})
            self.kdtrain._build_student(self.config)
            self.assertEqual(factory.call_args.kwargs, self.config["student"])
        original = deepcopy(self.config)
        self.kdtrain._student_options(self.config)
        self.assertEqual(self.config, original)

    def test_invalid_student_settings_fail_before_teacher_or_student_loading(self):
        invalid = (
            {"mode": "linear_probe"}, {"mode": "unknown"}, {"mode": True},
            {"mode": "lora", "rank": True}, {"mode": "lora", "rank": 0},
            {"mode": "lora", "rank": 2.0}, {"mode": "lora", "alpha": True},
            {"mode": "lora", "alpha": float("nan")},
            {"mode": "lora", "alpha": float("inf")},
            {"mode": "lora", "alpha": 0}, [],
        )
        for setting in invalid:
            config = deepcopy(self.config)
            config["student"] = setting
            with self.subTest(student=setting), redirect_stdout(io.StringIO()), mock.patch.object(
                self.kdtrain, "load_teacher", side_effect=AssertionError("Invalid config loaded teacher")
            ) as teacher, mock.patch.object(
                self.kdtrain, "build_vit_tiny_student", side_effect=AssertionError("Invalid config loaded student")
            ) as student:
                with self.assertRaises(ValueError):
                    self.kdtrain.run_experiments(config, project=self.root)
                teacher.assert_not_called()
                student.assert_not_called()

    def test_lora_optimizer_excludes_frozen_base_and_preserves_controlled_lr_and_decay(self):
        student = SmallLoRATokenStudent("synthetic", mode="lora", rank=2, alpha=4.0)
        kd = self.kdtrain.SingleTeacherDistillationLoss(student_dim=6, teacher_dim=12, pool_grid=(2, 2))
        train = {"backbone_lr": 5e-5, "lora_lr": 5e-5, "head_lr": 1e-4,
                 "projection_lr": 1e-4, "weight_decay": 0.03}
        for distillation in (None, kd):
            with self.subTest(distillation=distillation is not None):
                optimizer = self.kdtrain._optimizer(student, distillation, train)
                groups = {id(parameter): group for group in optimizer.param_groups for parameter in group["params"]}
                optimizer_count = sum(len(group["params"]) for group in optimizer.param_groups)
                expected = {id(p) for p in student.parameters() if p.requires_grad}
                if distillation is not None:
                    expected.update(id(p) for p in distillation.parameters() if p.requires_grad)
                self.assertEqual(set(groups), expected)
                self.assertEqual(optimizer_count, len(expected))
                for name, parameter in student.named_parameters():
                    if not parameter.requires_grad:
                        self.assertNotIn(id(parameter), groups)
                        continue
                    self.assertTrue(name.startswith("encoder.head.") or name.endswith((".lora_A", ".lora_B")))
                    expected_lr = train["head_lr"] if name.startswith("encoder.head.") else train["lora_lr"]
                    self.assertEqual(groups[id(parameter)]["lr"], expected_lr)
                    self.assertEqual(groups[id(parameter)]["weight_decay"], train["weight_decay"])
                if distillation is not None:
                    for parameter in distillation.parameters():
                        self.assertEqual(groups[id(parameter)]["lr"], train["projection_lr"])
                        self.assertEqual(groups[id(parameter)]["weight_decay"], train["weight_decay"])
        # Verify that the explicit LoRA LR is actually read, rather than an
        # accidental match with the controlled experiment's backbone LR.
        override = {**train, "lora_lr": 7e-5}
        optimizer = self.kdtrain._optimizer(student, None, override)
        groups = {id(parameter): group for group in optimizer.param_groups for parameter in group["params"]}
        for name, parameter in student.named_parameters():
            if name.endswith((".lora_A", ".lora_B")):
                self.assertEqual(groups[id(parameter)]["lr"], override["lora_lr"])

    def test_full_optimizer_and_legacy_full_checkpoint_keep_old_factory_contract(self):
        full = integration_fixture.SmallTokenStudent(str(self.student_checkpoint))
        train = {"backbone_lr": 5e-5, "lora_lr": 0.9, "head_lr": 1e-4,
                 "projection_lr": 1e-4, "weight_decay": 0.03}
        optimizer = self.kdtrain._optimizer(full, None, train)
        groups = {id(parameter): group for group in optimizer.param_groups for parameter in group["params"]}
        for name, parameter in full.named_parameters():
            expected_lr = train["head_lr"] if name.startswith("encoder.head.") else train["backbone_lr"]
            self.assertEqual(groups[id(parameter)]["lr"], expected_lr)
            self.assertEqual(groups[id(parameter)]["weight_decay"], train["weight_decay"])
        path = self.root / "legacy_full.pt"
        torch.save({"format_version": self.kdtrain.FORMAT,
                    "model_metadata": full.training_metadata(), "model": full.trainable_state_dict()}, path)
        self.student_checkpoint.unlink()
        with mock.patch.object(self.kdtrain, "build_vit_tiny_student",
                               side_effect=integration_fixture.SmallTokenStudent) as factory:
            restored, _ = self.kdtrain.load_student_checkpoint(path)
        self.assertEqual(set(factory.call_args.kwargs), {"initialization"})
        self.assert_state_equal(full.state_dict(), restored.state_dict())

    def test_paired_lora_training_frozen_base_teacher_and_self_contained_checkpoints(self):
        initial_states, frozen_states, final_states, projection_updates = {}, {}, {}, []
        teachers, teacher_states = [], []
        train_epoch = self.kdtrain._train_epoch
        load_teacher = self.kdtrain.load_teacher

        def capture_teacher(*args, **kwargs):
            result = load_teacher(*args, **kwargs)
            teachers.append(result[0])
            teacher_states.append({name: value.detach().clone() for name, value in result[0].state_dict().items()})
            return result

        def inspect_epoch(student, teacher, kd, loader, optimizer, hard_loss, device, train):
            condition = "supervised" if kd is None else "distillation"
            first = condition not in initial_states
            if first:
                initial_states[condition] = student.trainable_state_dict()
                frozen_states[condition] = {
                    name: parameter.detach().clone() for name, parameter in student.named_parameters()
                    if not parameter.requires_grad
                }
            projection_before = kd.projector.weight.detach().clone() if kd is not None else None
            result = train_epoch(student, teacher, kd, loader, optimizer, hard_loss, device, train)
            state = student.state_dict()
            for name, value in frozen_states[condition].items():
                torch.testing.assert_close(value, state[name], rtol=0, atol=0, msg=name)
            if first:
                self.assertTrue(any(not torch.equal(value, state[name]) for name, value in initial_states[condition].items()
                                    if name.endswith(".lora_B")))
                self.assertTrue(any(not torch.equal(value, state[name]) for name, value in initial_states[condition].items()
                                    if name.startswith("encoder.head.")))
            if kd is not None:
                projection_updates.append(not torch.equal(projection_before, kd.projector.weight.detach()))
            self.assertTrue(all(parameter.grad is None and not parameter.requires_grad for parameter in teacher.parameters()))
            final_states[condition] = student.trainable_state_dict()
            return result

        original_config = deepcopy(self.config)
        with redirect_stdout(io.StringIO()), mock.patch.object(
            self.kdtrain, "build_vit_tiny_student", side_effect=SmallLoRATokenStudent
        ), mock.patch.object(self.kdtrain, "load_teacher", side_effect=capture_teacher), mock.patch.object(
            self.kdtrain, "_train_epoch", side_effect=inspect_epoch
        ):
            suite = Path(self.kdtrain.run_experiments(self.config, project=self.root))
        self.assertEqual(self.config, original_config)
        self.assertEqual(set(initial_states), {"supervised", "distillation"})
        self.assert_state_equal(initial_states["supervised"], initial_states["distillation"])
        self.assertTrue(projection_updates and all(projection_updates))
        for teacher, state in zip(teachers, teacher_states):
            self.assert_state_equal(state, teacher.state_dict())
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in teacher.parameters()))

        # Deployment needs neither the original teacher snapshot nor the
        # original student ImageNet file, including for frozen base weights.
        self.student_checkpoint.unlink()
        self.teacher_checkpoint.unlink()
        shutil.rmtree(self.teacher_config["model_dir"])
        fingerprints = []
        for condition in ("supervised", "distillation"):
            directory = suite / condition
            history = json.loads((directory / "history.json").read_text())
            self.assertEqual(len(history), self.config["training"]["epochs"])
            for filename in ("best.pt", "last.pt"):
                with self.subTest(condition=condition, checkpoint=filename), mock.patch.object(
                    self.kdtrain, "build_vit_tiny_student", side_effect=SmallLoRATokenStudent
                ) as factory:
                    restored, payload = self.kdtrain.load_student_checkpoint(directory / filename, device="cpu")
                self.assertEqual(factory.call_args.kwargs["mode"], "lora")
                self.assertEqual(factory.call_args.kwargs["rank"], self.config["student"]["rank"])
                self.assertEqual(factory.call_args.kwargs["alpha"], self.config["student"]["alpha"])
                self.assertIn("initialization", factory.call_args.kwargs)
                self.assertNotIn("checkpoint", factory.call_args.kwargs)
                self.assert_state_equal(payload["model"], restored.state_dict())
                for name, value in frozen_states[condition].items():
                    torch.testing.assert_close(value, payload["model"][name], rtol=0, atol=0, msg=name)
                self.assertEqual(payload["model_metadata"]["mode"], "lora")
                self.assertEqual(payload["model_metadata"]["lora"]["rank"], self.config["student"]["rank"])
                if filename == "last.pt":
                    self.assert_state_equal(final_states[condition], payload["model"])
                    self.assertIn("optimizer", payload)
                    self.assertIn("best_model", payload)
                else:
                    fingerprints.append(payload["initial_state_sha256"])
        self.assertEqual(fingerprints[0], fingerprints[1])

    def test_unequal_microbatches_match_full_batch_for_lora_and_projector(self):
        torch.manual_seed(71)
        template = SmallLoRATokenStudent("synthetic", mode="lora", rank=2, alpha=4.0)
        teacher = SmallLoRATokenStudent("teacher").requires_grad_(False).eval()
        teacher_before = {name: value.detach().clone() for name, value in teacher.state_dict().items()}
        images, teacher_images = torch.randn(3, 3, 8, 8), torch.randn(3, 3, 8, 8)
        labels = torch.tensor([0.0, 1.0, 1.0])

        def batch(section):
            return {"image": images[section], "teacher_image": teacher_images[section], "label": labels[section]}

        for distillation in (False, True):
            with self.subTest(distillation=distillation):
                first, second = deepcopy(template), deepcopy(template)
                kd_first = (self.kdtrain.SingleTeacherDistillationLoss(
                    student_dim=6, teacher_dim=6, pool_grid=(2, 2)
                ) if distillation else None)
                kd_second = deepcopy(kd_first)
                before_projection = kd_first.projector.weight.detach().clone() if distillation else None
                first_parameters = [p for p in first.parameters() if p.requires_grad]
                second_parameters = [p for p in second.parameters() if p.requires_grad]
                if distillation:
                    first_parameters += list(kd_first.parameters())
                    second_parameters += list(kd_second.parameters())
                first_optimizer = torch.optim.SGD(first_parameters, lr=0.01)
                second_optimizer = torch.optim.SGD(second_parameters, lr=0.01)
                uneven = self.kdtrain._train_epoch(
                    first, teacher, kd_first, [batch(slice(0, 2)), batch(slice(2, 3))],
                    first_optimizer, nn.BCEWithLogitsLoss(), torch.device("cpu"),
                    {"accumulation_steps": 2, "gradient_clip": 100.0, "amp": False})
                whole = self.kdtrain._train_epoch(
                    second, teacher, kd_second, [batch(slice(None))],
                    second_optimizer, nn.BCEWithLogitsLoss(), torch.device("cpu"),
                    {"accumulation_steps": 1, "gradient_clip": 100.0, "amp": False})
                self.assertEqual(uneven["optimizer_steps"], 1)
                self.assertEqual(whole["optimizer_steps"], 1)
                for name, value in first.state_dict().items():
                    torch.testing.assert_close(value, second.state_dict()[name], rtol=1e-5, atol=1e-7, msg=name)
                    if name in {n for n, p in template.named_parameters() if not p.requires_grad}:
                        torch.testing.assert_close(value, template.state_dict()[name], rtol=0, atol=0, msg=name)
                if distillation:
                    for name, value in kd_first.state_dict().items():
                        torch.testing.assert_close(value, kd_second.state_dict()[name], rtol=1e-5, atol=1e-7, msg=name)
                    self.assertFalse(torch.equal(before_projection, kd_first.projector.weight.detach()))
        self.assert_state_equal(teacher_before, teacher.state_dict())
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))


if __name__ == "__main__":
    unittest.main()
