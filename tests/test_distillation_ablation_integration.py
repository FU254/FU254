"""Six loss groups against the real trainer, cached DICOMs, and a tiny HF teacher."""

from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import unittest
from unittest import mock

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_single_teacher_distillation as fixtures
from test_student_lora_distillation import SmallLoRATokenStudent

from foundation_sft import distillation_ablation as ablation
from foundation_sft.models import build_adapter


@unittest.skipUnless(importlib.util.find_spec("transformers"), "Tiny HF teacher needs pinned Transformers")
class LossAblationTrainerIntegrationTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.SingleTeacherDistillationTests(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.trainer, self.project = fixture.kdtrain, fixture.root
        self.config = deepcopy(fixture.config)
        self.config["training"].update(backbone_lr=5e-5, head_lr=1e-4, projection_lr=1e-4)
        self.config["distillation"]["pool_grid"] = [7, 7]
        self.config["cache_dir"] = str(self.project / "shared_cache")
        self.source_sha = hashlib.sha256(fixture.student_checkpoint.read_bytes()).hexdigest()

        # Keep the real pool7 protocol: enlarge the tiny teacher's spatial grid
        # rather than weakening the ablation planner to accept pool2 in tests.
        from transformers import Dinov2Config, Dinov2Model

        model_dir = Path(fixture.teacher_config["model_dir"])
        teacher_architecture = Dinov2Config.from_pretrained(model_dir)
        teacher_architecture.patch_size = 1
        teacher_architecture._attn_implementation = "eager"
        torch.manual_seed(901)
        Dinov2Model(teacher_architecture).save_pretrained(model_dir, safe_serialization=True)
        torch.manual_seed(4)
        teacher = self.trainer.supervised._classifier(build_adapter(fixture.teacher_config), fixture.teacher_config)
        teacher.load_trainable_state_dict(fixture.teacher_state)
        payload = self.trainer.supervised._checkpoint(
            teacher, fixture.teacher_config, self.trainer.supervised._split_hash(fixture.split))
        payload.update(threshold=0.43, best_val_patient_roc_auc=0.71)
        torch.save(payload, fixture.teacher_checkpoint)

        self.template = self.project / "full_student.json"
        self.template.write_text(json.dumps(self.config))
        defaults = Path(__file__).resolve().parents[1] / ablation.DEFAULT_CONFIG
        self.settings = json.loads(defaults.read_text())
        self.settings.update(
            template=str(self.template), output_root=str(self.project / "ablation"),
            seeds=[self.config["seed"], self.config["seed"] + 1], devices=["cpu"],
            max_parallel=1, reuse_suites=[],
        )
        self.settings_file = self.project / "ablation.json"

    def student_factory(self, checkpoint=None, *, initialization=None, **options):
        student = SmallLoRATokenStudent(checkpoint, initialization=initialization, **options)
        # Same eight-by-eight field of view as the teacher, and at least 7x7
        # true spatial tokens for the fixed local-relation pooling protocol.
        student.encoder.patch_embed = nn.Conv2d(3, 6, kernel_size=1, stride=1)
        if initialization is None:
            student.initialization["checkpoint_sha256"] = self.source_sha
        return student

    def assert_state_equal(self, expected, actual):
        self.assertEqual(set(expected), set(actual))
        for name, value in expected.items():
            torch.testing.assert_close(value, actual[name], rtol=0, atol=0, msg=name)

    def test_six_groups_reuse_abcd_controls_train_four_ablation_arms_and_report(self):
        trainer, project, seed = self.trainer, self.project, self.config["seed"]
        current = {"action": "legacy"}
        calls, receipts, initial_states, first_training_views = [], {}, {}, {}
        projector_gradients = {group: [] for group in ablation.GROUPS}
        projector_modules = []
        train_epoch, batch_loss, optimizer, loader = (
            trainer._train_epoch, trainer._batch_loss, trainer._optimizer, trainer._loader)

        def group_of(kd):
            if kd is None:
                return "supervised"
            weights = (kd.soft_weight, kd.global_weight, kd.local_weight)
            return next(group for group, definition in ablation.GROUPS.items()
                        if definition[0] == "distillation" and definition[1:] == weights)

        def inspect_loader(split, name, *args, **kwargs):
            if current["action"] == "check":
                self.assertEqual(name, "train", "Preflight read a held-out partition")
            return loader(split, name, *args, **kwargs)

        def inspect_batch(student, teacher, kd, batch, hard_loss, device):
            group = group_of(kd)
            if current["action"] in {"train", "legacy"} and group not in first_training_views:
                first_training_views[group] = (
                    batch["image"].clone(), batch["label"].clone(), list(batch["sample_id"]))
            _, soft, global_weight, local = ablation.GROUPS[group]
            if kd is not None:
                self.assertEqual(kd.projector is not None, global_weight > 0)
                if kd.projector is None:
                    self.assertEqual(sum(parameter.numel() for parameter in kd.parameters()), 0)
                elif all(kd is not previous for previous in projector_modules):
                    projector_modules.append(kd)

                    def capture(gradient, name=group):
                        projector_gradients[name].append(
                            (bool(torch.isfinite(gradient).all()), bool(gradient.abs().sum() > 0)))

                    kd.projector.weight.register_hook(capture)
            values = batch_loss(student, teacher, kd, batch, hard_loss, device)
            for term, weight in (("soft", soft), ("global", global_weight), ("local", local)):
                if weight == 0:
                    self.assertEqual(values[term].item(), 0.0)
            self.assertTrue(all(torch.isfinite(value).item() for value in values.values()))
            return values

        def inspect_optimizer(student, kd, training):
            result = optimizer(student, kd, training)
            self.assertEqual(len(result.param_groups), 3 if kd is not None and kd.projector is not None else 2)
            self.assertEqual(student.training_metadata()["mode"], "full_finetune")
            self.assertTrue(all(parameter.requires_grad for parameter in student.parameters()))
            return result

        def inspect_epoch(student, teacher, kd, *args, **kwargs):
            group = group_of(kd)
            if current["action"] in {"train", "legacy"} and group not in initial_states:
                initial_states[group] = student.trainable_state_dict()
            teacher_before = {name: value.detach().clone() for name, value in teacher.state_dict().items()}
            result = train_epoch(student, teacher, kd, *args, **kwargs)
            self.assert_state_equal(teacher_before, teacher.state_dict())
            self.assertTrue(all(not parameter.requires_grad and parameter.grad is None for parameter in teacher.parameters()))
            return result

        def launcher(job, action, project, directory):
            directory.mkdir(parents=True)
            config = deepcopy(job["config"])
            config["output_root"] = str(directory / "outputs")
            config_path, result_path = directory / "config.json", directory / "result.json"
            config_path.write_text(json.dumps(config))
            arguments = ["--project", str(project), "--config", str(config_path),
                         "--condition", job["condition"], "--result-json", str(result_path)]
            if action == "check":
                arguments.append("--check-only")
            calls.append((action, job["group"], job["seed"]))
            current.update(action=action, group=job["group"])
            try:
                trainer.main(arguments)
            finally:
                current.update(action="idle")
            value = json.loads(result_path.read_text())
            if action == "check":
                receipts[job["group"]] = ablation.coordinator._check_receipt(value)
            return value

        with redirect_stdout(io.StringIO()), mock.patch.object(
            trainer, "build_vit_tiny_student", side_effect=self.student_factory
        ), mock.patch.object(trainer, "_loader", side_effect=inspect_loader), mock.patch.object(
            trainer, "_batch_loss", side_effect=inspect_batch
        ), mock.patch.object(trainer, "_train_epoch", side_effect=inspect_epoch), mock.patch.object(
            trainer, "_optimizer", side_effect=inspect_optimizer
        ):
            # Existing historical full-finetune A/B artifacts retain their
            # original complete checkpoint format and unmodified KD settings.
            old_pair = Path(trainer.run_experiments(self.config, project))
            original_manifest = json.loads((old_pair / "suite.json").read_text())
            explicit_abcd = project / "explicit_abcd"
            explicit_abcd.mkdir()
            source_jobs = {}
            for group, condition in (("A", "supervised"), ("B", "distillation")):
                source_jobs[f"{group}_seed{seed}"] = {
                    "status": "completed", "job": {"group": group, "condition": condition,
                                                     "seed": seed, "student_mode": "full_finetune"},
                    "result": original_manifest["conditions"][condition],
                }
            (explicit_abcd / "suite.json").write_text(json.dumps({
                "format_version": "distillation_abcd_v1", "status": "completed", "jobs": source_jobs,
            }))
            self.settings["reuse_suites"] = [str(explicit_abcd)]
            self.settings_file.write_text(json.dumps(self.settings))
            plan = ablation.make_plan(self.settings_file, project, seeds=[seed])
            self.assertEqual(plan["seeds"], [seed])
            self.assertEqual(len(plan["jobs"]), 6)
            self.assertEqual({job["seed"] for job in plan["jobs"]}, {seed})
            suite = ablation.run_suite(plan, launcher=launcher)
            state = json.loads((suite / "suite.json").read_text())
            records = state["jobs"]
            self.assertEqual(state["status"], "completed", {
                name: value.get("error", value.get("reuse_rejections")) for name, value in records.items()})
            self.assertEqual([group for action, group, _ in calls if action == "train"],
                             ["soft_only", "no_local", "no_global", "no_soft"])
            self.assertEqual(sum(action == "check" for action, _, _ in calls), 6)
            for group in ablation.GROUPS:
                record = records[f"{group}_seed{seed}"]
                self.assertEqual(record["status"], "reused" if group in {"supervised", "full_kd"} else "completed")
                self.assertTrue(record["result"]["teacher_unchanged"])
                check = receipts[group]
                self.assertTrue(check["check_passed"])
                self.assertFalse(check["training_started"])
                self.assertEqual(check["student_mode"], "full_finetune")
                self.assertEqual(check["distillation_trainable_parameters"],
                                 6 * 12 if ablation.GROUPS[group][2] > 0 else 0)
                for index, term in enumerate(("soft", "global", "local"), start=1):
                    if ablation.GROUPS[group][index] == 0:
                        self.assertEqual(check["losses"][term], 0.0)
            self.assertEqual(len({check["initial_state_sha256"] for check in receipts.values()}), 1)
            self.assertTrue(all(check["shared_view"] == receipts["supervised"]["shared_view"]
                                for check in receipts.values()))
            reference_state = next(iter(initial_states.values()))
            reference_view = next(iter(first_training_views.values()))
            self.assertEqual(set(initial_states), set(ablation.GROUPS))
            for group, initial in initial_states.items():
                self.assert_state_equal(reference_state, initial)
                image, label, sample_id = first_training_views[group]
                torch.testing.assert_close(image, reference_view[0], rtol=0, atol=0)
                torch.testing.assert_close(label, reference_view[1], rtol=0, atol=0)
                self.assertEqual(sample_id, reference_view[2])
            for group, gradients in projector_gradients.items():
                if ablation.GROUPS[group][2] > 0:
                    self.assertTrue(gradients and all(finite and nonzero for finite, nonzero in gradients))
                else:
                    self.assertFalse(gradients)

            # A completed source B is still rejected when any complete-KD
            # scientific setting differs; scores do not determine this choice.
            job = next(job for job in plan["jobs"] if job["group"] == "full_kd")
            for key, value in (("temperature", 3.0), ("pool_grid", [6, 6]),
                               ("soft_weight", 0.25), ("global_weight", 0.2),
                               ("local_weight", 0.5)):
                altered = deepcopy(receipts["full_kd"])
                altered["config"]["distillation"][key] = value
                with self.subTest(mismatched_setting=key), self.assertRaisesRegex(ValueError, "distillation settings"):
                    ablation.coordinator.verify_run(old_pair / "distillation", altered, job)

            # Every best student is a standalone complete artifact after both
            # original weight sources and the teacher snapshot are removed.
            self.fixture.teacher_checkpoint.unlink()
            self.fixture.student_checkpoint.unlink()
            shutil.rmtree(self.fixture.teacher_config["model_dir"])
            for group in ablation.GROUPS:
                run = Path(records[f"{group}_seed{seed}"]["result"]["run"])
                restored, payload = trainer.load_student_checkpoint(run / "best.pt", device="cpu")
                self.assert_state_equal(payload["model"], restored.state_dict())
                self.assertEqual(payload["model_metadata"]["mode"], "full_finetune")
                cohort = self.fixture.split.loc[self.fixture.split["split"] == "test"]
                dataset = trainer.inference_dataset(cohort, payload, cache_dir=self.config["cache_dir"])
                inferred = trainer.supervised.predict(restored, DataLoader(dataset, batch_size=3), torch.device("cpu"))
                saved = pd.read_csv(run / "test" / "predictions.csv")
                pd.testing.assert_frame_equal(
                    inferred.sort_values("sample_id").reset_index(drop=True),
                    saved.sort_values("sample_id").reset_index(drop=True),
                    check_exact=False, rtol=0, atol=1e-14,
                )
                history = json.loads((run / "history.json").read_text())
                for term_index, term in enumerate(("soft", "global", "local"), start=1):
                    if ablation.GROUPS[group][term_index] == 0:
                        self.assertTrue(all(row[f"train_{term}_loss"] == 0 for row in history))

        comparison = pd.read_csv(suite / "report" / "comparison.csv")
        summary = pd.read_csv(suite / "report" / "summary.csv")
        self.assertEqual(set(comparison["group"]), set(ablation.GROUPS))
        self.assertEqual(set(summary["group"]), set(ablation.GROUPS))
        self.assertTrue(comparison["ap"].notna().all())
        before = {path: path.read_bytes() for path in suite.rglob("*") if path.is_file()}
        output = io.StringIO()
        with redirect_stdout(output), mock.patch.object(
            trainer, "load_teacher", side_effect=AssertionError("Reporting loaded teacher")
        ), mock.patch.object(trainer.supervised, "prepare", side_effect=AssertionError("Reporting read cohort")), mock.patch.object(
            torch, "load", side_effect=AssertionError("Reporting loaded weights")
        ):
            self.assertEqual(ablation.main(["--project", str(project), "--config", str(self.settings_file),
                                            "--show-metrics", "--suite", str(suite)]), 0)
        self.assertEqual(before, {path: path.read_bytes() for path in suite.rglob("*") if path.is_file()})
        self.assertIn("soft_only", output.getvalue())
        self.assertIn("no_global", output.getvalue())


if __name__ == "__main__":
    unittest.main()
