"""Exercise the coordinator against the real trainer and synthetic medical data."""

from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_single_teacher_distillation as fixtures
from test_student_lora_distillation import SmallLoRATokenStudent
from foundation_sft import distillation_suite as coordinator


class ABCDTrainerIntegrationTests(unittest.TestCase):
    def test_old_pair_reuse_new_lora_training_resume_and_file_only_reporting(self):
        fixture = fixtures.SingleTeacherDistillationTests(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        trainer, project = fixture.kdtrain, fixture.root
        source_sha = hashlib.sha256(fixture.student_checkpoint.read_bytes()).hexdigest()

        def student_factory(checkpoint=None, *, initialization=None, **options):
            student = SmallLoRATokenStudent(checkpoint, initialization=initialization, **options)
            if initialization is None:
                student.initialization["checkpoint_sha256"] = source_sha
            return student

        config = deepcopy(fixture.config)
        config["training"].update(backbone_lr=5e-5, head_lr=1e-4, projection_lr=1e-4)
        templates = {}
        for mode in ("full_finetune", "lora"):
            template = deepcopy(config)
            if mode == "lora":
                template["student"] = {"mode": "lora", "rank": 2, "alpha": 4.0}
                template["training"]["lora_lr"] = 5e-5
            path = project / (mode + ".json")
            path.write_text(json.dumps(template))
            templates[mode] = str(path)

        calls = []

        def launcher(job, action, project, directory):
            directory.mkdir(parents=True)
            settings = deepcopy(job["config"])
            settings["output_root"] = str(directory / "outputs")
            config_file, result = directory / "config.json", directory / "result.json"
            config_file.write_text(json.dumps(settings))
            args = ["--project", str(project), "--config", str(config_file),
                    "--condition", job["condition"], "--result-json", str(result)]
            if action == "check":
                args.append("--check-only")
            calls.append((action, job["group"]))
            trainer.main(args)
            return json.loads(result.read_text())

        with redirect_stdout(io.StringIO()), mock.patch.object(
            trainer, "build_vit_tiny_student", side_effect=student_factory
        ):
            old_pair = trainer.run_experiments(config, project)
            settings = {
                "output_root": str(project / "abcd"), "seeds": [config["seed"]],
                "devices": ["cpu"], "max_parallel": 1, "reuse_suites": [str(old_pair)],
                "groups": {group: {"student_mode": mode, "condition": condition,
                                   "config": templates[mode]}
                           for group, (mode, condition) in coordinator.GROUPS.items()},
            }
            settings_file = project / "abcd.json"
            settings_file.write_text(json.dumps(settings))
            plan = coordinator.make_plan(settings_file, project)
            suite = coordinator.run_suite(plan, launcher=launcher)
            state = json.loads((suite / "suite.json").read_text())
            records = state["jobs"]
            self.assertEqual(state["status"], "completed", {key: value.get("error", value.get("reuse_rejections"))
                                                           for key, value in records.items()})
            self.assertEqual([records[f"{group}_seed{config['seed']}"]["status"]
                              for group in "ABCD"], ["reused", "reused", "completed", "completed"])
            self.assertEqual([group for action, group in calls if action == "train"], ["C", "D"])
            rows = coordinator.metric_rows(state)
            self.assertTrue(all(isinstance(row["ap"], float) for row in rows))
            self.assertEqual(len(coordinator.summary_rows(rows)), 4)
            for group in "ABCD":
                self.assertEqual(records[f"{group}_seed{config['seed']}"]["result"]["teacher_unchanged"], True)
            old_runs = {key: value["result"]["run"] for key, value in records.items()}
            calls.clear()
            coordinator.run_suite(plan, resume_suite=suite, launcher=launcher)
            resumed = json.loads((suite / "suite.json").read_text())
            self.assertFalse(any(action == "train" for action, group in calls))
            self.assertTrue(all(value["status"] == "resumed" for value in resumed["jobs"].values()))
            self.assertEqual(old_runs, {key: value["result"]["run"] for key, value in resumed["jobs"].items()})
            before = {path: path.read_bytes() for path in suite.rglob("*") if path.is_file()}
            with mock.patch.object(trainer, "load_teacher", side_effect=AssertionError("Report loaded teacher")):
                self.assertEqual(coordinator.main(["--project", str(project), "--config", str(settings_file),
                                                   "--show-metrics", "--suite", str(suite)]), 0)
            self.assertEqual(before, {path: path.read_bytes() for path in suite.rglob("*") if path.is_file()})


if __name__ == "__main__":
    unittest.main()
