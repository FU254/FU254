"""Loss-ablation planning, saved-run provenance, scheduling, and reporting."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import unittest
from unittest import mock

from foundation_sft import distillation_ablation as ablation
from foundation_sft import distillation_suite as coordinator
from tests import test_distillation_suite as fixtures

put = fixtures.put


class DistillationAblationTests(unittest.TestCase):
    def setUp(self):
        # The fake child writes real JSON provenance/checkpoint artifacts on
        # disk. Its verifier exercises the production provenance checks.
        self.fixture = fixtures.DistillationSuiteTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        source = Path(__file__).resolve().parents[1] / "foundation_sft/configs/distillation_ablation.json"
        self.settings = coordinator._read(source)
        self.settings.update(template="full.json", output_root="runs/ablation", reuse_suites=[])
        put(self.root / "ablation.json", self.settings)

    def plan(self, **kwargs):
        return ablation.make_plan("ablation.json", self.root, **kwargs)

    def execute(self, plan, **kwargs):
        with redirect_stdout(io.StringIO()):
            path = ablation.run_suite(plan, launcher=self.fixture.launcher, verifier=self.fixture.verifier, **kwargs)
        return path, coordinator._read(path / "suite.json")

    def source_abcd(self, seeds=(42, 43, 44)):
        plan = self.fixture.plan(groups=["A", "B", "C", "D"], seeds=list(seeds))
        source = self.root / "explicit_abcd"
        records = {}
        for job in plan["jobs"]:
            result = self.fixture.saved_run(source / "saved" / job["id"], job)
            records[job["id"]] = {"status": "completed", "job": job, "result": result}
        put(source / "suite.json", {"format_version": coordinator.FORMAT, "jobs": records})
        return source

    def test_fixed_eighteen_jobs_only_weights_and_runtime_fields_vary(self):
        plan = self.plan()
        self.assertEqual(plan["format_version"], ablation.FORMAT)
        self.assertEqual(len(plan["jobs"]), 18)
        original = coordinator._read(self.root / "full.json")
        for job in plan["jobs"]:
            condition, *weights = ablation.GROUPS[job["group"]]
            self.assertEqual(job["condition"], condition)
            self.assertEqual(job["student_mode"], "full_finetune")
            self.assertEqual([job["config"]["distillation"][key] for key in ablation.WEIGHTS], weights)
            self.assertEqual(job["config"]["training"], original["training"])
            restored = deepcopy(job["config"])
            for key in ("seed", "device", "output_root"):
                restored.pop(key)
            restored["distillation"] = original["distillation"]
            self.assertEqual(restored, original)
        self.assertEqual([job["device"] for job in plan["jobs"][::3]],
                         ["cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:0", "cuda:1"])

    def test_plan_subset_no_write_and_no_torch_report(self):
        before = set(self.root.rglob("*"))
        plan = self.plan(groups=["no_soft", "full_kd"], seeds=[44], devices=["cpu"])
        self.assertEqual([job["id"] for job in plan["jobs"]], ["no_soft_seed44", "full_kd_seed44"])
        with redirect_stdout(io.StringIO()), mock.patch.dict("sys.modules", {"torch": None}):
            self.assertEqual(ablation.main(["--project", str(self.root), "--config", "ablation.json", "--plan"]), 0)
            self.assertEqual(ablation.main(["--project", str(self.root), "--config", "ablation.json", "--show-metrics"]), 0)
        self.assertEqual(before, set(self.root.rglob("*")))

    def test_fixed_weights_temperature_mode_and_training_reject_bad_configs(self):
        cases = []
        settings = deepcopy(self.settings)
        settings["groups"]["no_local"]["local_weight"] = .5
        cases.append(("settings", settings))
        for change in ("temperature", "pool_grid", "student", "training"):
            template = deepcopy(self.fixture.template)
            if change == "temperature":
                template["distillation"]["temperature"] = 3
            elif change == "pool_grid":
                template["distillation"]["pool_grid"] = [4, 4]
            elif change == "student":
                template["student"] = {"mode": "lora", "rank": 8, "alpha": 16}
            else:
                template["training"]["backbone_lr"] = 1e-5
            cases.append((change, template))
        for name, value in cases:
            with self.subTest(name=name):
                put(self.root / "ablation.json", value if name == "settings" else self.settings)
                put(self.root / "full.json", self.fixture.template if name == "settings" else value)
                with self.assertRaises(ValueError):
                    self.plan()

    def test_explicit_abcd_reuses_six_matching_controls_and_trains_twelve(self):
        source = self.source_abcd()
        plan = self.plan(reuse_suites=[str(source)], devices=["cpu", "cpu"])
        path, state = self.execute(plan)
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["format_version"], ablation.FORMAT)
        reused = [record for record in state["jobs"].values() if record["status"] == "reused"]
        self.assertEqual(len(reused), 6)
        self.assertEqual({record["job"]["group"] for record in reused}, {"supervised", "full_kd"})
        self.assertEqual(sum(action == "check" for action, _ in self.fixture.calls), 18)
        self.assertEqual(sum(action == "train" for action, _ in self.fixture.calls), 12)
        summary = coordinator.summary_rows(coordinator.metric_rows(state))
        self.assertEqual([row["group"] for row in summary], list(ablation.GROUPS))
        self.assertTrue(all(row["n_completed"] == 3 for row in summary))
        rows = coordinator.metric_rows(state)
        for row in rows:
            self.assertEqual(tuple(row[key] for key in ablation.WEIGHTS), ablation.GROUPS[row["group"]][1:])
        self.assertTrue((path / "report" / "comparison.csv").is_file())
        self.assertIn("global_weight", (path / "report" / "summary.csv").read_text())
        before = {file: file.read_bytes() for file in path.rglob("*") if file.is_file()}
        with redirect_stdout(io.StringIO()), mock.patch.dict("sys.modules", {"torch": None}):
            self.assertEqual(ablation.main(["--project", str(self.root), "--config", "ablation.json",
                                           "--show-metrics", "--suite", str(path)]), 0)
        self.assertTrue(all(file.read_bytes() == data for file, data in before.items()))

    def test_wrong_kd_full_student_config_cannot_be_reused(self):
        source = self.source_abcd(seeds=(42,))
        run = source / "saved" / "B_seed42"
        config = coordinator._read(run / "config.json")
        config["distillation"]["soft_weight"] = .25
        put(run / "config.json", config)
        path, state = self.execute(self.plan(groups=["full_kd"], seeds=[42], reuse_suites=[str(source)]))
        record = state["jobs"]["full_kd_seed42"]
        self.assertEqual(record["status"], "completed")
        self.assertIn("distillation settings", record["reuse_rejections"][0]["reason"])
        self.assertNotEqual(Path(record["result"]["run"]), run)

    def test_abcd_candidate_filter_ignores_failed_wrong_seed_and_lora(self):
        source = self.source_abcd(seeds=(42, 43))
        saved = coordinator._read(source / "suite.json")
        saved["jobs"]["B_seed42"]["status"] = "failed"
        put(source / "suite.json", saved)
        self.assertEqual(list(coordinator._reuse_candidates([source], "distillation", 42, "full_finetune")), [])
        self.assertEqual(list(coordinator._reuse_candidates([source], "distillation", 43, "full_finetune")),
                         [source / "saved" / "B_seed43"])
        # Legacy caller still receives candidates; filtering never uses test scores.
        all_runs = list(coordinator._reuse_candidates([source], "distillation"))
        self.assertEqual(len(all_runs), 3)

    def test_no_reuse_trains_fresh_and_reuse_append_is_ordered(self):
        source = self.source_abcd(seeds=(42,))
        self.settings["reuse_suites"] = [str(source)]
        put(self.root / "ablation.json", self.settings)
        plan = self.plan(groups=["supervised", "full_kd"], seeds=[42], no_reuse=True)
        self.assertEqual(plan["reuse_suites"], [])
        _, state = self.execute(plan)
        self.assertTrue(all(record["status"] == "completed" for record in state["jobs"].values()))
        appended = self.plan(reuse_suites=[str(source), "other_explicit_suite"])
        self.assertEqual(appended["reuse_suites"], [str(source), str(self.root / "other_explicit_suite")])
        with self.assertRaises(ValueError):
            self.plan(no_reuse=True, reuse_suites=[str(source)])

    def test_partial_failures_are_reported_and_resume_retries_only_failed(self):
        normal = self.fixture.launcher
        def failed(job, action, project, directory):
            if job["group"] == "no_global" and action == "train":
                raise RuntimeError("synthetic failure")
            return normal(job, action, project, directory)
        plan = self.plan(groups=["soft_only", "no_global"], seeds=[42], devices=["cpu"])
        with redirect_stdout(io.StringIO()):
            path = ablation.run_suite(plan, launcher=failed, verifier=self.fixture.verifier)
        state = coordinator._read(path / "suite.json")
        self.assertEqual(state["status"], "partial")
        self.assertEqual(state["jobs"]["soft_only_seed42"]["status"], "completed")
        self.assertEqual(state["jobs"]["no_global_seed42"]["status"], "failed")
        self.fixture.calls.clear()
        _, state = self.execute(plan, resume_suite=path)
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["jobs"]["soft_only_seed42"]["status"], "resumed")
        self.assertEqual(sum(action == "train" for action, _ in self.fixture.calls), 1)
        altered = deepcopy(plan)
        altered["format_version"] = coordinator.FORMAT
        with self.assertRaisesRegex(ValueError, "differs"):
            self.execute(altered, resume_suite=path)

    def test_check_only_preserves_zero_weights_and_does_not_train(self):
        path, state = self.execute(self.plan(seeds=[42], devices=["cpu"]), check_only=True)
        self.assertEqual(state["status"], "checked")
        self.assertTrue(all(action == "check" for action, _ in self.fixture.calls))
        self.assertEqual(len(coordinator.summary_rows(coordinator.metric_rows(state))), 6)
        self.assertTrue((path / "report" / "paired_summary.csv").is_file())


if __name__ == "__main__":
    unittest.main()
