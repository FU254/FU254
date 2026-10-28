"""One shared-intensity candidate through the existing partial-study queue."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest

from foundation_sft import partial_finetune_study as study
from tests import test_partial_finetune_study as freeze_fixture


class PartialAugmentationStudyTests(unittest.TestCase):
    def setUp(self):
        self.freeze = freeze_fixture.PartialFinetuneStudyTests(methodName="runTest")
        self.freeze.setUp()
        self.addCleanup(self.freeze.doCleanups)
        self.freeze.scores = {"last8":.91,"last4":.85,"last2":.83}
        self.source = self.freeze.execute()
        self.root, self.repo = self.freeze.root, self.freeze.repo
        self.saved = json.loads((self.source/"study.json").read_text())
        self.baseline_check = deepcopy(self.saved["candidates"]["last8"]["check"])
        self.source_bytes = {str(p.relative_to(self.source)):p.read_bytes()
                             for p in self.source.rglob("*") if p.is_file()}
        self.config = self.root/"foundation_sft/configs/partial_finetune_augmentation.json"
        settings = {"study_type":"intensity_augmentation","source_study":str(self.source),
            "seeds":[42],"brightness_jitter":.1,"contrast_jitter":.1,
            "devices":["cuda:0"],"max_parallel":1,"cache_dir":None,
            "output_root":"runs/partial_finetune_augmentation"}
        self.config.write_text(json.dumps(settings))
        self.freeze.calls.clear()
        self.freeze.scores["augmented"] = .92
        self.freeze.counts["augmented"] = 3_559_489
        self.score = .92
        self.failure = None
        self.defect = None

    def plan(self, **kwargs):
        return study.make_plan(self.config,self.root,**kwargs)

    def check(self, job):
        result = deepcopy(self.baseline_check)
        result["config"] = deepcopy(job["config"])
        result["config"]["evaluation"] = deepcopy(self.baseline_check["config"]["evaluation"])
        if self.defect:
            self.defect(job,result)
        return result

    def launch(self, job, action, project, directory):
        if action=="evaluate":
            raise AssertionError("Augmentation screening must not evaluate test")
        if action=="check":
            self.freeze.calls.append((action,job["id"],job["config"]["phase"]))
            return self.check(job)
        if self.failure:
            self.freeze.calls.append((action,job["id"],job["config"]["phase"]))
            raise RuntimeError("fixture interrupted augmentation")
        self.freeze.scores["augmented"] = self.score
        return self.freeze.launch(job,action,project,directory)

    def execute(self, plan=None, **kwargs):
        with redirect_stdout(io.StringIO()):
            return study.run_study(plan or self.plan(),launcher=self.launch,
                verifier=self.freeze.verify,reporter=lambda *_:None,**kwargs)

    def assert_source_unchanged(self):
        self.assertEqual(self.source_bytes,{str(p.relative_to(self.source)):p.read_bytes()
                         for p in self.source.rglob("*") if p.is_file()})

    def test_plan_reuses_last8_and_changes_only_two_intensity_settings(self):
        plan=self.plan()
        self.assertEqual(plan["new_optimization_trainings"],1)
        self.assertEqual(plan["evaluation_policy"],"validation_only")
        self.assertEqual([job["id"] for job in plan["jobs"]],["augmented"])
        base, candidate=plan["baseline"]["job"]["config"],plan["jobs"][0]["config"]
        self.assertEqual(base["student"],{"mode":"partial_finetune","train_last_blocks":8})
        self.assertEqual(candidate["student"],base["student"])
        self.assertEqual(candidate["student_checkpoint"],base["student_checkpoint"])
        expected=deepcopy(base["training"])
        expected.update(brightness_jitter=.1,contrast_jitter=.1)
        self.assertEqual(candidate["training"],expected)
        for name in ("teachers","seed","distillation","method"):
            self.assertEqual(candidate[name],base[name])
        self.assertEqual(candidate["phase"],"pilot")
        self.assertNotIn("selected_checkpoint_sha256",candidate)
        self.assertEqual(study.protocol_hash(plan),study.protocol_hash(self.plan(devices=["cuda:4"])))
        self.assert_source_unchanged()

    def test_one_training_validation_only_and_completed_resume_does_not_repeat(self):
        path=self.execute()
        state=json.loads((path/"study.json").read_text())
        self.assertEqual(state["status"],"completed")
        self.assertEqual(state["selection"]["arm"],"augmented")
        self.assertFalse(state["selection"]["test_metrics_used"])
        self.assertEqual(state["baseline"]["status"],"reused")
        self.assertEqual(state["selected_evaluation"]["status"],"not_requested")
        self.assertEqual([c[1] for c in self.freeze.calls if c[0]=="train"],["augmented"])
        result=state["candidates"]["augmented"]["result"]
        self.assertNotIn("test_patient",result)
        self.assertFalse((Path(result["run"])/"evaluation_result.json").exists())
        self.freeze.calls.clear()
        self.execute(self.plan(devices=["cuda:3"]),resume_study=path)
        self.assertEqual([c[0] for c in self.freeze.calls],["check","check"])
        self.assert_source_unchanged()

    def test_exact_validation_tie_retains_baseline_and_test_scores_do_not_select(self):
        self.score=.91
        path=self.execute()
        state=json.loads((path/"study.json").read_text())
        self.assertEqual(state["selection"]["arm"],"baseline")
        previous=deepcopy(state["selection"])
        state["baseline"]["result"]["test_patient"]["roc_auc"]=0.
        self.assertEqual(study.select_setting(state),previous)
        self.assertEqual(state["selected_evaluation"]["status"],"not_requested")

    def test_failed_new_training_can_retry_without_retraining_baseline(self):
        self.failure=True
        path=self.execute()
        state=json.loads((path/"study.json").read_text())
        self.assertEqual(state["status"],"partial")
        self.assertEqual(state["selection"]["status"],"incomplete")
        self.failure=False
        self.freeze.calls.clear()
        self.execute(resume_study=path)
        self.assertEqual([c[1] for c in self.freeze.calls if c[0]=="train"],["augmented"])
        self.assert_source_unchanged()

    def test_undeclared_learning_rate_geometry_or_frozen_tensors_block_training(self):
        for field in ("lr","geometry","frozen"):
            with self.subTest(field=field):
                self.freeze.calls.clear()
                def defect(job,check):
                    if job["id"]!="augmented":return
                    if field=="lr":check["config"]["training"]["backbone_lr"]=1e-5
                    elif field=="geometry":check["shared_view"]={"different_crop":True}
                    else:check["frozen_parameter_sha256"]="0"*64
                self.defect=defect
                path=self.execute()
                state=json.loads((path/"study.json").read_text())
                self.assertEqual(state["status"],"partial")
                self.assertFalse(any(c[0]=="train" for c in self.freeze.calls))
        self.assert_source_unchanged()

    def test_check_only_and_source_receipt_mismatch_fail_closed(self):
        path=self.execute(check_only=True)
        self.assertEqual(json.loads((path/"study.json").read_text())["status"],"checked")
        self.assertFalse(any(c[0]=="train" for c in self.freeze.calls))
        target=self.source/"study.json"
        state=json.loads(target.read_text())
        state["selected_evaluation"]["result"]["checkpoint_sha256"]="f"*64
        target.write_text(json.dumps(state))
        with self.assertRaises(ValueError):self.plan()

    def test_configuration_and_stdlib_cli(self):
        original=json.loads(self.config.read_text())
        for key,value in (("brightness_jitter",True),("contrast_jitter",float("nan")),
                          ("seeds",[43]),("learning_rate",.01)):
            state=deepcopy(original);state[key]=value
            self.config.write_text(json.dumps(state))
            with self.assertRaises(ValueError):self.plan()
        self.config.write_text(json.dumps(original))
        result=subprocess.run([sys.executable,"-S",str(self.repo/"run_partial_finetune_experiments.py"),
            "--project",str(self.root),"--config",str(self.config),"--plan"],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result.stdout)["new_optimization_trainings"],1)


if __name__=="__main__":unittest.main()
