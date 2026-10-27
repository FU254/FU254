"""Three-candidate screening, fixed-source reuse and validation-only selection."""
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest

from foundation_sft import partial_finetune_study as study
from foundation_sft import teacher_weight_study as original
from tests import test_teacher_weight_study as source_fixture


class PartialFinetuneStudyTests(unittest.TestCase):
    def setUp(self):
        self.base = source_fixture.TeacherWeightStudyTests(methodName='runTest')
        self.base.setUp()
        self.addCleanup(self.base.doCleanups)
        self.root = self.base.root
        self.repo = self.base.repo
        self.source = self.base.execute()
        state_path = self.source / 'study.json'
        source = json.loads(state_path.read_text())
        # This study deliberately uses the actual user's selected 5e-5 P-E
        # recipe, rather than the older fixture's 1e-5 winner.
        for record in source['pilots'].values():
            if record['job']['student_initialization'] == 'imagenet':
                record['result']['best_val_patient_roc_auc'] = (
                    .8864 if record['job']['config']['training']['backbone_lr'] == 5e-5 else .85)
        source['selection'] = original.select_learning_rates(source)
        baseline_job = next(j for j in original._formal_jobs(source, source['plan']) if j['id'] == 'P-E')
        selected_pilot = source['pilots'][source['selection']['imagenet']['job_id']]
        baseline_result = deepcopy(selected_pilot['result'])
        baseline_result.update(phase='main', source_phase='pilot', student_trainable_parameters=5_524_609,
            test_patient={'roc_auc':.8493, 'pr_auc':.5182, 'average_precision':.5214,
                          'sensitivity':.625, 'specificity':.7892})
        baseline_check = self.base.check(baseline_job)
        baseline_check['model_metadata'].update(mode='full_finetune', total_parameters=5_524_609,
            trainable_parameters=5_524_609)
        source['main']['P-E'].update(job=baseline_job, result=baseline_result,
            check=baseline_check, status='completed')
        state_path.write_text(json.dumps(source))
        self.baseline_check = baseline_check
        self.baseline_result = baseline_result
        self._history(Path(baseline_result['run']), .8864)
        self.source_bytes = {str(p.relative_to(self.source)):p.read_bytes() for p in self.source.rglob('*') if p.is_file()}
        self.config = self.root / study.DEFAULT_CONFIG
        self.config.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.repo / study.DEFAULT_CONFIG, self.config)
        settings = json.loads(self.config.read_text())
        settings['source_study'] = str(self.source)
        self.config.write_text(json.dumps(settings))
        self.calls = []
        self.receipts = {(baseline_result['run'], 'main'):deepcopy(baseline_result)}
        self.scores = {'last8':.89, 'last4':.9, 'last2':.88}
        self.counts = {'last8':3_559_489, 'last4':1_780_033, 'last2':890_305}
        self.failure = None
        self.defect = None

    @staticmethod
    def _history(run, score):
        run.mkdir(parents=True, exist_ok=True)
        rows = [{'epoch':index, 'train_eval_loss':.07-index*.01,
            'val_loss':.08-index*.002, 'train_eval_patient_roc_auc':.9+index*.02,
            'val_patient_roc_auc':auc, 'train_eval_patient_pr_auc':.6+index*.05,
            'val_patient_pr_auc':.4+index*.02, 'train_loss':.2-index*.01,
            'train_hard_loss':.03, 'train_kd_loss':.1}
            for index, auc in ((1,score-.01),(2,score),(3,score-.02))]
        (run/'history.json').write_text(json.dumps(rows))

    def plan(self, **kwargs):
        return study.make_plan(self.config, self.root, **kwargs)

    def check(self, job):
        value = deepcopy(self.baseline_check)
        value['config'] = deepcopy(job['config'])
        value['config']['evaluation'] = deepcopy(self.baseline_check['config']['evaluation'])
        if job['id'] != 'baseline':
            value['model_metadata'].update(mode='partial_finetune',
                trainable_parameters=self.counts[job['id']],
                partial_finetune={'train_last_blocks':job['config']['student']['train_last_blocks']})
            value.update(partial_frozen_state_verified=True, partial_optimizer_step_verified=True,
                feature_kd_gradients_verified={'soft':True,'global':True,'local':True},
                frozen_parameter_sha256='f'*64)
        if self.defect:
            self.defect(job, value)
        return value

    def launch(self, job, action, project, directory):
        self.calls.append((action, job['id'], job['config']['phase']))
        if self.failure == (job['id'], action):
            raise RuntimeError('fixture interrupted worker')
        if action == 'check':
            return self.check(job)
        if action == 'evaluate':
            run = job['evaluation_run']
            source = deepcopy(self.receipts[(run, 'pilot')])
            immutable = {name:(Path(run)/name).read_bytes() for name in ('best.pt','result.json')}
            source.update(phase='main', source_phase='pilot',
                test_patient={'roc_auc':.85,'pr_auc':.53,'average_precision':.54,
                              'sensitivity':.7,'specificity':.8})
            (Path(run)/'evaluation_result.json').write_text(json.dumps(source))
            self.receipts[(run,'main')] = deepcopy(source)
            for name, data in immutable.items():
                self.assertEqual((Path(run)/name).read_bytes(), data)
            return source
        self.assertEqual(action, 'train')
        self.assertEqual(job['config']['phase'], 'pilot')
        run = directory / 'run'
        run.mkdir(parents=True)
        data = ('selected student '+job['id']).encode()
        (run/'best.pt').write_bytes(data)
        result = {'status':'completed','phase':'pilot','run':str(run),
            'checkpoint_sha256':hashlib.sha256(data).hexdigest(),
            'best_val_patient_roc_auc':self.scores[job['id']],
            'student_trainable_parameters':self.counts[job['id']]}
        (run/'result.json').write_text(json.dumps(result))
        self._history(run, result['best_val_patient_roc_auc'])
        self.receipts[(str(run),'pilot')] = deepcopy(result)
        return result

    def verify(self, run, check, job):
        phase = job['config']['phase']
        result = deepcopy(self.receipts[(str(run),phase)])
        self.assertEqual(check['config']['seed'],42)
        self.assertEqual(check['config']['method'],'equal')
        self.assertEqual(result['phase'],phase)
        return result

    def execute(self, plan=None, **kwargs):
        with redirect_stdout(io.StringIO()):
            return study.run_study(plan or self.plan(), launcher=self.launch,
                verifier=self.verify, reporter=lambda *_:None, **kwargs)

    def source_unchanged(self):
        self.assertEqual({str(p.relative_to(self.source)):p.read_bytes() for p in self.source.rglob('*') if p.is_file()}, self.source_bytes)

    def test_plan_reuses_exact_source_and_only_changes_update_range(self):
        before = {str(p):p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        plan = self.plan(devices=['cuda:0','cuda:3','cuda:4'],max_parallel=3)
        self.assertEqual(plan['new_optimization_trainings'],3)
        self.assertEqual([j['id'] for j in plan['jobs']],['last8','last4','last2'])
        self.assertEqual([j['device'] for j in plan['jobs']],['cuda:0','cuda:3','cuda:4'])
        self.assertEqual([j['config']['student']['train_last_blocks'] for j in plan['jobs']],[8,4,2])
        for job in plan['jobs']:
            self.assertEqual(job['config']['phase'],'pilot')
            self.assertEqual(job['config']['training'],plan['baseline']['job']['config']['training'])
            self.assertEqual(job['config']['student_checkpoint'],plan['baseline']['job']['config']['student_checkpoint'])
            self.assertNotIn('selected_checkpoint_sha256',job['config'])
        self.assertEqual(study.protocol_hash(plan),study.protocol_hash(self.plan(devices=['cuda:0'],max_parallel=1)))
        self.assertEqual(before,{str(p):p.read_bytes() for p in self.root.rglob('*') if p.is_file()})

    def test_three_trainings_one_selected_evaluation_and_immutable_pilot_receipts(self):
        path = self.execute()
        state = json.loads((path/'study.json').read_text())
        self.assertEqual(state['status'],'completed')
        self.assertEqual(sorted(c[1] for c in self.calls if c[0]=='train'),['last2','last4','last8'])
        self.assertEqual([c[1] for c in self.calls if c[0]=='evaluate'],['last4_selected'])
        self.assertEqual(state['selection']['arm'],'last4')
        self.assertFalse(state['selection']['test_metrics_used'])
        self.assertEqual(state['baseline']['status'],'reused')
        for arm, entry in state['candidates'].items():
            self.assertEqual(entry['result']['phase'],'pilot')
            self.assertNotIn('test_patient',entry['result'])
            self.assertNotIn('test_patient',json.loads((Path(entry['result']['run'])/'result.json').read_text()))
        self.assertFalse((Path(state['candidates']['last2']['result']['run'])/'evaluation_result.json').exists())
        self.source_unchanged()

    def test_baseline_winner_requires_no_new_test_evaluation(self):
        self.scores={'last8':.84,'last4':.85,'last2':.86}
        path=self.execute()
        state=json.loads((path/'study.json').read_text())
        self.assertEqual(state['selection']['arm'],'baseline')
        self.assertEqual(state['selected_evaluation']['status'],'reused')
        self.assertEqual(state['selected_evaluation']['result'],self.baseline_result)
        self.assertEqual(len([c for c in self.calls if c[0]=='train']),3)
        self.assertFalse(any(c[0]=='evaluate' for c in self.calls))

    def test_validation_ties_use_actual_trainable_counts_and_ignore_test_scores(self):
        self.scores={'last8':.9,'last4':.9,'last2':.9}
        path=self.execute()
        state=json.loads((path/'study.json').read_text())
        self.assertEqual(state['selection']['arm'],'last2')
        state['baseline']['result']['test_patient']['roc_auc']=1.
        self.assertEqual(study.select_setting(state),state['selection'])
        state['candidates']['last4']['check']['model_metadata']['trainable_parameters']=100
        self.assertEqual(study.select_setting(state)['arm'],'last4')
        state['candidates']['last4']['result']['test_patient']={'roc_auc':1.}
        with self.assertRaisesRegex(ValueError,'validation-only'):
            study.select_setting(state)

    def test_failed_candidate_blocks_selection_and_resume_only_trains_failed_arm(self):
        self.failure=('last2','train')
        path=self.execute()
        state=json.loads((path/'study.json').read_text())
        self.assertEqual(state['status'],'partial')
        self.assertEqual(state['selection']['status'],'incomplete')
        self.assertFalse(any(c[0]=='evaluate' for c in self.calls))
        self.failure=None; self.calls.clear()
        self.execute(resume_study=path)
        self.assertEqual([c[1] for c in self.calls if c[0]=='train'],['last2'])
        state=json.loads((path/'study.json').read_text())
        self.assertEqual(state['status'],'completed')
        self.assertEqual(state['candidates']['last4']['status'],'resumed')
        self.assertTrue(any(a['status']=='failed' for a in state['candidates']['last2']['attempts']))
        self.source_unchanged()

    def test_complete_resume_rechecks_without_retraining_or_retesting(self):
        path=self.execute()
        first=json.loads((path/'study.json').read_text())
        receipt=deepcopy(first['selected_evaluation']['result'])
        self.calls.clear()
        self.execute(self.plan(devices=['cuda:4'],max_parallel=1),resume_study=path)
        self.assertEqual(len(self.calls),4)
        self.assertTrue(all(action=='check' for action,_,_ in self.calls))
        state=json.loads((path/'study.json').read_text())
        self.assertEqual(state['status'],'completed')
        self.assertEqual(state['selected_evaluation']['result'],receipt)
        self.assertEqual(state['selected_evaluation']['status'],'resumed')

    def test_check_only_then_resume_preserves_source_and_does_not_optimize_early(self):
        path=self.execute(check_only=True)
        state=json.loads((path/'study.json').read_text())
        self.assertEqual(state['status'],'checked')
        self.assertEqual(state['selection']['status'],'incomplete')
        self.assertTrue(all(action=='check' for action,_,_ in self.calls))
        self.calls.clear();self.execute(resume_study=path)
        self.assertEqual(len([c for c in self.calls if c[0]=='train']),3)
        self.source_unchanged()

    def test_changed_initial_tensors_cohort_or_view_block_all_candidate_optimization(self):
        for field,value in (('head_initial_state_sha256','0'*64),('cohort_hash','wrong'),('shared_view',{'wrong':True})):
            with self.subTest(field=field):
                self.calls.clear()
                def change(job, check):
                    if job['id']!='baseline':check[field]=value
                self.defect=change
                path=self.execute()
                state=json.loads((path/'study.json').read_text())
                self.assertEqual(state['selection']['status'],'incomplete')
                self.assertFalse(any(action=='train' for action,_,_ in self.calls))
        self.source_unchanged()

    def test_source_protocol_fingerprint_and_screening_seed_changes_are_rejected(self):
        path=self.source/'study.json'
        raw=path.read_bytes()
        for defect in ('protocol','checkpoint','seed'):
            with self.subTest(defect=defect):
                state=json.loads(raw)
                if defect=='protocol':state['protocol_sha256']='0'*64
                elif defect=='checkpoint':state['main']['P-E']['result']['checkpoint_sha256']='f'*64
                else:state['main']['P-E']['job']['config']['seed']=43
                path.write_text(json.dumps(state))
                with self.assertRaises(ValueError):self.plan()
        path.write_bytes(raw)
        config=json.loads(self.config.read_text());config['seeds']=[43]
        self.config.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError,'seed42'):self.plan()
        self.assertFalse(self.calls)

    def test_plan_and_saved_reporting_cli_need_no_torch(self):
        entry=self.repo/'run_partial_finetune_experiments.py'
        command=[sys.executable,'-S',str(entry),'--project',str(self.root),'--source-study',str(self.source),'--plan']
        result=subprocess.run(command,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result.stdout)['new_optimization_trainings'],3)
        path=self.execute()
        result=subprocess.run([sys.executable,'-S',str(entry),'--project',str(self.root),
            '--show-metrics','--study',str(path)],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        for label in ('Screening is validation only','sensitivity','specificity','Single seed42'):
            self.assertIn(label,result.stdout)


if __name__=='__main__':
    unittest.main()
