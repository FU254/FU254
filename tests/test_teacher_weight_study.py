"""Exact two-stage job counts, validation-only selection and restart safety."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from foundation_sft import teacher_weight_study as study


class TeacherWeightStudyTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(prefix='teacher-weight-study-')
        self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)
        self.repo=Path(__file__).resolve().parents[1]
        for relative in (study.DEFAULT_CONFIG,'foundation_sft/configs/equal_teacher_distillation.json',
                'foundation_sft/configs/distillation_rad_dino_vit_tiny.json'):
            target=self.root/relative;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(self.repo/relative,target)
        self.config=self.root/study.DEFAULT_CONFIG
        self.calls=[];self.receipts={};self.failure=None;self.defect=None

    def plan(self,**kwargs):
        return study.make_plan(self.config,self.root,**kwargs)

    def check(self,job):
        initialization=job['student_initialization']
        result={'check_passed':True,'config':deepcopy(job['config']),
            'model_metadata':{'model':'tiny','initialization':{'mode':initialization}},
            'initial_state_sha256':('a' if initialization=='random' else 'b')*64,
            'head_initial_state_sha256':'c'*64,'distillation_initial_state_sha256':'d'*64,
            'cohort_hash':'fixed','teacher':{'checkpoint_sha256':'e'*64},
            'teachers':[{'model':name,'checkpoint_sha256':'e'*64} for name in study.TEACHERS],
            'shared_view':{'same':True},'teacher_views':{'same':True},'teacher_unchanged':True,
            'student_backbone_gradient_verified':True,'projector_gradients_verified':True,
            'policy_gradient_verified':job['method']=='rl'}
        result['config']['evaluation']={'bootstrap_samples':10}
        if self.defect:self.defect(result)
        return result

    def launch(self,job,action,project,directory):
        self.calls.append((action,job['id'],job['phase']))
        if self.failure==(job['id'],action):raise RuntimeError('fixture interrupted worker')
        if action=='check':return self.check(job)
        if action=='evaluate':
            run=job['evaluation_run']
            source=deepcopy(self.receipts[(run,'pilot')])
            source.update(phase='main',source_phase='pilot',test_patient={'roc_auc':.78,'pr_auc':.4})
            self.receipts[(run,'main')]=source
            return source
        run=directory/'run';run.mkdir(parents=True)
        rate=job['config']['training']['backbone_lr']
        scores={'random':{5e-5:.75,1e-4:.8,3e-4:.77},'imagenet':{1e-5:.8,3e-5:.8,5e-5:.76}}
        result={'status':'completed','phase':job['phase'],'run':str(run),
            'checkpoint_sha256':('f' if job['student_initialization']=='random' else 'e')*64,
            'best_val_patient_roc_auc':scores[job['student_initialization']][rate]}
        if job['phase']=='main':result['test_patient']={'roc_auc':.79,'pr_auc':.4}
        (run/'result.json').write_text(json.dumps(result))
        self.receipts[(str(run),job['phase'])]=deepcopy(result)
        return result

    def verify(self,run,check,job):
        result=deepcopy(self.receipts[(str(run),job['config']['phase'])])
        self.assertEqual(result['phase'],job['config']['phase'])
        if result['phase']=='pilot':self.assertNotIn('test_patient',result)
        return result

    def execute(self,plan=None,**kwargs):
        with redirect_stdout(io.StringIO()):
            return study.run_study(plan or self.plan(),launcher=self.launch,verifier=self.verify,
                reporter=lambda *_:None,**kwargs)

    def test_plan_six_pilots_and_declared_grids_without_writing_files(self):
        before={str(p):p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        plan=self.plan(devices=['cuda:4','cuda:2'],max_parallel=2)
        self.assertEqual(len(plan['pilots']),6)
        self.assertEqual(plan['formal_groups'],['R-E','R-C','R-R','P-E','P-C','P-R'])
        self.assertEqual(plan['new_optimization_trainings_maximum'],10)
        self.assertTrue(all(j['phase']=='pilot' and j['method']=='equal' for j in plan['pilots']))
        self.assertEqual(before,{str(p):p.read_bytes() for p in self.root.rglob('*') if p.is_file()})
        self.assertEqual(study.protocol_hash(plan),study.protocol_hash(self.plan(devices=['cuda:0'])))

    def test_ten_trainings_two_evaluations_and_completed_resume_does_not_retrain(self):
        path=self.execute()
        state=json.loads((path/'study.json').read_text())
        self.assertEqual(state['status'],'completed')
        self.assertEqual(len([c for c in self.calls if c[0]=='train']),10)
        self.assertEqual(len([c for c in self.calls if c[0]=='evaluate']),2)
        self.assertEqual(state['selection']['random']['backbone_lr'],1e-4)
        self.assertEqual(state['selection']['imagenet']['backbone_lr'],1e-5)
        for record in state['pilots'].values():self.assertNotIn('test_patient',record['result'])
        self.calls.clear()
        self.execute(self.plan(devices=['cuda:4']),resume_study=path)
        self.assertTrue(all(c[0]=='check' for c in self.calls))
        state=json.loads((path/'study.json').read_text())
        self.assertEqual(state['status'],'completed')
        self.assertTrue(all(r['status']=='resumed' for r in state['main'].values()))

    def test_check_only_then_resume_and_failed_candidate_is_not_skipped(self):
        path=self.execute(check_only=True)
        self.assertTrue(all(c[0]=='check' for c in self.calls))
        self.assertEqual(json.loads((path/'study.json').read_text())['selection']['random']['status'],'incomplete')
        self.calls.clear();self.failure=(self.plan()['pilots'][0]['id'],'train')
        self.execute(resume_study=path)
        state=json.loads((path/'study.json').read_text())
        self.assertEqual(state['selection']['random']['status'],'incomplete')
        self.assertFalse(any(key.startswith('R-') for key in state['main']))
        self.failure=None;self.calls.clear()
        self.execute(resume_study=path)
        state=json.loads((path/'study.json').read_text())
        self.assertEqual(state['status'],'completed')
        self.assertTrue(any(a['status']=='failed' for a in state['pilots'][self.plan()['pilots'][0]['id']]['attempts']))

    def test_preflight_changed_views_block_optimization(self):
        def change(value):value['shared_view']={'changed':value['config']['training']['backbone_lr']}
        self.defect=change
        path=self.execute()
        state=json.loads((path/'study.json').read_text())
        self.assertNotEqual(state['status'],'completed')
        self.assertLess(len([c for c in self.calls if c[0]=='train']),6)

    def test_selection_ignores_test_and_rejects_bad_pilot_receipts(self):
        path=self.execute()
        state=json.loads((path/'study.json').read_text())
        selection=study.select_learning_rates(state)
        for record in state['main'].values():record['result']['test_patient']={'roc_auc':1.}
        self.assertEqual(study.select_learning_rates(state),selection)
        next(iter(state['pilots'].values()))['result']['test_patient']={'roc_auc':1.}
        with self.assertRaises(ValueError):study.select_learning_rates(state)

    def test_cli_plan_and_import_need_no_torch(self):
        result=subprocess.run([sys.executable,'-S',str(self.repo/'run_teacher_weight_experiments.py'),
            '--project',str(self.root),'--plan'],text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(len(json.loads(result.stdout)['pilots']),6)


if __name__=='__main__':unittest.main()
