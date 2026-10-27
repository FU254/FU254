"""Real native CPU workers across partial-study screening and selected evaluation.

The tiny DICOM/teacher fixture keeps this test fast. Validation ranking alone is
controlled so the coordinator must promote last4; student optimization, frozen
tensor checks, checkpoint restore, receipts and test inference are genuine.
Source-plan/LR validation is exercised separately in test_partial_finetune_study.
"""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import unittest
from unittest import mock

import torch

from foundation_sft import partial_finetune_study as study
from foundation_sft import teacher_weight_study as original
from foundation_sft import teacher_weight_training as worker
from foundation_sft import train as supervised
from tests import test_partial_teacher_weight_training as native_fixture


class PartialFinetuneNativeIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        native_fixture.PartialTeacherWeightTrainingTests.setUpClass()

    @classmethod
    def tearDownClass(cls):
        native_fixture.PartialTeacherWeightTrainingTests.tearDownClass()

    def setUp(self):
        self.native = native_fixture.PartialTeacherWeightTrainingTests(methodName='runTest')
        self.native.setUp()
        self.addCleanup(self.native.doCleanups)
        self.root = self.native.root
        self.calls = []
        self.accesses = []
        self.initial = {}
        self.training_snapshots = {}
        self.immutable_pilots = {}
        # Original study verification uses the same tiny synthetic token grid.
        patcher = mock.patch.object(original, 'LOSS_SETTINGS', deepcopy(self.native.config['distillation']))
        patcher.start()
        self.addCleanup(patcher.stop)

    def prepared(self, config):
        return self.native.prepared(config)

    def train(self, config, score):
        metric = worker.binary_metrics
        calls = 0

        def validation_rank(*args, **kwargs):
            nonlocal calls
            result = metric(*args, **kwargs)
            calls += 1
            # Every epoch evaluates train then validation; preserve real train
            # metrics while fixing the validation ranking for selection.
            if calls % 2 == 0:
                result['roc_auc'] = score - .01 * ((calls // 2) - 1)
            return result

        name = config['student']['mode']
        count = config['student'].get('train_last_blocks')
        key = (name, count)
        epoch = worker._train_epoch

        def capture(student, teachers, kd, *args):
            if key not in self.initial:
                self.initial[key] = {n:p.detach().clone() for n,p in student.named_parameters()}
            result = epoch(student, teachers, kd, *args)
            self.training_snapshots[key] = {n:p.detach().clone() for n,p in student.named_parameters()}
            return result

        with mock.patch.object(worker, 'binary_metrics', side_effect=validation_rank), \
                mock.patch.object(worker, '_train_epoch', side_effect=capture):
            return self.native.train(config)

    def test_three_real_candidates_selected_test_once_and_resume_without_optimization(self):
        config = deepcopy(self.native.config)
        config.update(student={'mode':'full_finetune'}, phase='main',
                      output_root=str(self.root/'source_p_e'))
        config['training'].update(backbone_lr=5e-5, head_lr=1e-4, projection_lr=1e-4)
        source_check = self.native.check(config)
        source = self.train(config, .6)
        source_run = Path(source['run'])
        source_bytes = {str(p.relative_to(source_run)):p.read_bytes()
                        for p in source_run.rglob('*') if p.is_file()}
        config['selected_checkpoint_sha256'] = source['checkpoint_sha256']
        baseline = study._job(deepcopy(config), 'baseline')
        candidates = []
        for count in (8,4,2):
            candidate = deepcopy(config)
            candidate.pop('selected_checkpoint_sha256')
            candidate.update(phase='pilot', student={'mode':'partial_finetune','train_last_blocks':count})
            candidates.append(study._job(candidate, 'last'+str(count)))
        plan = {'format_version':study.FORMAT, 'project':str(self.root),
            'config_path':str(self.root/'fixture_config.json'),
            'source_study':str(self.root/'source_p_e'), 'source_protocol_sha256':'c'*64,
            'output_root':str(self.root/'partial_study'), 'devices':['cpu'],
            'max_parallel':1, 'cache_dir':None, 'seeds':[42],
            'selection_rule':deepcopy(study.SELECTION_RULE),
            'baseline':{'job':baseline,'check':source_check,'result':source,
                        'source_study':str(self.root/'source_p_e')},
            'jobs':candidates, 'new_optimization_trainings':3}

        predict = supervised.predict

        def audit_predict(model, loader, device, **kwargs):
            base = getattr(loader.dataset, 'base', loader.dataset)
            self.accesses.append(set(base.cohort['split']))
            return predict(model, loader, device, **kwargs)

        def launch(job, action, project, directory):
            self.calls.append((action,job['id']))
            effective = deepcopy(job['config'])
            effective['output_root'] = str(directory/'outputs')
            if action == 'check':
                return self.native.check(effective)
            if action == 'evaluate':
                selected = Path(job['evaluation_run'])
                self.immutable_pilots[selected] = {n:(selected/n).read_bytes()
                    for n in ('config.json','result.json','best.pt')}
                return worker.evaluate_selected(selected,config=effective)
            self.assertEqual(action,'train')
            self.assertEqual(effective['phase'],'pilot')
            return self.train(effective, .9 if job['id']=='last4' else .8)

        def verify(run, check, job):
            # This is the actual production receipt/protocol/tensor validator.
            return worker.verify_completed_run(run,check,job)

        with redirect_stdout(io.StringIO()), mock.patch.object(supervised,'predict',side_effect=audit_predict):
            path = study.run_study(plan,launcher=launch,verifier=verify,reporter=lambda *_:None)
            state = json.loads((path/'study.json').read_text())
            self.assertEqual(state['status'],'completed')
            self.assertEqual(state['selection']['arm'],'last4')
            self.assertEqual(sorted(name for action,name in self.calls if action=='train'),['last2','last4','last8'])
            self.assertEqual([name for action,name in self.calls if action=='evaluate'],['last4_selected'])
            self.assertEqual(self.accesses.count({'test'}),1)
            for arm in ('last8','last4','last2'):
                record = state['candidates'][arm]
                self.assertEqual(record['result']['phase'],'pilot')
                self.assertNotIn('test_patient',record['result'])
                check = record['check']
                for field in ('initial_state_sha256','head_initial_state_sha256','distillation_initial_state_sha256'):
                    self.assertEqual(check[field],state['baseline']['check'][field])
                self.assertTrue(check['partial_frozen_state_verified'])
                self.assertTrue(check['partial_optimizer_step_verified'])
                self.assertEqual(check['feature_kd_gradients_verified'],{'soft':True,'global':True,'local':True})
                count = int(arm.removeprefix('last'))
                key = ('partial_finetune',count)
                before, after = self.initial[key], self.training_snapshots[key]
                frozen = record['result']['student_frozen_parameter_names']
                for name in frozen:
                    torch.testing.assert_close(before[name],after[name],rtol=0,atol=0,msg=name)
                learned = record['result']['student_trainable_parameter_names']
                self.assertTrue(any(not torch.equal(before[name],after[name])
                    for name in learned if name.startswith('encoder.blocks.')))
            for selected, immutable in self.immutable_pilots.items():
                for name, data in immutable.items():
                    self.assertEqual((selected/name).read_bytes(),data)
            calls_before = len(self.calls)
            receipt = deepcopy(state['selected_evaluation']['result'])
            study.run_study(plan,resume_study=path,launcher=launch,verifier=verify,reporter=lambda *_:None)
            self.assertEqual(self.calls[calls_before:],[('check','baseline'),('check','last8'),('check','last4'),('check','last2')])
            self.assertEqual(self.accesses.count({'test'}),1)
            resumed = json.loads((path/'study.json').read_text())
            self.assertEqual(resumed['status'],'completed')
            self.assertEqual(resumed['selected_evaluation']['result'],receipt)
        self.assertEqual({str(p.relative_to(source_run)):p.read_bytes()
            for p in source_run.rglob('*') if p.is_file()},source_bytes)
        self.native.fixture.assert_teachers_unchanged()


if __name__ == '__main__':
    unittest.main()
