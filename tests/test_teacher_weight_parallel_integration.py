"""Actual CPU RL optimization and old-queue handoff of one selected checkpoint."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import types
from unittest import mock
import unittest
import zipfile
import recover_teacher_weight_result as recovery
import foundation_sft

from foundation_sft import teacher_weight_parallel as parallel
from foundation_sft import teacher_weight_study as control
from foundation_sft import teacher_weight_training as worker
from tests import test_teacher_weight_study as study_fixture
from tests import test_teacher_weight_training as training_fixture


class TeacherWeightParallelIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.study_fixture = study_fixture.TeacherWeightStudyTests(methodName='runTest')
        self.study_fixture.setUp()
        self.addCleanup(self.study_fixture.doCleanups)
        self.training = training_fixture.TeacherWeightTrainingTests(methodName='runTest')
        self.training.setUp()
        self.addCleanup(self.training.doCleanups)
        self.project = self.study_fixture.root
        self.study = self.study_fixture.execute()
        state_path = self.study / 'study.json'
        state = json.loads(state_path.read_text())
        config = deepcopy(self.training.config)
        config.update(student_initialization='imagenet',
            student_checkpoint=str(self.training.base.fixture.student_checkpoint))
        config['training'].update(head_lr=1e-4, projection_lr=1e-4)
        patcher = mock.patch.object(control, 'LOSS_SETTINGS', deepcopy(config['distillation']))
        patcher.start()
        self.addCleanup(patcher.stop)
        # Keep the genuine two-stage protocol, but use the existing synthetic
        # DICOMs and tiny CPU networks for this integration exercise.
        for pilot in state['plan']['pilots']:
            cfg = deepcopy(config)
            initialization = pilot['student_initialization']
            cfg.update(student_initialization=initialization, method='equal', phase='pilot',
                student_checkpoint=None if initialization == 'random' else config['student_checkpoint'])
            cfg['training']['backbone_lr'] = pilot['config']['training']['backbone_lr']
            pilot['config'] = cfg
            state['pilots'][pilot['id']]['job'] = deepcopy(pilot)
            state['pilots'][pilot['id']].pop('check', None)
        state['protocol_sha256'] = control.protocol_hash(state['plan'])
        self.assertEqual(control.select_learning_rates(state), state['selection'])
        for job in control._formal_jobs(state, state['plan']):
            record = state['main'][job['id']]
            record['job'] = job
            record.pop('check', None)
        self.target_config = deepcopy(state['main']['P-R']['job']['config'])
        self.target_config['device'] = 'cpu'
        for group in ('P-C', 'P-R'):
            cfg = deepcopy(state['main'][group]['job']['config'])
            cfg['device'] = 'cpu'
            state['main'][group]['check'] = self.training.check_config(cfg)
            state['main'][group].pop('result', None)
        state['status'] = 'running'
        state['main']['P-C']['status'] = 'training'
        state['main']['P-R']['status'] = 'checked'
        state_path.write_text(json.dumps(state))
        self.state_bytes = state_path.read_bytes()
        self.pc_log = self.study / 'jobs/P-C/current.log'
        self.pc_log.write_text('epoch 11 continues on cuda:0\n')

    def test_real_rl_result_reused_by_future_worker_without_second_optimization(self):
        plan = parallel.make_plan(self.study, 'P-R', 'cuda:3', self.project)
        calls = []

        def cpu_check(config, project):
            config = deepcopy(config)
            config['device'] = 'cpu'
            return self.training.check_config(config)

        def cpu_train(config, project):
            calls.append('optimization')
            config = deepcopy(config)
            config['device'] = 'cpu'
            return self.training.run_config(config)

        with redirect_stdout(io.StringIO()):
            result = parallel.run_parallel(plan, check_runner=cpu_check,
                train_runner=cpu_train, process_scanner=lambda *_: [])
        run = Path(result['run'])
        history = json.loads((run / 'history.json').read_text())
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]['policy_optimizer_steps'], 0)
        self.assertGreater(history[1]['policy_optimizer_steps'], 0)
        self.assertIn('test_patient', result)
        preserved = {name: (run / name).read_bytes() for name in (
            'best.pt', 'last.pt', 'history.json', 'evaluation_result.json',
            'test/patient_predictions.csv')}
        attempt = self.study / 'jobs/P-R/train_old_queue'
        attempt.mkdir()
        config = deepcopy(plan['job']['config'])
        config.update(device='cuda:0', output_root=str(attempt / 'outputs'))
        (attempt / 'config.json').write_text(json.dumps(config))
        output = attempt / 'result.json'
        with redirect_stdout(io.StringIO()), \
                mock.patch.object(parallel, '_default_check', side_effect=AssertionError('GPU check repeated')), \
                mock.patch.object(parallel, '_default_train', side_effect=AssertionError('optimization repeated')):
            adopted = parallel.worker_main(['--project', str(self.project),
                '--config', str(attempt / 'config.json'), '--action', 'train',
                '--result-json', str(output)])
        self.assertEqual(calls, ['optimization'])
        self.assertEqual(adopted, result)
        self.assertEqual(json.loads(output.read_text()), result)
        for name, value in preserved.items():
            self.assertEqual((run / name).read_bytes(), value, name)
        self.assertEqual((self.study / 'study.json').read_bytes(), self.state_bytes)
        self.assertEqual(self.pc_log.read_text(), 'epoch 11 continues on cuda:0\n')
        # Reproduce the server's failed parent handoff while the actual RL
        # model, policy and test artifacts are already complete. The repair
        # must verify those artifacts and register them without calling any
        # GPU/optimization/evaluation entry point.
        state = json.loads((self.study / 'study.json').read_text())
        state['main']['P-C']['status'] = 'completed'
        state['main']['P-R'].update(status='failed', error='lingering helper was misclassified')
        state['status'] = 'partial'
        (self.study / 'study.json').write_text(json.dumps(state))
        before_recovery = (self.study / 'study.json').read_bytes()
        untouched = {name: deepcopy(value) for name, value in state['main'].items() if name != 'P-R'}
        # Exercise the exact previously shipped module, not just the fixed
        # source tree. Users can recover without replacing any training file.
        legacy = types.ModuleType('foundation_sft._legacy_teacher_weight_parallel')
        legacy.__package__ = 'foundation_sft'
        legacy.__file__ = str(Path(__file__).resolve().parents[1] / 'foundation_sft/teacher_weight_parallel.py')
        with zipfile.ZipFile(Path(__file__).resolve().parents[1] / 'packages/FU254-teacher-weight-parallel.zip') as archive:
            source = archive.read('teacher_weight_parallel_update/foundation_sft/teacher_weight_parallel.py')
        exec(compile(source, '<previously shipped parallel module>', 'exec'), legacy.__dict__)
        with redirect_stdout(io.StringIO()), \
                mock.patch.object(foundation_sft, 'teacher_weight_parallel', legacy), \
                mock.patch.object(legacy, 'run_parallel', side_effect=AssertionError('legacy training path entered')), \
                mock.patch.object(parallel, 'run_parallel', side_effect=AssertionError('training path entered')), \
                mock.patch.object(parallel, '_default_check', side_effect=AssertionError('GPU check entered')), \
                mock.patch.object(worker, 'evaluate_selected', side_effect=AssertionError('test reevaluated')):
            recovered = recovery.recover(self.study, project=self.project,
                expected_checkpoint_sha256=result['checkpoint_sha256'], reporter=lambda *_: None)
        self.assertEqual(recovered['status'], 'completed')
        self.assertEqual(recovered['main']['P-R']['status'], 'resumed')
        self.assertEqual(recovered['main']['P-R']['result'], result)
        self.assertEqual({name: value for name, value in recovered['main'].items() if name != 'P-R'}, untouched)
        backups = list(self.study.glob('study.json.before_result_recovery_*.bak'))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), before_recovery)
        for name, value in preserved.items():
            self.assertEqual((run / name).read_bytes(), value, name)
        self.assertEqual(calls, ['optimization'])
        self.training.base.assert_teachers_unchanged()


if __name__ == '__main__':
    unittest.main()
