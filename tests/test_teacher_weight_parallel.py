"""Live queue handoff without duplicate optimization or manifest overwrites."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
import multiprocessing
from pathlib import Path
import shutil
import subprocess
import sys
import time
import unittest
from unittest import mock

from foundation_sft import teacher_weight_parallel as parallel
from foundation_sft import teacher_weight_study as study
from tests import test_teacher_weight_study as study_fixture


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TeacherWeightParallelTests(unittest.TestCase):
    def setUp(self):
        self.base = study_fixture.TeacherWeightStudyTests(methodName='runTest')
        self.base.setUp()
        self.addCleanup(self.base.doCleanups)
        self.root = self.base.root
        self.path = self.base.execute()
        self.state_path = self.path / 'study.json'
        self.state = json.loads(self.state_path.read_text())
        self.state['status'] = 'running'
        self.state['main']['P-C']['status'] = 'training'
        self.state['main']['P-C'].pop('result', None)
        self.state['main']['P-R']['status'] = 'checked'
        self.state['main']['P-R'].pop('result', None)
        self.state_path.write_text(json.dumps(self.state))
        self.saved_state = self.state_path.read_bytes()
        self.active_log = self.path / 'jobs/P-C/active_train.log'
        self.active_log.parent.mkdir(parents=True, exist_ok=True)
        self.active_log.write_text('epoch 11 started\n')
        self.counter = self.root / 'optimization_calls.jsonl'
        self.check_counter = self.root / 'check_calls.jsonl'

    def save_state(self):
        self.state_path.write_text(json.dumps(self.state))

    def plan(self, group='P-R', device='cuda:3'):
        return parallel.make_plan(self.path, group, device=device, project=self.root)

    def check(self, config, project):
        with self.check_counter.open('a') as stream:
            stream.write(json.dumps({'device':config['device']}) + '\n')
        check = deepcopy(self.state['main']['P-R']['check'])
        check['config'] = deepcopy(config)
        check['config']['evaluation'] = {'bootstrap_samples':10}
        return check

    def train(self, config, project):
        with self.counter.open('a') as stream:
            stream.write(json.dumps({'device':config['device'], 'method':config['method']}) + '\n')
        time.sleep(.15)
        run = Path(config['output_root']) / 'fixture_run'
        run.mkdir(parents=True)
        (run / 'best.pt').write_bytes(b'exact selected student checkpoint')
        (run / 'config.json').write_text(json.dumps(config))
        receipt = {
            'status':'completed', 'phase':'main', 'method':config['method'],
            'student_initialization':config['student_initialization'],
            'seed':config['seed'], 'run':str(run),
            'checkpoint_sha256':_hash(run / 'best.pt'),
            'best_val_patient_roc_auc':.81,
            'test_patient':{'roc_auc':.78, 'pr_auc':.4},
            'cohort_hash':self.state['main']['P-R']['check']['cohort_hash'],
        }
        (run / 'fixture_receipt.json').write_text(json.dumps(receipt))
        return receipt

    def verify(self, run, check, job):
        run = Path(run)
        receipt = json.loads((run / 'fixture_receipt.json').read_text())
        if _hash(run / 'best.pt') != receipt['checkpoint_sha256']:
            raise ValueError('fixture selected checkpoint fingerprint differs')
        saved = json.loads((run / 'config.json').read_text())
        if study._scientific_config(saved) != study._scientific_config(job['config']):
            raise ValueError('fixture scientific config differs')
        if receipt['cohort_hash'] != check['cohort_hash']:
            raise ValueError('fixture frozen cohort differs')
        return receipt

    def execute(self, plan=None, **kwargs):
        with redirect_stdout(io.StringIO()):
            return parallel.run_parallel(plan or self.plan(),
                check_runner=self.check, train_runner=self.train,
                verifier=self.verify, process_scanner=lambda *_:[], **kwargs)

    def calls(self):
        return [json.loads(row) for row in self.counter.read_text().splitlines()] if self.counter.exists() else []

    def test_plan_is_read_only_and_keeps_frozen_recipe_on_new_gpu(self):
        before = {str(p):p.read_bytes() for p in self.path.rglob('*') if p.is_file()}
        plan = self.plan()
        original = self.state['main']['P-R']['job']['config']
        self.assertEqual(plan['job']['config']['device'], 'cuda:3')
        self.assertEqual(study._scientific_config(plan['job']['config']), study._scientific_config(original))
        self.assertEqual(plan['group'], 'P-R')
        self.assertEqual(before, {str(p):p.read_bytes() for p in self.path.rglob('*') if p.is_file()})

    def test_same_result_adopted_by_old_coordinator_without_retraining_or_state_writes(self):
        result = self.execute()
        result_file = self.path / 'jobs/P-R/old_queue_attempt/result.json'
        result_file.parent.mkdir(parents=True)
        second = self.execute(origin='coordinator', result_file=result_file)
        self.assertEqual(result, second)
        self.assertEqual(json.loads(result_file.read_text()), result)
        self.assertEqual(len(self.calls()), 1)
        self.assertEqual(self.state_path.read_bytes(), self.saved_state)
        self.assertEqual(self.active_log.read_text(), 'epoch 11 started\n')

    def test_updated_worker_entry_adopts_when_old_coordinator_marks_group_training(self):
        result = self.execute()
        self.state['main']['P-R']['status'] = 'training'
        self.save_state()
        frozen_state = self.state_path.read_bytes()
        attempt = self.path / 'jobs/P-R/train_from_old_coordinator'
        attempt.mkdir(parents=True)
        config = deepcopy(self.state['main']['P-R']['job']['config'])
        config['device'] = 'cuda:0'
        config['output_root'] = str(attempt / 'outputs')
        (attempt / 'config.json').write_text(json.dumps(config))
        receipt_path = attempt / 'result.json'
        with redirect_stdout(io.StringIO()), \
                mock.patch.object(parallel, '_default_verify', self.verify), \
                mock.patch.object(parallel, '_default_check', side_effect=AssertionError('adoption must not load GPU models')), \
                mock.patch.object(parallel, '_default_train', side_effect=AssertionError('adoption must not optimize twice')):
            adopted = parallel.worker_main(['--project',str(self.root),
                '--config',str(attempt / 'config.json'), '--action','train',
                '--result-json',str(receipt_path)])
        self.assertEqual(adopted, result)
        self.assertEqual(json.loads(receipt_path.read_text()), result)
        self.assertEqual(len(self.calls()), 1)
        self.assertEqual(self.state_path.read_bytes(), frozen_state)
        self.assertEqual(self.active_log.read_text(), 'epoch 11 started\n')

    def test_preinstall_worker_receipt_is_adopted_before_parent_records_completion(self):
        plan = self.plan()
        attempt = self.path / 'jobs/P-R/train_preinstall_race'
        attempt.mkdir(parents=True)
        config = deepcopy(self.state['main']['P-R']['job']['config'])
        config['output_root'] = str(attempt / 'outputs')
        receipt = self.train(config, self.root)
        (attempt / 'result.json').write_text(json.dumps(receipt))
        self.state['main']['P-R']['status'] = 'training'
        self.state['main']['P-R']['attempts'].append({
            'action':'train', 'directory':str(attempt), 'status':'running'})
        self.save_state()
        frozen_state = self.state_path.read_bytes()
        adopted = self.execute(plan)
        self.assertEqual(adopted, receipt)
        self.assertEqual(len(self.calls()), 1)
        self.assertFalse(self.check_counter.exists())
        registry = json.loads((self.path / 'jobs/P-R/parallel_registry.json').read_text())
        self.assertFalse(registry['optimization_performed'])
        self.assertEqual(registry['origin'], 'unrecorded_legacy_receipt')
        self.assertEqual(self.state_path.read_bytes(), frozen_state)
        self.assertEqual(self.active_log.read_text(), 'epoch 11 started\n')

    def test_invalid_preinstall_worker_checkpoint_cannot_trigger_retraining(self):
        plan = self.plan()
        attempt = self.path / 'jobs/P-R/train_preinstall_invalid'
        attempt.mkdir(parents=True)
        config = deepcopy(self.state['main']['P-R']['job']['config'])
        config['output_root'] = str(attempt / 'outputs')
        receipt = self.train(config, self.root)
        (attempt / 'result.json').write_text(json.dumps(receipt))
        (Path(receipt['run']) / 'best.pt').write_bytes(b'checkpoint changed after receipt')
        self.state['main']['P-R']['attempts'].append({
            'action':'train', 'directory':str(attempt), 'status':'running'})
        self.save_state()
        with self.assertRaises((ValueError, RuntimeError)):
            self.execute(plan)
        self.assertEqual(len(self.calls()), 1)
        self.assertFalse(self.check_counter.exists())
        self.assertFalse((self.path / 'jobs/P-R/parallel_registry.json').exists())

    def test_requested_gpu_is_free_present_and_parseable_before_training(self):
        cases = (
            ('3, 48683, 0\n4, 48683, 0\n', 'cuda:3', True),
            ('3, 20000, 5\n', 'cuda:3', True),
            ('3, 19999, 0\n', 'cuda:3', False),
            ('3, 48683, 6\n', 'cuda:3', False),
            ('4, 48683, 0\n', 'cuda:3', False),
            ('3, N/A, N/A\n', 'cuda:3', False),
            ('3, 48683\n', 'cuda:3', False),
        )
        for output, device, valid in cases:
            with self.subTest(output=output), mock.patch.object(
                    parallel.subprocess, 'check_output', return_value=output) as query:
                if valid:
                    result = parallel.validate_available_device(device)
                    self.assertEqual(result['device'], device)
                    self.assertGreaterEqual(result['free_mb'], 20000)
                    self.assertLessEqual(result['utilization_percent'], 5)
                else:
                    with self.assertRaises(ValueError):
                        parallel.validate_available_device(device)
                self.assertIn('--format=csv,noheader,nounits', query.call_args.args[0])
        with mock.patch.object(parallel.subprocess, 'check_output') as query:
            with self.assertRaises(ValueError):
                parallel.validate_available_device('cpu')
            query.assert_not_called()
        with mock.patch.object(parallel.subprocess, 'check_output', side_effect=FileNotFoundError('no nvidia-smi')):
            with self.assertRaises(ValueError):
                parallel.validate_available_device('cuda:3')

    def test_cli_busy_gpu_never_launches_and_plan_status_never_query_gpu(self):
        argv = ['--project',str(self.root), '--study',str(self.path),
                '--group','P-R', '--device','cuda:3']
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), \
                mock.patch.object(parallel.subprocess, 'check_output', return_value='3, 1000, 99\n'), \
                mock.patch.object(parallel, 'run_parallel', side_effect=AssertionError('busy GPU must not launch')) as runner:
            self.assertEqual(parallel.main(argv), 1)
            runner.assert_not_called()
        for mode in ('--plan', '--status'):
            with self.subTest(mode=mode), redirect_stdout(io.StringIO()) as stream, \
                    mock.patch.object(parallel.subprocess, 'check_output', side_effect=AssertionError('read-only modes must not query GPUs')) as query, \
                    mock.patch.object(parallel, 'active_unguarded_workers', return_value=[]):
                self.assertEqual(parallel.main(argv + [mode]), 0)
                self.assertEqual(json.loads(stream.getvalue())['group'], 'P-R')
                query.assert_not_called()
        self.assertEqual(self.state_path.read_bytes(), self.saved_state)
        self.assertEqual(self.calls(), [])

    def test_two_real_processes_share_group_lock_and_only_train_once_in_either_order(self):
        if 'fork' not in multiprocessing.get_all_start_methods():
            self.skipTest('flock subprocess test needs POSIX fork')
        context = multiprocessing.get_context('fork')
        for first_origin in ('parallel', 'coordinator'):
            with self.subTest(first=first_origin):
                case = self.path / 'jobs/P-R'
                registry = case / 'parallel_registry.json'
                if registry.exists():
                    registry.unlink()
                for attempt in case.glob('parallel_*'):
                    if attempt.is_dir():
                        shutil.rmtree(attempt)
                self.counter.unlink(missing_ok=True)
                started = context.Event()
                errors = context.Queue()
                original_train = self.train

                def train(config, project):
                    started.set()
                    time.sleep(.25)
                    return original_train(config, project)

                def process(origin):
                    try:
                        with redirect_stdout(io.StringIO()):
                            parallel.run_parallel(self.plan(), origin=origin,
                                check_runner=self.check, train_runner=train,
                                verifier=self.verify, process_scanner=lambda *_:[])
                        errors.put(None)
                    except BaseException as error:
                        errors.put(repr(error))

                first = context.Process(target=process, args=(first_origin,))
                second = context.Process(target=process, args=('coordinator' if first_origin == 'parallel' else 'parallel',))
                first.start()
                self.addCleanup(lambda p=first:p.terminate() if p.is_alive() else None)
                self.assertTrue(started.wait(10), 'first process did not reach optimization')
                second.start()
                self.addCleanup(lambda p=second:p.terminate() if p.is_alive() else None)
                first.join(15)
                second.join(15)
                self.assertFalse(first.is_alive())
                self.assertFalse(second.is_alive())
                self.assertEqual(first.exitcode, 0)
                self.assertEqual(second.exitcode, 0)
                self.assertIsNone(errors.get(timeout=2))
                self.assertIsNone(errors.get(timeout=2))
                self.assertEqual(len(self.calls()), 1)
                self.assertEqual(self.state_path.read_bytes(), self.saved_state)

    def test_registry_and_checkpoint_corruption_fail_closed_instead_of_retraining(self):
        result = self.execute()
        (Path(result['run']) / 'best.pt').write_bytes(b'changed checkpoint')
        with self.assertRaises((ValueError, RuntimeError)):
            self.execute(origin='coordinator')
        self.assertEqual(len(self.calls()), 1)
        registry = self.path / 'jobs/P-R/parallel_registry.json'
        registry.write_text('{bad-json')
        with self.assertRaises((ValueError, RuntimeError)):
            self.execute()
        self.assertEqual(len(self.calls()), 1)

    def test_completed_receipt_precedes_lingering_helper_process_scan(self):
        result = self.execute()
        checks_before = self.check_counter.read_bytes()
        receipt_file = self.path / 'jobs/P-R/old_queue_after_helper/result.json'
        scanner = mock.Mock(side_effect=AssertionError(
            'verified completion must be adopted before scanning a finishing helper'))
        with redirect_stdout(io.StringIO()):
            adopted = parallel.run_parallel(self.plan(), origin='coordinator',
                check_runner=self.check, train_runner=self.train,
                verifier=self.verify, process_scanner=scanner,
                result_file=receipt_file)
        self.assertEqual(adopted, result)
        self.assertEqual(json.loads(receipt_file.read_text()), result)
        scanner.assert_not_called()
        self.assertEqual(len(self.calls()), 1)
        self.assertEqual(self.check_counter.read_bytes(), checks_before)
        self.assertEqual(self.state_path.read_bytes(), self.saved_state)
        self.assertEqual(self.active_log.read_text(), 'epoch 11 started\n')

    def test_corrupt_completed_receipt_rejected_before_lingering_helper_scan(self):
        result = self.execute()
        (Path(result['run']) / 'best.pt').write_bytes(b'changed after completion')
        scanner = mock.Mock(side_effect=AssertionError(
            'corrupt completion must fail verification before any process scan'))
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'fingerprint'):
            parallel.run_parallel(self.plan(), origin='coordinator',
                check_runner=self.check, train_runner=self.train,
                verifier=self.verify, process_scanner=scanner)
        scanner.assert_not_called()
        self.assertEqual(len(self.calls()), 1)
        self.assertEqual(self.state_path.read_bytes(), self.saved_state)

    def test_real_finishing_helper_pid_does_not_block_verified_completion(self):
        result = self.execute()
        # Model a helper after it released the lease and removed its marker,
        # while its process is still alive finishing console output.
        finishing = subprocess.Popen([sys.executable, '-c',
            'import time; time.sleep(30)',
            str(self.base.repo / 'run_teacher_weight_parallel.py'),
            '--study',str(self.path), '--group','P-R'],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)
        try:
            active = parallel.active_unguarded_workers(self.path, 'P-R')
            self.assertIn(finishing.pid, [item['pid'] for item in active])
            with redirect_stdout(io.StringIO()):
                adopted = parallel.run_parallel(self.plan(), origin='coordinator',
                    check_runner=self.check, train_runner=self.train,
                    verifier=self.verify,
                    process_scanner=parallel.active_unguarded_workers)
            self.assertEqual(adopted, result)
            self.assertIsNone(finishing.poll())
            self.assertEqual(len(self.calls()), 1)
            self.assertEqual(self.state_path.read_bytes(), self.saved_state)
        finally:
            if finishing.poll() is None:
                finishing.terminate()
            finishing.wait(timeout=5)

    def test_preexisting_unguarded_target_worker_blocks_new_training(self):
        scanner = mock.Mock(return_value=[{'pid':1234, 'config':'existing target worker'}])
        with self.assertRaises((ValueError, RuntimeError)):
            parallel.run_parallel(self.plan(), check_runner=self.check,
                train_runner=self.train, verifier=self.verify,
                process_scanner=scanner)
        scanner.assert_called_once_with(str(self.path), 'P-R')
        self.assertEqual(self.calls(), [])
        self.assertEqual(self.state_path.read_bytes(), self.saved_state)

    def test_incomplete_pilot_or_wrong_frozen_recipe_cannot_launch(self):
        for defect in ('incomplete_pilot', 'wrong_lr', 'wrong_seed'):
            with self.subTest(defect=defect):
                baseline = deepcopy(self.state)
                if defect == 'incomplete_pilot':
                    next(iter(self.state['pilots'].values()))['status'] = 'training'
                elif defect == 'wrong_lr':
                    self.state['main']['P-R']['job']['config']['training']['backbone_lr'] = .25
                else:
                    self.state['main']['P-R']['job']['config']['seed'] = 99
                self.save_state()
                with self.assertRaises((ValueError, RuntimeError)):
                    self.plan()
                self.state = baseline
                self.save_state()
        self.assertEqual(self.calls(), [])

    def test_fresh_preflight_changed_cohort_blocks_optimization(self):
        def changed(config, project):
            value = self.check(config, project)
            value['cohort_hash'] = 'different frozen patients'
            return value
        with self.assertRaises((ValueError, RuntimeError)):
            parallel.run_parallel(self.plan(), check_runner=changed,
                train_runner=self.train, verifier=self.verify,
                process_scanner=lambda *_:[])
        self.assertEqual(self.calls(), [])
        self.assertEqual(self.state_path.read_bytes(), self.saved_state)

    def test_test_scores_do_not_select_lr_and_equal_or_running_groups_are_not_retrained(self):
        before = self.plan()['job']['config']
        for record in self.state['main'].values():
            if record.get('result'):
                record['result']['test_patient'] = {'roc_auc':1.0, 'pr_auc':1.0}
        self.save_state()
        self.assertEqual(self.plan()['job']['config'], before)
        for group in ('P-E', 'P-C'):
            with self.subTest(group=group), self.assertRaises((ValueError, RuntimeError)):
                self.plan(group)
        self.assertEqual(self.calls(), [])

    def test_plan_cli_runs_without_torch_or_site_packages(self):
        entry = self.base.repo / 'run_teacher_weight_parallel.py'
        result = subprocess.run([sys.executable, '-S', str(entry),
            '--project',str(self.root), '--study',str(self.path),
            '--group','P-R', '--device','cuda:3', '--plan'],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = json.loads(result.stdout)
        self.assertEqual(plan['group'], 'P-R')
        self.assertEqual(self.state_path.read_bytes(), self.saved_state)


if __name__ == '__main__':
    unittest.main()
