"""Register a verified finished result without GPU work or training."""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import fcntl
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

import recover_teacher_weight_result as recovery
from foundation_sft import teacher_weight_parallel as parallel
from foundation_sft import teacher_weight_study as control
from tests import test_teacher_weight_parallel as parallel_fixture


class TeacherWeightResultRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.base = parallel_fixture.TeacherWeightParallelTests(methodName='runTest')
        self.base.setUp()
        self.addCleanup(self.base.doCleanups)
        saved_path = list(sys.path)
        self.addCleanup(lambda:sys.path.__setitem__(slice(None), saved_path))
        self.path = self.base.path
        self.root = self.base.root
        self.result = self.base.execute()
        self.state_path = self.path / 'study.json'
        self.state = json.loads(self.state_path.read_text())
        # The real incident happened after P-C had finished and only the
        # P-R coordinator handoff failed. Restore that earlier fixture result.
        pc = self.state['main']['P-C']
        pc_attempt = next(Path(item['directory']) for item in pc['attempts']
                          if item['action'] == 'train')
        pc.update(status='completed', result=json.loads((pc_attempt / 'run/result.json').read_text()))
        target = self.state['main']['P-R']
        target.update(status='failed', error='finishing helper was incorrectly considered active')
        target['attempts'].append({'action':'train', 'status':'failed',
            'directory':str(self.path / 'jobs/P-R/train_failed_handoff'),
            'error':'unguarded finishing helper'})
        self.state['status'] = 'partial'
        self.state_path.write_text(json.dumps(self.state, indent=2) + '\n')
        self.old_bytes = self.state_path.read_bytes()
        self.reporter = mock.Mock()

    def snapshot(self):
        return {str(path.relative_to(self.path)):path.read_bytes()
                for path in self.path.rglob('*') if path.is_file()}

    def backups(self):
        return sorted(self.path.glob('study.json.before_result_recovery_*.bak'))

    def recover(self, **kwargs):
        with redirect_stdout(io.StringIO()):
            return recovery.recover(self.path, project=self.root,
                verifier=self.base.verify, reporter=self.reporter, **kwargs)

    def test_verified_recovery_preserves_failed_attempts_other_groups_and_all_artifacts(self):
        before = self.snapshot()
        with ExitStack() as stack:
            for name in ('run_parallel', 'active_unguarded_workers',
                         'validate_available_device', '_default_check', '_default_train'):
                stack.enter_context(mock.patch.object(parallel, name,
                    side_effect=AssertionError('result-only recovery must not call ' + name)))
            state = self.recover(expected_checkpoint_sha256=self.result['checkpoint_sha256'])
        target = state['main']['P-R']
        self.assertEqual(target['status'], 'resumed')
        self.assertEqual(target['phase'], 'verify_completed_result')
        self.assertEqual(target['result'], self.result)
        self.assertNotIn('error', target)
        self.assertEqual(target['attempts'], self.state['main']['P-R']['attempts'])
        self.assertEqual(target['job'], self.state['main']['P-R']['job'])
        self.assertEqual(target['check'], self.state['main']['P-R']['check'])
        self.assertEqual(state['status'], 'completed')
        for group, record in self.state['main'].items():
            if group != 'P-R':
                self.assertEqual(state['main'][group], record)
        for key in self.state:
            if key not in {'status', 'main', 'updated_at', 'recovery_history'}:
                self.assertEqual(state[key], self.state[key])
        self.assertEqual(len(self.backups()), 1)
        self.assertEqual(self.backups()[0].read_bytes(), self.old_bytes)
        audit = target['recovery_history'][-1]
        self.assertFalse(audit['training_performed'])
        self.assertFalse(audit['test_evaluation_performed'])
        self.assertEqual(audit, state['recovery_history'][-1])
        self.assertEqual(audit['previous_status'], 'failed')
        self.assertEqual(audit['previous_error'], self.state['main']['P-R']['error'])
        for name, contents in before.items():
            if name != 'study.json':
                self.assertEqual((self.path / name).read_bytes(), contents, name)
        self.assertEqual(len(self.base.calls()), 1)
        self.reporter.assert_called_once_with(self.path, state)

    def test_repeating_the_same_recovery_does_not_add_backup_or_history(self):
        first = self.recover()
        before = self.snapshot()
        second = self.recover()
        self.assertEqual(first, second)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(len(self.backups()), 1)
        self.assertEqual(len(second['recovery_history']), 1)
        self.assertEqual(len(second['main']['P-R']['recovery_history']), 1)
        self.assertEqual(len(self.base.calls()), 1)

    def test_wrong_pin_corrupt_checkpoint_or_uncompleted_registry_make_no_changes(self):
        checkpoint = Path(self.result['run']) / 'best.pt'
        original_checkpoint = checkpoint.read_bytes()
        registry_path = self.path / 'jobs/P-R/parallel_registry.json'
        original_registry = registry_path.read_bytes()
        for defect in ('wrong_pin', 'invalid_pin', 'corrupt_checkpoint', 'uncompleted_registry'):
            with self.subTest(defect=defect):
                kwargs = {}
                if defect == 'wrong_pin':
                    kwargs['expected_checkpoint_sha256'] = 'f' * 64
                elif defect == 'invalid_pin':
                    kwargs['expected_checkpoint_sha256'] = 'not-a-checkpoint-hash'
                elif defect == 'corrupt_checkpoint':
                    checkpoint.write_bytes(b'checkpoint changed')
                else:
                    registry = json.loads(original_registry)
                    registry['status'] = 'running'
                    registry_path.write_text(json.dumps(registry))
                before = self.snapshot()
                with self.assertRaises((ValueError, RuntimeError)):
                    self.recover(**kwargs)
                self.assertEqual(self.snapshot(), before)
                self.assertEqual(self.backups(), [])
                checkpoint.write_bytes(original_checkpoint)
                registry_path.write_bytes(original_registry)
        self.assertEqual(len(self.base.calls()), 1)
        self.reporter.assert_not_called()

    def test_active_original_study_lock_prevents_any_state_write(self):
        before = self.snapshot()
        with control._study_lock(self.path):
            with self.assertRaises((ValueError, RuntimeError)):
                self.recover()
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.backups(), [])
        self.reporter.assert_not_called()

    def test_active_group_lease_prevents_any_state_write(self):
        before = self.snapshot()
        path = self.path / 'jobs/P-R/.parallel_execution.lock'
        with path.open('a+b') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                with self.assertRaises((ValueError, RuntimeError)):
                    self.recover()
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.backups(), [])
        self.reporter.assert_not_called()

    def test_already_registered_different_receipt_is_not_overwritten(self):
        state = self.recover()
        state['main']['P-R']['result']['checkpoint_sha256'] = 'e' * 64
        self.state_path.write_text(json.dumps(state))
        before = self.snapshot()
        with self.assertRaises((ValueError, RuntimeError)):
            self.recover()
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(len(self.backups()), 1)

    def test_another_unfinished_group_keeps_study_partial_and_is_preserved(self):
        self.state['main']['P-C']['status'] = 'training'
        self.state_path.write_text(json.dumps(self.state))
        state = self.recover()
        self.assertEqual(state['status'], 'partial')
        self.assertEqual(state['main']['P-C'], self.state['main']['P-C'])
        self.assertEqual(state['main']['P-R']['status'], 'resumed')

    def test_help_requires_no_torch_or_site_packages(self):
        entry = self.base.base.repo / 'recover_teacher_weight_result.py'
        result = subprocess.run([sys.executable, '-S', str(entry), '--help'],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--expected-checkpoint-sha256', result.stdout)


if __name__ == '__main__':
    unittest.main()
