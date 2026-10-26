"""One random soft-only B run keeps the existing training and checkpoint path."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import torch

from foundation_sft import distillation as single
from foundation_sft import random_student_distillation as trainer
from foundation_sft import random_student_profiles as profiles
from foundation_sft import random_student_suite as suite
from foundation_sft import train as supervised
from tests import test_random_student_training as random_fixture
from tests import test_single_teacher_distillation as teacher_fixture


class RandomSoftPlanningTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.repo = Path(__file__).resolve().parents[1]
        self.settings = json.loads((self.repo / suite.DEFAULT_CONFIG).read_text())
        self.settings.update(distillation_profile='soft_only', groups=['B'], seeds=[42])
        self.settings['training'].update(weight_decay=.01, drop_path_rate=0.0)
        self.settings['distillation'] = deepcopy(profiles.SOFT_ONLY_LOSS_SETTINGS)
        self.path = self.root / 'soft.json'

    def plan(self, settings=None, **kwargs):
        self.path.write_text(json.dumps(self.settings if settings is None else settings))
        return suite.make_plan(self.path, self.root, **kwargs)

    def test_one_explicit_soft_job_is_stdlib_planned_without_source_weights(self):
        plan = self.plan(devices=['cuda:4'], max_parallel=1)
        self.assertEqual([(j['id'], j['condition']) for j in plan['jobs']], [('B_seed42', 'distillation')])
        config = plan['jobs'][0]['config']
        self.assertEqual(config['distillation_profile'], 'soft_only')
        self.assertEqual(config['distillation'], profiles.SOFT_ONLY_LOSS_SETTINGS)
        self.assertEqual(config['training']['weight_decay'], .01)
        self.assertEqual(config['training']['drop_path_rate'], 0)
        self.assertFalse((self.root / 'runs').exists())
        program = self.repo / 'run_random_student_experiments.py'
        result = subprocess.run([sys.executable, '-S', str(program), '--project', str(self.root),
            '--config', str(self.path), '--plan'], cwd=self.repo, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(json.loads(result.stdout)['jobs']), 1)

    def test_profile_and_one_job_contract_cannot_silently_change(self):
        for change in (
            {'distillation_profile': 'three_term'}, {'distillation_profile': None},
            {'groups': ['A', 'B']}, {'groups': ['A']}, {'seeds': [43]}, {'seeds': [42, 43]},
            {'distillation': {**profiles.SOFT_ONLY_LOSS_SETTINGS, 'soft_weight': .6}},
        ):
            settings = deepcopy(self.settings)
            settings.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.plan(settings)
        for key, value in (('weight_decay', .001), ('drop_path_rate', .1), ('amp', True)):
            settings = deepcopy(self.settings)
            settings['training'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.plan(settings)

    def test_default_plan_stays_profile_free_and_three_term_coefficients_are_strict(self):
        settings = deepcopy(self.settings)
        settings.pop('distillation_profile')
        settings['distillation'] = deepcopy(profiles.THREE_TERM_LOSS_SETTINGS)
        plan = self.plan(settings)
        self.assertTrue(all('distillation_profile' not in job['config'] for job in plan['jobs']))
        self.assertNotIn('distillation_profile', plan)
        settings['distillation'] = deepcopy(profiles.SOFT_ONLY_LOSS_SETTINGS)
        with self.assertRaises(ValueError):
            self.plan(settings)
        copied = profiles.loss_settings(self.settings)
        copied['pool_grid'][0] = 99
        self.assertEqual(profiles.SOFT_ONLY_LOSS_SETTINGS['pool_grid'], [7, 7])

    def test_native_student_initialization_metadata_and_rng_match_legacy(self):
        config = self.plan()['jobs'][0]['config']
        legacy = deepcopy(config)
        legacy.pop('distillation_profile')
        legacy['distillation'] = deepcopy(profiles.THREE_TERM_LOSS_SETTINGS)
        torch.manual_seed(628)
        before = torch.get_rng_state().clone()
        soft = trainer._build_student(config)
        source = trainer._build_student(legacy)
        self.assertEqual(soft.training_metadata(), source.training_metadata())
        self.assertEqual(single._state_hash(soft.state_dict()), single._state_hash(source.state_dict()))
        torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)


class RandomSoftTrainingTests(unittest.TestCase):
    def setUp(self):
        fixture = teacher_fixture.SingleTeacherDistillationTests(methodName='runTest')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture, self.root = fixture, fixture.root
        self.raw = deepcopy(fixture.config)
        self.raw.update(student_checkpoint=None, student_initialization='random', seed=42,
            student={'mode': 'full_finetune'}, distillation_profile='soft_only',
            output_root=str(self.root / 'soft_runs'), distillation=deepcopy(profiles.SOFT_ONLY_LOSS_SETTINGS))
        self.raw['training'].update(weight_decay=.01, drop_path_rate=0.0)
        factory = mock.patch.object(trainer, 'build_vit_tiny_student', side_effect=random_fixture.SmallRandomStudent)
        restore = mock.patch.object(single, 'build_vit_tiny_student', side_effect=random_fixture.SmallRandomStudent)
        self.factory, self.restore = factory.start(), restore.start()
        self.addCleanup(factory.stop)
        self.addCleanup(restore.stop)
        with redirect_stdout(io.StringIO()):
            self.config, self.teacher, self.split, self.info = trainer._prepare(self.raw, self.root)
        self.before_teacher = single._cpu_state(self.teacher)

    def assert_state_equal(self, expected, actual):
        self.assertEqual(set(expected), set(actual))
        for name in expected:
            torch.testing.assert_close(expected[name], actual[name], rtol=0, atol=0, msg=name)

    def assert_teacher_unchanged(self):
        trainer._assert_frozen(self.teacher)
        self.assert_state_equal(self.before_teacher, self.teacher.state_dict())

    def completed(self):
        directory = self.root / 'soft_suite'
        directory.mkdir()
        with redirect_stdout(io.StringIO()):
            result = trainer._run_condition('distillation', directory, self.teacher,
                self.split, self.config, self.info)
        run = directory / 'distillation'
        check = {'config': self.config, 'teacher': self.info, 'distillation_profile': 'soft_only',
            'model_metadata': json.loads((run / 'model_metadata.json').read_text()),
            'shared_view': json.loads((run / 'shared_view.json').read_text()),
            'initial_state_sha256': result['initial_state_sha256'], 'cohort_hash': result['cohort_hash'],
            'distillation_initial_state_sha256': result['distillation_initial_state_sha256']}
        job = {'condition': 'distillation', 'seed': 42, 'student_mode': 'full_finetune', 'config': deepcopy(self.config)}
        return run, result, check, job

    def test_soft_preflight_proves_soft_backbone_gradient_and_frozen_teacher_without_projector(self):
        self.fixture.student_checkpoint.unlink()
        with redirect_stdout(io.StringIO()), mock.patch.object(
            supervised, 'predict', side_effect=AssertionError('Preflight cannot read validation/test')
        ):
            directory = trainer.check(self.raw, self.root)
        check = json.loads((directory / 'check.json').read_text())
        self.assertTrue(check['soft_kd_gradient_verified'])
        self.assertEqual(check['distillation_trainable_parameters'], 0)
        self.assertEqual(check['distillation_profile'], 'soft_only')
        self.assertGreater(check['losses']['soft'], 0)
        self.assertEqual(check['losses']['global'], 0)
        self.assertEqual(check['losses']['local'], 0)
        self.assertAlmostEqual(check['losses']['kd'], .5 * check['losses']['soft'], places=7)
        self.assertTrue(check['student_state_exact'])
        self.assertFalse(check['training_started'])
        self.assertFalse(list(directory.glob('*.pt')))

    def test_two_epochs_use_original_optimizer_and_best_reload_with_zero_relation_losses(self):
        epoch_states, accesses = [], []
        epoch, predict, metrics = single._train_epoch, supervised.predict, trainer.binary_metrics
        metric_count = 0

        def capture_epoch(model, teacher, kd, loader, optimizer, *args):
            self.assertIsNone(kd.projector)
            self.assertEqual(list(kd.parameters()), [])
            self.assertEqual(len(optimizer.param_groups), 2)
            if not epoch_states:
                self.assertEqual([g['lr'] for g in optimizer.param_groups],
                    [self.config['training']['backbone_lr'], self.config['training']['head_lr']])
            result = epoch(model, teacher, kd, loader, optimizer, *args)
            epoch_states.append(model.trainable_state_dict())
            return result

        def controlled_metrics(*args, **kwargs):
            nonlocal metric_count
            result = metrics(*args, **kwargs)
            metric_count += 1
            if metric_count % 2 == 0:
                result['roc_auc'] = (.9, .7)[len(epoch_states) - 1]
            return result

        def audit_predict(model, loader, device, **kwargs):
            split = set(loader.dataset.base.cohort['split'])
            accesses.append(split)
            if split == {'test'}:
                self.assert_state_equal(model.state_dict(), epoch_states[0])
            return predict(model, loader, device, **kwargs)

        with mock.patch.object(single, '_train_epoch', side_effect=capture_epoch), mock.patch.object(
            trainer, 'binary_metrics', side_effect=controlled_metrics
        ), mock.patch.object(supervised, 'predict', side_effect=audit_predict):
            run, result, check, job = self.completed()
        history = json.loads((run / 'history.json').read_text())
        self.assertEqual(accesses, [{'train'}, {'val'}, {'train'}, {'val'}, {'val'}, {'test'}])
        self.assertEqual(len(history), 2)
        for row in history:
            self.assertEqual(row['train_global_loss'], 0)
            self.assertEqual(row['train_local_loss'], 0)
            self.assertGreater(row['train_soft_loss'], 0)
            self.assertAlmostEqual(row['train_kd_loss'], .5 * row['train_soft_loss'], places=7)
            self.assertAlmostEqual(row['train_loss'], row['train_hard_loss'] + row['train_kd_loss'], places=7)
            self.assertIn('train_eval_loss', row)
            self.assertIn('train_eval_patient_roc_auc', row)
        best = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
        last = torch.load(run / 'last.pt', map_location='cpu', weights_only=True)
        self.assert_state_equal(best['model'], epoch_states[0])
        self.assert_state_equal(last['model'], epoch_states[1])
        self.assertEqual(best['distillation_state'], {})
        self.assertEqual(last['best_distillation_state'], {})
        self.assertEqual(result['distillation_trainable_parameters'], 0)
        self.assertEqual(result['distillation_profile'], 'soft_only')
        self.assertEqual(trainer.verify_completed_run(run, check, job)['status'], 'completed')
        self.assert_teacher_unchanged()
        self.fixture.student_checkpoint.unlink()
        self.fixture.teacher_checkpoint.unlink()
        restored, payload = single.load_student_checkpoint(run / 'best.pt', 'cpu')
        self.assert_state_equal(restored.state_dict(), best['model'])
        self.assertEqual(payload['config']['distillation_profile'], 'soft_only')

    def test_soft_and_three_term_share_initial_student_and_first_augmented_training_view(self):
        original = single._train_epoch
        starts, batches = {}, {}

        def capture_epoch(model, teacher, kd, loader, *args):
            label = 'soft' if kd.global_weight == 0 else 'three'
            if label not in starts:
                starts[label] = model.trainable_state_dict()
                generator = getattr(loader, 'loader', loader).generator
                rng = supervised._rng_state(generator)
                batch = next(iter(loader))
                batches[label] = {'ids': batch['sample_id'], 'image': batch['image'].clone(),
                                  'teacher_image': batch['teacher_image'].clone()}
                supervised._restore_rng(rng, generator)
            return original(model, teacher, kd, loader, *args)

        three = deepcopy(self.config)
        three.pop('distillation_profile')
        three['distillation'] = deepcopy(profiles.THREE_TERM_LOSS_SETTINGS)
        # The synthetic teacher/student expose only a 2x2 native token grid.
        # Soft-only never reads pool_grid; the source branch uses this fixture's
        # legal 2x2 pool while we audit unchanged initialization and image RNG.
        three['distillation']['pool_grid'] = [2, 2]
        with redirect_stdout(io.StringIO()), mock.patch.object(single, '_train_epoch', side_effect=capture_epoch), mock.patch.object(
            trainer, 'LOSS_SETTINGS', three['distillation']
        ):
            first, second = self.root / 'three', self.root / 'soft'
            first.mkdir()
            second.mkdir()
            a = trainer._run_condition('distillation', first, self.teacher, self.split, three, self.info)
            b = trainer._run_condition('distillation', second, self.teacher, self.split, self.config, self.info)
        self.assertEqual(a['initial_state_sha256'], b['initial_state_sha256'])
        self.assert_state_equal(starts['three'], starts['soft'])
        self.assertEqual(batches['three']['ids'], batches['soft']['ids'])
        for field in ('image', 'teacher_image'):
            torch.testing.assert_close(batches['three'][field], batches['soft'][field], rtol=0, atol=0)
        self.assert_teacher_unchanged()

    def test_soft_profile_verification_rejects_markers_and_nonempty_projector_state(self):
        run, result, check, job = self.completed()
        payload = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
        for defect in ('job_profile', 'check_profile', 'payload_profile', 'payload_marker', 'projection'):
            altered_check, altered_job, altered_payload = deepcopy(check), deepcopy(job), deepcopy(payload)
            if defect == 'job_profile':
                altered_job['config'].pop('distillation_profile')
            elif defect == 'check_profile':
                altered_check['config'].pop('distillation_profile')
            elif defect == 'payload_profile':
                altered_payload['config'].pop('distillation_profile')
            elif defect == 'payload_marker':
                altered_payload.pop('distillation_profile')
            else:
                altered_payload['distillation_state'] = {'projector.weight': torch.ones(6, 6)}
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                trainer.verify_completed_run(run, altered_check, altered_job,
                    checkpoint_loader=lambda path: altered_payload)

    def test_soft_trial_refuses_supervised_and_wrong_fixed_training_before_teacher_load(self):
        for condition in ('supervised',):
            with mock.patch.object(single, 'load_teacher', side_effect=AssertionError('Reject before teacher load')):
                with self.assertRaisesRegex(ValueError, 'only the B'):
                    trainer.check(self.raw, self.root, condition=condition)
        for key, value in (('weight_decay', .001), ('drop_path_rate', .1), ('amp', True)):
            raw = deepcopy(self.raw)
            raw['training'][key] = value
            with self.subTest(key=key), mock.patch.object(single, 'load_teacher',
                side_effect=AssertionError('Reject before teacher load')), self.assertRaises(ValueError):
                trainer._prepare(raw, self.root)


if __name__ == '__main__':
    unittest.main()
