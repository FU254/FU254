"""Real partial DeiT optimization, frozen-prefix proofs and source-free restore."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from foundation_sft import distillation as single
from foundation_sft import teacher_weight_training as worker
from foundation_sft import train as supervised
from foundation_sft.models import vit_tiny_student as native
from tests import test_equal_teacher_training as teacher_fixture


class PartialTeacherWeightTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import timm
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.temporary = tempfile.TemporaryDirectory(prefix='partial-native-weight-')
        cls.source = Path(cls.temporary.name) / 'fixture_imagenet.pth'
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(712)
            model = timm.create_model(native.ARCHITECTURE, pretrained=False, num_classes=1000)
            torch.save({'model': model.state_dict()}, cls.source)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()
        torch.set_num_threads(cls.original_threads)

    def setUp(self):
        self.fixture = teacher_fixture.EqualTeacherTrainingTests(methodName='runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.config = deepcopy(self.fixture.config)
        self.config.update(seed=42, student_initialization='imagenet', student_checkpoint=str(self.source),
            method='equal', phase='pilot', output_root=str(self.root / 'partial'),
            student={'mode': 'partial_finetune', 'train_last_blocks': 4},
            teachers=[{'model': name, 'mode': 'lora', 'seed': 42} for name in worker.TEACHER_NAMES])
        self.config['training'].update(weight_decay=.01, drop_path_rate=0., amp=False)
        for target, name, value in (
                (worker, 'LOSS_SETTINGS', deepcopy(self.config['distillation'])),
                (native, 'sha256_file', lambda _: native.CHECKPOINT_SHA256_PREFIX + '0' * 56)):
            patcher = mock.patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def prepared(self, config):
        return deepcopy(config), self.fixture.teachers, self.fixture.split, self.fixture.infos, self.fixture.anchor

    def check(self, config):
        with redirect_stdout(io.StringIO()), mock.patch.object(worker, '_prepare', side_effect=lambda c,p: self.prepared(c)):
            directory = worker.check(config, self.root)
        return json.loads((directory / 'check.json').read_text())

    def train(self, config):
        with redirect_stdout(io.StringIO()), mock.patch('foundation_sft.training_history.save_history_plots'):
            return worker._run_training(*self.prepared(config))

    def test_options_reject_ambiguous_scope_or_random_freezing_before_weights(self):
        self.assertEqual(single._student_options({}), {'mode': 'full_finetune'})
        for last in (8, 4, 2):
            with self.subTest(last=last):
                self.assertEqual(single._student_options({'student': {'mode': 'partial_finetune', 'train_last_blocks': last}}),
                                 {'mode': 'partial_finetune', 'train_last_blocks': last})
        for value in ({'mode':'partial_finetune'}, {'mode':'partial_finetune','train_last_blocks':True},
                {'mode':'partial_finetune','train_last_blocks':6},
                {'mode':'partial_finetune','train_last_blocks':2,'rank':8},
                {'mode':'full_finetune','train_last_blocks':2}):
            with self.subTest(student=value), self.assertRaises(ValueError):
                single._student_options({'student':value})
        config = deepcopy(self.config)
        config.update(student_initialization='random', student_checkpoint=None)
        with self.assertRaisesRegex(ValueError, 'ImageNet'):
            worker._validate_config(config)

    def test_all_scopes_start_identically_and_real_preflight_proves_three_kd_gradients(self):
        full = deepcopy(self.config)
        full['student'] = {'mode': 'full_finetune'}
        reference = worker._build_student(full)
        kd = worker._build_loss(reference, self.fixture.teachers, full)
        proof = worker._initial_proof(reference, kd, full)
        self.assertNotIn('frozen_parameter_sha256', proof)
        for last in (8, 4, 2):
            with self.subTest(last=last):
                config = deepcopy(self.config)
                config['student']['train_last_blocks'] = last
                student = worker._build_student(config)
                kd = worker._build_loss(student, self.fixture.teachers, config)
                actual = worker._initial_proof(student, kd, config)
                for key in proof:
                    self.assertEqual(actual[key], proof[key])
                check = self.check(config)
                self.assertEqual(check['student_mode'], 'partial_finetune')
                self.assertTrue(check['partial_frozen_state_verified'])
                self.assertTrue(check['partial_optimizer_step_verified'])
                self.assertEqual(check['feature_kd_gradients_verified'], {'soft':True,'global':True,'local':True})
                self.assertEqual(check['frozen_parameter_sha256'], actual['frozen_parameter_sha256'])
                self.assertEqual(check['student_trainable_parameter_names'], actual['student_trainable_parameter_names'])
                self.assertTrue(all(not name.startswith('encoder.patch_embed.')
                                    for name in check['student_trainable_parameter_names']))
        self.fixture.assert_teachers_unchanged()

    def test_pilot_never_tests_and_frozen_prefix_restores_without_source_file(self):
        config = deepcopy(self.config)
        config['student']['train_last_blocks'] = 2
        check = self.check(config)
        accesses = []
        original = supervised.predict
        def predict(model, loader, device, **kwargs):
            base = getattr(loader.dataset, 'base', loader.dataset)
            accesses.append(set(base.cohort['split']))
            return original(model, loader, device, **kwargs)
        with mock.patch.object(supervised, 'predict', side_effect=predict):
            result = self.train(config)
            self.assertNotIn({'test'}, accesses)
            run = Path(result['run'])
            self.assertNotIn('test_patient', result)
            verified = worker.verify_completed_run(run, check, {'config': config})
            self.assertEqual(verified, json.loads(json.dumps(result)))
            payload = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
            frozen = {name:payload['model'][name] for name in result['student_frozen_parameter_names']}
            self.assertEqual(single._state_hash(frozen), check['frozen_parameter_sha256'])
            history = json.loads((run / 'history.json').read_text())
            self.assertTrue(all(row['partial_frozen_state_verified'] for row in history))
            # Keep the original source out of reach during restore and final
            # evaluation; saved full student weights include the frozen prefix.
            absent = self.source.with_suffix('.temporarily_absent')
            self.source.rename(absent)
            try:
                restored, _ = single.load_student_checkpoint(run / 'best.pt', 'cpu')
                self.assertEqual(restored.training_mode, 'partial_finetune')
                self.assertEqual(restored.training_metadata(), payload['model_metadata'])
                promoted = deepcopy(config)
                promoted['phase'] = 'main'
                immutable = {name:(run/name).read_bytes() for name in ('config.json','result.json','best.pt')}
                with redirect_stdout(io.StringIO()):
                    evaluated = worker.evaluate_selected(run, config=promoted)
                    second = worker.evaluate_selected(run, config=promoted)
                self.assertEqual(evaluated, second)
                self.assertEqual(accesses.count({'test'}), 1)
                for name, data in immutable.items():
                    self.assertEqual((run/name).read_bytes(), data)
            finally:
                absent.rename(self.source)
            corrupted = deepcopy(payload)
            key = check['student_frozen_parameter_names'][0]
            corrupted['model'][key].view(-1)[0] += 1
            with self.assertRaisesRegex(ValueError, 'frozen values'):
                worker._partial_checkpoint_proof(corrupted)
            changed_scope = deepcopy(payload)
            changed_scope['config']['student']['train_last_blocks'] = 4
            bad_path = self.root / 'wrong_scope.pt'
            torch.save(changed_scope, bad_path)
            with self.assertRaisesRegex(ValueError, 'configuration and metadata'):
                single.load_student_checkpoint(bad_path, 'cpu')
        self.fixture.assert_teachers_unchanged()

    def test_runtime_rejects_any_frozen_weight_change(self):
        original = worker._train_epoch
        def changed(student, *args, **kwargs):
            result = original(student, *args, **kwargs)
            with torch.no_grad():
                student.encoder.patch_embed.proj.weight.view(-1)[0].add_(.001)
            return result
        with mock.patch.object(worker, '_train_epoch', side_effect=changed), \
                self.assertRaisesRegex(ValueError, 'frozen parameter values'):
            self.train(self.config)


if __name__ == '__main__':
    unittest.main()
