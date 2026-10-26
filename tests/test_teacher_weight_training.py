"""Real CPU optimization, pilot isolation and source-free student evaluation."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import unittest
from unittest import mock
import torch
from foundation_sft import teacher_weight_training as worker
from foundation_sft import distillation as single
from foundation_sft import train as supervised
from tests import test_equal_teacher_training as equal_fixture
from tests import test_random_student_training as random_fixture
from tests import test_single_teacher_distillation as single_fixture


def small_factory(checkpoint=None, *, initialization=None, random_seed=None):
    if random_seed is not None:
        return random_fixture.SmallRandomStudent(random_seed=random_seed)
    result = single_fixture.SmallTokenStudent(checkpoint, initialization=initialization)
    if initialization is None:
        with torch.no_grad():
            result.encoder.patch_embed.weight.add_(.1)
    return result


class TeacherWeightTrainingTests(unittest.TestCase):
    def setUp(self):
        self.base = equal_fixture.EqualTeacherTrainingTests(methodName='runTest')
        self.base.setUp()
        self.addCleanup(self.base.doCleanups)
        self.root = self.base.root
        self.config = deepcopy(self.base.config)
        self.config.update(seed=42, student_initialization='random', student_checkpoint=None,
            method='equal', phase='pilot', output_root=str(self.root/'weighted'),
            teachers=[{'model':name, 'mode':'lora', 'seed':42} for name in worker.TEACHER_NAMES])
        self.config['training'].update(weight_decay=.01, drop_path_rate=0., amp=False)
        for target, name, replacement in ((worker, 'LOSS_SETTINGS', deepcopy(self.config['distillation'])),
                (worker, 'build_vit_tiny_student', small_factory),
                (single, 'build_vit_tiny_student', small_factory)):
            patcher = mock.patch.object(target, name, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)

    def prepared(self, config):
        return deepcopy(config), self.base.teachers, self.base.split, self.base.infos, self.base.anchor

    def run_config(self, config):
        with redirect_stdout(io.StringIO()), mock.patch('foundation_sft.training_history.save_history_plots'):
            return worker._run_training(*self.prepared(config))

    def check_config(self, config):
        with redirect_stdout(io.StringIO()), mock.patch.object(worker, '_prepare', side_effect=lambda c,p:self.prepared(c)):
            directory = worker.check(config, self.root)
        return json.loads((directory/'check.json').read_text())

    def test_initial_student_head_projectors_and_data_views_are_controlled(self):
        proofs = {}
        heads = []
        first_batch = None
        for initialization in ('random', 'imagenet'):
            for method in ('equal', 'ca', 'rl'):
                cfg = deepcopy(self.config)
                cfg.update(student_initialization=initialization, method=method,
                    student_checkpoint=None if initialization=='random' else str(self.base.fixture.student_checkpoint))
                supervised.set_seed(42)
                student = worker._build_student(cfg)
                kd = worker._build_loss(student,self.base.teachers,cfg)
                proof = worker._initial_proof(student,kd,cfg)
                if initialization in proofs:
                    self.assertEqual(proof, proofs[initialization])
                proofs[initialization]=proof
                heads.append(proof['head_initial_state_sha256'])
                supervised.set_seed(42)
                batch = next(iter(worker.equal._loader(self.base.split,'train',self.base.teachers,
                    student,cfg,torch.Generator().manual_seed(42))))
                if first_batch is None:first_batch=batch
                torch.testing.assert_close(batch['image'], first_batch['image'],rtol=0,atol=0)
                for name in worker.TEACHER_NAMES:
                    torch.testing.assert_close(batch['teacher_images'][name],first_batch['teacher_images'][name],rtol=0,atol=0)
        self.assertEqual(len(set(heads)),1)
        self.assertEqual(proofs['random']['distillation_initial_state_sha256'],proofs['imagenet']['distillation_initial_state_sha256'])
        self.assertNotEqual(proofs['random']['initial_state_sha256'],proofs['imagenet']['initial_state_sha256'])

    def test_rl_optimizer_parameters_are_disjoint_and_preflight_exercises_policy(self):
        cfg=deepcopy(self.config);cfg['method']='rl';cfg['phase']='main'
        student=worker._build_student(cfg)
        kd=worker._build_loss(student,self.base.teachers,cfg)
        optimizer=worker._optimizer(student,kd,cfg['training'])
        actor=worker._policy_optimizer(kd,cfg)
        left={id(p) for group in optimizer.param_groups for p in group['params']}
        right={id(p) for group in actor.param_groups for p in group['params']}
        self.assertFalse(left & right)
        self.assertTrue({id(p) for p in kd.projector_parameters()} <= left)
        check=self.check_config(cfg)
        self.assertTrue(check['policy_gradient_verified'])
        self.assertTrue(check['student_state_exact'])
        self.base.assert_teachers_unchanged()

    def test_pilot_never_touches_test_and_selected_evaluation_is_once_and_immutable(self):
        accesses=[]
        original=supervised.predict
        def predict(model,loader,device,**kwargs):
            dataset=getattr(loader.dataset,'base',loader.dataset)
            accesses.append(set(dataset.cohort['split']))
            return original(model,loader,device,**kwargs)
        check=self.check_config(self.config)
        with mock.patch.object(supervised,'predict',side_effect=predict):
            receipt=self.run_config(self.config)
            self.assertNotIn({'test'},accesses)
            self.assertNotIn('test_patient',receipt)
            run=Path(receipt['run'])
            immutable={name:(run/name).read_bytes() for name in ('config.json','result.json','best.pt')}
            verified=worker.verify_completed_run(run,check,{'config':self.config})
            self.assertEqual(verified['phase'],'pilot')
            promoted=deepcopy(self.config);promoted['phase']='main'
            with redirect_stdout(io.StringIO()):
                final=worker.evaluate_selected(run,config=promoted)
                second=worker.evaluate_selected(run,config=promoted)
            self.assertEqual(accesses.count({'test'}),1)
            self.assertEqual(final,second)
            self.assertEqual(final['source_phase'],'pilot')
            for name,data in immutable.items():self.assertEqual((run/name).read_bytes(),data)
            main_check=deepcopy(check);main_check['config']['phase']='main'
            self.assertEqual(worker.verify_completed_run(run,main_check,{'config':promoted})['phase'],'main')
        self.base.assert_teachers_unchanged()

    def test_ca_and_rl_real_training_checkpoint_policy_and_single_test_evaluation(self):
        for method in ('ca','rl'):
            cfg=deepcopy(self.config);cfg.update(method=method,phase='main')
            check=self.check_config(cfg)
            result=self.run_config(cfg)
            run=Path(result['run'])
            self.assertIn('test_patient',result)
            verified=worker.verify_completed_run(run,check,{'config':cfg})
            self.assertEqual(verified['method'],method)
            history=json.loads((run/'history.json').read_text())
            self.assertEqual(len(history),2)
            self.assertIn('train_eval_loss',history[-1])
            self.assertAlmostEqual(sum(history[-1]['teacher_weights'].values()),1.,places=5)
            payload=torch.load(run/'best.pt',map_location='cpu',weights_only=True)
            last=torch.load(run/'last.pt',map_location='cpu',weights_only=True)
            restored,_=single.load_student_checkpoint(run/'best.pt','cpu')
            self.assertEqual(single._state_hash(restored.state_dict()),single._state_hash(payload['model']))
            if method=='rl':
                self.assertEqual(history[0]['policy_optimizer_steps'],0)
                self.assertGreater(history[1]['policy_optimizer_steps'],0)
                self.assertIsNotNone(last['policy_optimizer'])
                self.assertIsNotNone(last['distillation_stochastic_state'])
                self.assertTrue(any(k.startswith('policy.network.') for k in payload['distillation_state']))
            self.base.assert_teachers_unchanged()

    def test_best_policy_projectors_sampler_and_student_share_selected_epoch(self):
        cfg=deepcopy(self.config);cfg.update(method='rl',phase='pilot')
        snapshots=[]
        original_epoch=worker._train_epoch
        original_metrics=worker.binary_metrics
        metric_calls=[0]
        def epoch(student,teachers,kd,*args):
            result=original_epoch(student,teachers,kd,*args)
            snapshots.append((single._cpu_state(student),single._cpu_state(kd),kd.stochastic_state_dict()))
            return result
        def metrics(*args,**kwargs):
            result=original_metrics(*args,**kwargs)
            metric_calls[0]+=1
            if metric_calls[0]%2==0:
                result['roc_auc']=.9 if metric_calls[0]==2 else .7
            return result
        with mock.patch.object(worker,'_train_epoch',side_effect=epoch),mock.patch.object(worker,'binary_metrics',side_effect=metrics):
            receipt=self.run_config(cfg)
        run=Path(receipt['run'])
        best=torch.load(run/'best.pt',map_location='cpu',weights_only=True)
        last=torch.load(run/'last.pt',map_location='cpu',weights_only=True)
        self.assertEqual(best['best_epoch'],1)
        self.assertEqual(single._state_hash(best['model']),single._state_hash(snapshots[0][0]))
        self.assertEqual(single._state_hash(best['distillation_state']),single._state_hash(snapshots[0][1]))
        torch.testing.assert_close(best['distillation_stochastic_state']['cpu'],snapshots[0][2]['cpu'],rtol=0,atol=0)
        torch.testing.assert_close(last['distillation_stochastic_state']['cpu'],snapshots[1][2]['cpu'],rtol=0,atol=0)
        self.assertFalse(torch.equal(best['distillation_stochastic_state']['cpu'],last['distillation_stochastic_state']['cpu']))

    def test_changed_selected_checkpoint_pin_is_rejected_before_test(self):
        receipt=self.run_config(self.config)
        cfg=deepcopy(self.config);cfg.update(phase='main',selected_checkpoint_sha256='0'*64)
        with mock.patch.object(supervised,'predict',side_effect=AssertionError('Must reject before reading test pixels')):
            with self.assertRaisesRegex(ValueError,'selected pilot checkpoint'):
                worker.evaluate_selected(receipt['run'],config=cfg)


if __name__ == '__main__':
    unittest.main()
