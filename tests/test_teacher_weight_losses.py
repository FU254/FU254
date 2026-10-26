"""Math, gradient isolation and exact policy-stream recovery."""
from copy import deepcopy
import unittest
import torch
from foundation_sft.equal_teacher_losses import EqualTeacherDistillationLoss, TEACHER_NAMES
from foundation_sft.teacher_weight_losses import WeightedTeacherDistillationLoss, confidence_weights


class TeacherWeightLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(93)
        self.logits = torch.randn(4, requires_grad=True)
        self.global_features = torch.randn(4, 6, requires_grad=True)
        self.local = torch.randn(4, 4, 6, requires_grad=True)
        self.labels = torch.tensor([0., 1., 0., 1.])
        self.dims = dict(zip(TEACHER_NAMES, (5, 7, 8)))
        self.references = {name: {"logits": torch.randn(4, requires_grad=True),
            "global_features": torch.randn(4, dim, requires_grad=True),
            "local_features": torch.randn(4, 4, dim, requires_grad=True), "grid_shape": (2, 2)}
            for name, dim in self.dims.items()}

    def call(self, module, epoch=2):
        return module(self.logits, self.references, self.labels, student_global=self.global_features,
            student_patches=self.local, student_grid=(2, 2), epoch=epoch, total_epochs=3,
            hard_losses=torch.tensor([.1, .2, .3, .4]))

    def test_equal_matches_original_scalar_loss_and_gradients(self):
        old = EqualTeacherDistillationLoss(self.dims, student_dim=6, pool_grid=(2, 2))
        new = WeightedTeacherDistillationLoss(self.dims, student_dim=6, pool_grid=(2, 2))
        new.load_state_dict(old.state_dict())
        expected = old(self.logits, self.references, student_global=self.global_features,
            student_patches=self.local, student_grid=(2, 2))
        actual = self.call(new)
        for key in ("total", "soft", "global", "local"):
            torch.testing.assert_close(actual[key], expected[key], rtol=1e-5, atol=1e-6)
        tensors = (self.logits, self.global_features, self.local)
        a = torch.autograd.grad(actual['total'], tensors, retain_graph=True)
        b = torch.autograd.grad(expected['total'], tensors)
        for first, second in zip(a, b):
            torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-6)

    def test_ca_is_ground_truth_ce_rule_and_is_sample_wise(self):
        logits = torch.tensor([[5., -5., 0.], [5., -5., 0.]])
        labels = torch.tensor([1., 0.])
        weights = confidence_weights(logits, labels)
        errors = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels[:, None].expand_as(logits), reduction='none')
        torch.testing.assert_close(weights, (1-errors.softmax(1))/2)
        self.assertGreater(weights[0, 0], weights[0, 1])
        self.assertGreater(weights[1, 1], weights[1, 0])
        self.assertTrue((weights <= .5).all())
        torch.testing.assert_close(weights.sum(1), torch.ones(2))

    def test_all_methods_warm_up_at_same_scale_without_actor_update(self):
        for method in ('equal', 'ca', 'rl'):
            module = WeightedTeacherDistillationLoss(self.dims, 6, method=method, pool_grid=(2, 2))
            result = self.call(module, epoch=1)
            torch.testing.assert_close(result['weights'], torch.full((4, 3), 1/3))
            self.assertIsNone(result['policy_loss'])
            if module.policy is not None:
                self.assertEqual(module.policy.reward_updates.item(), 0)

    def test_rl_student_and_actor_gradients_are_separate_and_teachers_frozen(self):
        module = WeightedTeacherDistillationLoss(self.dims, 6, method='rl', pool_grid=(2, 2))
        result = self.call(module)
        result['total'].backward()
        self.assertIsNotNone(self.logits.grad)
        self.assertTrue(all(p.grad is None for p in module.policy_parameters()))
        before = self.logits.grad.clone()
        result['policy_loss'].backward()
        torch.testing.assert_close(self.logits.grad, before)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.policy_parameters()))
        for reference in self.references.values():
            self.assertTrue(all(value.grad is None for value in reference.values() if isinstance(value, torch.Tensor)))
        torch.testing.assert_close(result['weights'].sum(1), torch.ones(4))

    def test_policy_sampling_and_checkpoint_do_not_change_outer_rng(self):
        torch.manual_seed(582)
        before = torch.get_rng_state().clone()
        first = WeightedTeacherDistillationLoss(self.dims, 6, method='rl', pool_grid=(2, 2))
        # Projector initialization is allowed to consume its caller's stream;
        # the worker isolates that construction separately.
        before = torch.get_rng_state().clone()
        self.call(first)
        torch.testing.assert_close(torch.get_rng_state(), before)
        state, stream = deepcopy(first.state_dict()), first.stochastic_state_dict()
        expected = self.call(first)['weights']
        second = WeightedTeacherDistillationLoss(self.dims, 6, method='rl', pool_grid=(2, 2))
        second.load_state_dict(state)
        second.load_stochastic_state_dict(stream)
        actual = self.call(second)['weights']
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        invalid = deepcopy(stream)
        invalid['seed'] += 1
        with self.assertRaises(ValueError):
            second.load_stochastic_state_dict(invalid)

    def test_single_sample_and_constant_rewards_are_finite(self):
        from foundation_sft.teacher_weight_losses import DirichletTeacherPolicy
        policy = DirichletTeacherPolicy()
        for size in (1, 4):
            weights, log_prob, entropy = policy.actions(torch.zeros(size, 16))
            loss = policy.reinforce_loss(log_prob, entropy, torch.full((size,), -.2))
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in policy.parameters()))
            policy.zero_grad(set_to_none=True)
        policy.eval()
        state = torch.zeros(2, 16)
        weights, _, _ = policy.actions(state, sample=False)
        concentrations = policy.concentrations(state)
        torch.testing.assert_close(weights, concentrations/concentrations.sum(1, keepdim=True))


if __name__ == '__main__':
    unittest.main()
