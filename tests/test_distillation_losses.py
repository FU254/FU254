import unittest

import torch

from foundation_sft.distillation_losses import (
    SingleTeacherDistillationLoss, binary_kl_loss,
    global_cosine_loss, local_relation_loss,
)


class DistillationLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    def test_bernoulli_reference_identity_and_gradients(self):
        student = torch.tensor([0.7, -1.2, 1.8], requires_grad=True)
        teacher = torch.tensor([-0.4, 2.5, 0.3], requires_grad=True)
        temperature = 2.0
        p = torch.sigmoid(teacher.detach() / temperature)
        q = torch.sigmoid(student / temperature)
        reference = (p * (p.log() - q.log()) + (1 - p) * ((1 - p).log() - (1 - q).log())).mean() * temperature**2
        actual = binary_kl_loss(student, teacher, temperature)
        torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-5)
        self.assertLess(abs(binary_kl_loss(student, student.detach()).item()), 1e-6)
        actual.backward()
        self.assertIsNone(teacher.grad)
        self.assertGreater(student.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(student.grad).all())

    def test_extreme_logits_and_column_inputs(self):
        student = torch.tensor([1000., -1000., 80., -80.], requires_grad=True)
        teacher = -student.detach()
        vector = binary_kl_loss(student, teacher)
        column = binary_kl_loss(student[:, None], teacher[:, None])
        torch.testing.assert_close(vector, column)
        vector.backward()
        self.assertTrue(torch.isfinite(vector))
        self.assertTrue(torch.isfinite(student.grad).all())

    def test_relation_isometry_and_spatial_permutation(self):
        student = torch.randn(2, 6, 4)
        teacher = torch.cat((student, torch.zeros(2, 6, 3)), dim=-1)
        same = local_relation_loss(student, teacher, (2, 3), (2, 3), (2, 3))
        self.assertLess(abs(same.item()), 1e-10)
        tokens = torch.tensor([[[1., 0., 0.], [1., 1., 0.], [0., 0., 1.], [0., 1., 1.]]])
        changed = local_relation_loss(tokens, tokens[:, [2, 0, 3, 1]], (2, 2), (2, 2), (2, 2))
        self.assertGreater(changed.item(), 1e-3)

    def test_all_terms_and_different_real_grids(self):
        criterion = SingleTeacherDistillationLoss(student_dim=3, teacher_dim=5, pool_grid=(2, 2))
        sl = torch.randn(2, requires_grad=True)
        sg = torch.randn(2, 3, requires_grad=True)
        sp = torch.randn(2, 16, 3, requires_grad=True)
        tl = torch.randn(2, requires_grad=True)
        tg = torch.randn(2, 5, requires_grad=True)
        tp = torch.randn(2, 30, 5, requires_grad=True)
        result = criterion(sl, tl, student_global=sg, teacher_global=tg,
                           student_patches=sp, teacher_patches=tp,
                           student_grid=(4, 4), teacher_grid=(5, 6))
        result["total"].backward()
        for value in (sl, sg, sp):
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(value.grad.abs().sum().item(), 0)
        for value in (tl, tg, tp):
            self.assertIsNone(value.grad)
        self.assertIsNotNone(criterion.projector.weight.grad)

    def test_soft_only_and_disabled_branches(self):
        soft_only = SingleTeacherDistillationLoss(global_weight=0, local_weight=0)
        value = soft_only(torch.tensor([0.2, -0.1]), torch.tensor([1., -2.]))
        self.assertIsNone(soft_only.projector)
        self.assertEqual(value["global"].item(), 0)
        self.assertEqual(value["local"].item(), 0)
        disabled = SingleTeacherDistillationLoss(soft_weight=0, global_weight=0, local_weight=0)
        logits = torch.tensor([0.2, -0.4], requires_grad=True)
        unused = torch.tensor([[float("nan")]], requires_grad=True)
        value = disabled(logits, unused, student_global=unused, student_patches=unused)
        # Stand-in scalar: the actual supervised FocalLoss remains external.
        (logits.square().mean() + value["total"]).backward()
        torch.testing.assert_close(logits.grad, logits.detach())
        self.assertIsNone(unused.grad)

    def test_identity_and_zero_vectors_are_finite(self):
        features = torch.randn(3, 8)
        self.assertLess(abs(global_cosine_loss(features, features).item()), 1e-6)
        self.assertTrue(torch.isfinite(global_cosine_loss(torch.zeros(1, 3), torch.zeros(1, 3))))
        zero = torch.zeros(1, 4, 3)
        self.assertEqual(local_relation_loss(zero, zero, (2, 2), (2, 2), (2, 2)).item(), 0)

    def test_loss_math_is_float32_under_outer_cpu_autocast(self):
        criterion = SingleTeacherDistillationLoss(student_dim=3, teacher_dim=5, pool_grid=(2, 2))
        args = (torch.randn(2), torch.randn(2))
        kwargs = dict(student_global=torch.randn(2, 3), teacher_global=torch.randn(2, 5),
                      student_patches=torch.randn(2, 9, 3), teacher_patches=torch.randn(2, 16, 5),
                      student_grid=(3, 3), teacher_grid=(4, 4))
        reference = criterion(*args, **kwargs)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            actual = criterion(*args, **kwargs)
        for key in reference:
            self.assertEqual(actual[key].dtype, torch.float32)
            torch.testing.assert_close(actual[key], reference[key], atol=0, rtol=0)

    def test_bad_parameters_and_active_inputs(self):
        for temperature in (0, -1, float("nan"), float("inf"), 1e-300):
            with self.subTest(temperature=temperature), self.assertRaises(ValueError):
                binary_kl_loss(torch.zeros(2), torch.zeros(2), temperature)
        for name in ("soft_weight", "global_weight", "local_weight"):
            for weight in (-1, float("nan"), float("inf")):
                with self.subTest(name=name, weight=weight), self.assertRaises(ValueError):
                    SingleTeacherDistillationLoss(**{name: weight})
        for student, teacher in ((torch.zeros(2, 2), torch.zeros(2)),
                                 (torch.zeros(2), torch.zeros(3)),
                                 (torch.tensor([float("nan")]), torch.zeros(1))):
            with self.assertRaises(ValueError):
                binary_kl_loss(student, teacher)
        with self.assertRaises(TypeError):
            binary_kl_loss(torch.tensor([0, 1]), torch.tensor([0, 1]))

    def test_bad_grids_and_forbidden_upsampling(self):
        student, teacher = torch.randn(1, 4, 3), torch.randn(1, 4, 5)
        for grid, pool in (((1, 3), (1, 2)), ((2, 2), (3, 3)), ((2, 2), (1, 1))):
            with self.subTest(grid=grid, pool=pool), self.assertRaises(ValueError):
                local_relation_loss(student, teacher, grid, (2, 2), pool)
        with self.assertRaises(ValueError):
            global_cosine_loss(torch.zeros(1, 3), torch.tensor([[0., float("inf"), 1.]]))


if __name__ == "__main__":
    unittest.main()
