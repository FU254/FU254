"""Equal averaging, separate feature spaces, and frozen-teacher gradients."""

import copy
import unittest

import torch

from foundation_sft.distillation_losses import SingleTeacherDistillationLoss
from foundation_sft.equal_teacher_losses import EqualTeacherDistillationLoss, TEACHER_NAMES


class EqualTeacherLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(172)
        self.dims = dict(zip(TEACHER_NAMES, (4, 5, 6)))
        self.student = {
            "logits": torch.randn(3, requires_grad=True),
            "global": torch.randn(3, 3, requires_grad=True),
            "patches": torch.randn(3, 16, 3, requires_grad=True),
        }
        self.references = {}
        for name, grid in zip(TEACHER_NAMES, ((3, 3), (4, 4), (2, 2))):
            dim = self.dims[name]
            self.references[name] = {
                "logits": torch.randn(3, requires_grad=True),
                "global_features": torch.randn(3, dim, requires_grad=True),
                "local_features": torch.randn(3, grid[0] * grid[1], dim, requires_grad=True),
                "grid_shape": grid,
            }

    def criterion(self, **kwargs):
        return EqualTeacherDistillationLoss(self.dims, student_dim=3, pool_grid=(2, 2), **kwargs)

    def evaluate(self, criterion, references=None):
        return criterion(
            self.student["logits"], self.references if references is None else references,
            student_global=self.student["global"], student_patches=self.student["patches"],
            student_grid=(4, 4),
        )

    def test_exact_average_of_independent_legacy_objectives(self):
        criterion = self.criterion()
        actual = self.evaluate(criterion)
        expected = []
        for name in TEACHER_NAMES:
            legacy = SingleTeacherDistillationLoss(student_dim=3, teacher_dim=self.dims[name],
                                                   pool_grid=(2, 2))
            legacy.load_state_dict(criterion.criteria[name].state_dict())
            teacher = self.references[name]
            expected.append(legacy(
                self.student["logits"], teacher["logits"],
                student_global=self.student["global"], teacher_global=teacher["global_features"],
                student_patches=self.student["patches"], teacher_patches=teacher["local_features"],
                student_grid=(4, 4), teacher_grid=teacher["grid_shape"],
            ))
        for key in ("total", "soft", "global", "local"):
            torch.testing.assert_close(actual[key], sum(value[key] for value in expected) / 3)
            for name, value in zip(TEACHER_NAMES, expected):
                torch.testing.assert_close(actual["per_teacher"][name][key], value[key], rtol=0, atol=0)
        torch.testing.assert_close(
            actual["total"], 0.5 * actual["soft"] + 0.1 * actual["global"] + actual["local"],
        )

    def test_student_and_each_separate_projector_learn_teachers_remain_detached(self):
        criterion = self.criterion()
        self.evaluate(criterion)["total"].backward()
        for tensor in self.student.values():
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(tensor.grad.abs().sum().item(), 0)
        projector_parameters = []
        for name in TEACHER_NAMES:
            projector = criterion.criteria[name].projector
            self.assertEqual(tuple(projector.weight.shape), (self.dims[name], 3))
            self.assertTrue(torch.isfinite(projector.weight.grad).all())
            self.assertGreater(projector.weight.grad.abs().sum().item(), 0)
            projector_parameters.append(projector.weight)
            for key in ("logits", "global_features", "local_features"):
                self.assertIsNone(self.references[name][key].grad)
        self.assertEqual(len({id(value) for value in projector_parameters}), 3)
        self.assertEqual({id(value) for value in criterion.parameters()},
                         {id(value) for value in projector_parameters})

    def test_canonical_order_round_trip_restores_all_projectors(self):
        criterion = self.criterion()
        reversed_dims = dict(reversed(list(self.dims.items())))
        restored = EqualTeacherDistillationLoss(reversed_dims, student_dim=3, pool_grid=(2, 2))
        restored.load_state_dict(copy.deepcopy(criterion.state_dict()))
        self.assertEqual(tuple(restored.criteria), TEACHER_NAMES)
        self.assertEqual(restored.teacher_names, TEACHER_NAMES)
        self.assertEqual(restored.teacher_weights, {name: 1 / 3 for name in TEACHER_NAMES})
        torch.testing.assert_close(self.evaluate(restored)["total"], self.evaluate(criterion)["total"],
                                   rtol=0, atol=0)

    def test_equal_weights_do_not_multiply_soft_loss_by_teacher_count(self):
        criterion = self.criterion(global_weight=0, local_weight=0)
        same = {name: {"logits": self.references["rad_dino"]["logits"]} for name in TEACHER_NAMES}
        result = criterion(self.student["logits"], same)
        reference = SingleTeacherDistillationLoss(global_weight=0, local_weight=0)(
            self.student["logits"], self.references["rad_dino"]["logits"],
        )
        torch.testing.assert_close(result["total"], reference["total"])
        self.assertEqual(result["global"].item(), 0)
        self.assertEqual(result["local"].item(), 0)

    def test_fp32_math_is_preserved_under_autocast(self):
        criterion = self.criterion()
        expected = self.evaluate(criterion)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            actual = self.evaluate(criterion)
        for key in ("total", "soft", "global", "local"):
            self.assertEqual(actual[key].dtype, torch.float32)
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    def test_invalid_teacher_names_dimensions_and_weights_are_rejected(self):
        for dims in ({}, {"rad_dino": 4}, {**self.dims, "eva_x": 5},
                     {"rad_dino": 4, "chexfound": 5, "CheXzero": 6}, list(self.dims.values())):
            with self.subTest(dims=dims), self.assertRaises(ValueError):
                EqualTeacherDistillationLoss(dims)
        for dim in (0, -3, True, 4.5, float("nan"), float("inf"), "4"):
            dims = {**self.dims, "rad_dino": dim}
            with self.subTest(dim=dim), self.assertRaises((TypeError, ValueError)):
                EqualTeacherDistillationLoss(dims)
        for weights in ([1/3] * 3, {"rad_dino": 1/3}, {**dict.fromkeys(TEACHER_NAMES, 1/3), "eva_x": 0},
                        dict.fromkeys(TEACHER_NAMES, 1.0), dict.fromkeys(TEACHER_NAMES, float("nan")),
                        dict.fromkeys(TEACHER_NAMES, float("inf")), dict.fromkeys(TEACHER_NAMES, True)):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                self.criterion(teacher_weights=weights)
        self.criterion(teacher_weights=dict.fromkeys(TEACHER_NAMES, 1/3))

    def test_incomplete_or_incompatible_references_are_rejected(self):
        criterion = self.criterion()
        bad = [None, {}, {"rad_dino": self.references["rad_dino"]},
               {**self.references, "chexfound": None}, {**self.references, "chexfound": {}},
               {**self.references, "chexfound": {**self.references["chexfound"],
                                                 "global_features": torch.zeros(3, 4)}},
               {**self.references, "chexfound": {**self.references["chexfound"],
                                                 "logits": torch.zeros(4)}}]
        for references in bad:
            with self.subTest(reference_type=type(references).__name__), self.assertRaises(ValueError):
                criterion(self.student["logits"], references,
                          student_global=self.student["global"], student_patches=self.student["patches"],
                          student_grid=(4, 4))


if __name__ == "__main__":
    unittest.main()
