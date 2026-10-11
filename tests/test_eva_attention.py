"""CPU attention regressions against an independent PyTorch SDPA oracle."""

import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "classification"))
from models import models_eva as eva


torch.set_num_threads(2)


def sdpa_reference(attention, inputs, *, keep_mask=None, relative_bias=None):
    """Project each tensor separately and attend over tokens using SDPA."""
    if attention.subln:
        weights = (attention.q_proj.weight, attention.k_proj.weight, attention.v_proj.weight)
    else:
        weights = attention.qkv.weight.chunk(3, dim=0)
    biases = (attention.q_bias, None, attention.v_bias)
    q, k, v = (
        F.linear(inputs, weight, bias)
        .unflatten(-1, (attention.num_heads, -1))
        .transpose(1, 2)
        for weight, bias in zip(weights, biases)
    )
    mask = None
    if relative_bias is not None:
        mask = relative_bias.unsqueeze(0).expand(inputs.shape[0], -1, -1, -1)
    if keep_mask is not None:
        allowed = keep_mask[:, None, None, :]
        mask = allowed if mask is None else mask.masked_fill(~allowed, float("-inf"))
    # Torch 2.0 SDPA has no scale keyword; compensate for its default scaling.
    q = q * (attention.scale * q.shape[-1] ** 0.5)
    attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
    merged = attended.transpose(1, 2).flatten(2)
    return F.linear(merged, attention.proj.weight, attention.proj.bias)


class CpuAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1701)

    def make_attention(self, subln, *, training=False):
        attention = eva.Attention(
            dim=12,
            num_heads=3,
            qkv_bias=True,
            qk_scale=0.37,
            subln=subln,
            xattn=True,
            attn_drop=0.35,
            proj_drop=0.2,
        )
        with torch.no_grad():
            attention.q_bias.copy_(torch.linspace(-0.3, 0.25, 12))
            attention.v_bias.copy_(torch.linspace(0.1, 0.4, 12))
        return attention.train(training)

    def assert_sdpa_outputs_and_gradients(self, attention, *, keep_mask=None, relative_bias=None):
        # Seven tokens and three heads expose accidental attention over heads.
        inputs = torch.randn(2, 7, 12, requires_grad=True)
        reference_inputs = inputs.detach().clone().requires_grad_()
        actual = attention(inputs, attn_mask=keep_mask, rel_pos_bias=relative_bias)
        expected = sdpa_reference(
            attention, reference_inputs, keep_mask=keep_mask, relative_bias=relative_bias
        )
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)

        probe = torch.randn_like(expected)
        named_parameters = list(attention.named_parameters())
        parameters = [parameter for _, parameter in named_parameters]
        actual_gradients = torch.autograd.grad((actual * probe).sum(), [inputs, *parameters])
        expected_gradients = torch.autograd.grad(
            (expected * probe).sum(), [reference_inputs, *parameters]
        )
        for name, actual_gradient, expected_gradient in zip(
            ["inputs", *(name for name, _ in named_parameters)],
            actual_gradients,
            expected_gradients,
        ):
            with self.subTest(gradient=name):
                torch.testing.assert_close(
                    actual_gradient, expected_gradient, rtol=5e-5, atol=3e-6
                )

    def test_cpu_fallback_matches_sdpa_with_both_projection_layouts(self):
        for subln in (False, True):
            for xformers_available in (False, True):
                with self.subTest(subln=subln, xformers_available=xformers_available):
                    attention = self.make_attention(subln)
                    with patch.object(eva, "XFORMERS_AVAILABLE", xformers_available), patch.object(
                        eva, "xops"
                    ) as xops:
                        self.assert_sdpa_outputs_and_gradients(attention)
                        xops.memory_efficient_attention.assert_not_called()

    def test_key_mask_and_relative_bias_match_sdpa(self):
        keep_mask = torch.tensor(
            [[True, True, False, True, False, True, True],
             [True, False, True, False, True, True, False]]
        )
        relative_bias = torch.randn(3, 7, 7) * 0.2
        for subln in (False, True):
            with self.subTest(subln=subln):
                attention = self.make_attention(subln)
                with patch.object(eva, "XFORMERS_AVAILABLE", False):
                    self.assert_sdpa_outputs_and_gradients(
                        attention, keep_mask=keep_mask, relative_bias=relative_bias
                    )

    def test_training_dropout_matches_existing_native_path(self):
        for subln in (False, True):
            with self.subTest(subln=subln):
                attention = self.make_attention(subln, training=True)
                native = copy.deepcopy(attention)
                native.xattn = False
                inputs = torch.randn(2, 7, 12)
                torch.manual_seed(2903)
                expected = native(inputs)
                torch.manual_seed(2903)
                with patch.object(eva, "XFORMERS_AVAILABLE", False):
                    actual = attention(inputs)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
