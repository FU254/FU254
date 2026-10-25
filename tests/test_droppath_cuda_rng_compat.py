"""CUDA RNG layouts are owned by the backend, not a guessed byte structure.

The target-device checks here use a simulated CUDA Generator contract. Native
CPU behavior remains real; these tests do not claim execution on a CUDA GPU.
"""

from contextlib import contextmanager
from copy import deepcopy
import types
import unittest
from unittest import mock

import torch
from torch import nn

from foundation_sft import distillation as single
from foundation_sft.models import vit_tiny_student as student


class DeviceOnlyHead(nn.Module):
    """Expose a destination without allocating a CUDA tensor in CPU CI."""

    def __init__(self, device):
        super().__init__()
        self.weight = types.SimpleNamespace(device=torch.device(device))


class BackendGenerator:
    def __init__(self, seed, *, canonical=None, error=None):
        self.seed = seed
        self.canonical = canonical
        self.error = error
        self.received = None

    def set_state(self, state):
        self.received = state.clone()
        if self.error is not None:
            raise self.error
        return self

    def initial_seed(self):
        return self.seed

    def get_state(self):
        return (self.canonical if self.canonical is not None else self.received).clone()

    def manual_seed(self, seed):
        raise AssertionError("A saved stream must be restored, never silently reseeded")


class DropPathCudaRngCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.model = student.build_vit_tiny_student(random_seed=42, drop_path_rate=.1)
        self.original = self.model.stochastic_state_dict()
        self.real_generator = torch.Generator

    def assert_stream_equal(self, first, second):
        self.assertEqual(set(first), set(second))
        for key in ("format_version", "seed"):
            self.assertEqual(first[key], second[key])
        for key in ("cpu", "cuda"):
            if first[key] is None:
                self.assertIsNone(second[key])
            else:
                torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)

    def selective_generator(self, backend, devices):
        def create(*args, **kwargs):
            device = kwargs.get("device", args[0] if args else "cpu")
            if torch.device(device).type == "cuda":
                devices.append(torch.device(device))
                return backend
            return self.real_generator(*args, **kwargs)
        return create

    def advanced_cpu_state(self):
        generator = self.real_generator(device="cpu").manual_seed(self.original["seed"])
        torch.rand(12, generator=generator)
        return generator.get_state()

    def test_cpu_load_preserves_an_opaque_cuda_blob_without_touching_cuda(self):
        value = deepcopy(self.original)
        value["cuda"] = torch.arange(512, dtype=torch.int64).to(torch.uint8)
        expected = value["cuda"].clone()
        before = torch.get_rng_state().clone()
        with mock.patch.object(torch.cuda, "is_available", return_value=True), mock.patch.object(
            torch.cuda, "_lazy_init", side_effect=AssertionError("CPU load must not initialize CUDA")
        ), mock.patch.object(torch.cuda, "current_device", side_effect=AssertionError("No default GPU")), mock.patch.object(
            torch.cuda, "get_rng_state", side_effect=AssertionError("No CUDA RNG read")
        ), mock.patch.object(torch.cuda, "set_rng_state", side_effect=AssertionError("No CUDA RNG write")):
            self.model.load_stochastic_state_dict(value)
        value["cuda"].zero_()
        torch.testing.assert_close(self.model.stochastic_state_dict()["cuda"], expected, rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)

    def test_cpu_load_does_not_impose_a_64k_cuda_layout_limit(self):
        value = deepcopy(self.original)
        value["cuda"] = torch.arange(70000, dtype=torch.int64).to(torch.uint8)
        self.model.load_stochastic_state_dict(value)
        torch.testing.assert_close(self.model.stochastic_state_dict()["cuda"], value["cuda"], rtol=0, atol=0)

    def test_actual_target_generator_accepts_and_canonicalizes_non16_state(self):
        value = deepcopy(self.original)
        value.update(cpu=self.advanced_cpu_state(), cuda=torch.arange(37, dtype=torch.uint8))
        canonical = torch.arange(53, dtype=torch.uint8)
        backend = BackendGenerator(self.original["seed"], canonical=canonical)
        devices = []
        self.model.encoder.head = DeviceOnlyHead("cuda:4")
        before = torch.get_rng_state().clone()
        with mock.patch.object(torch, "Generator", side_effect=self.selective_generator(backend, devices)), mock.patch.object(
            torch.cuda, "current_device", side_effect=AssertionError("Do not select GPU0")):
            self.model.load_stochastic_state_dict(value)
        self.assertEqual(devices, [torch.device("cuda:4")])
        torch.testing.assert_close(backend.received, value["cuda"], rtol=0, atol=0)
        torch.testing.assert_close(self.model.stochastic_state_dict()["cuda"], canonical, rtol=0, atol=0)
        torch.testing.assert_close(self.model.stochastic_state_dict()["cpu"], value["cpu"], rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)

    def test_backend_rejection_keeps_original_stream_atomic(self):
        self.model.encoder.head = DeviceOnlyHead("cuda:4")
        for error in (RuntimeError("unsupported runtime RNG layout"), TypeError("bad RNG state"), ValueError("bad bytes")):
            value = deepcopy(self.original)
            value.update(cpu=self.advanced_cpu_state(), cuda=torch.arange(31, dtype=torch.uint8))
            backend = BackendGenerator(self.original["seed"], error=error)
            devices = []
            before = torch.get_rng_state().clone()
            with self.subTest(error=type(error).__name__), mock.patch.object(
                torch, "Generator", side_effect=self.selective_generator(backend, devices)
            ), self.assertRaises(ValueError):
                self.model.load_stochastic_state_dict(value)
            self.assert_stream_equal(self.original, self.model.stochastic_state_dict())
            torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)

    def test_backend_seed_mismatch_keeps_original_stream_atomic(self):
        self.model.encoder.head = DeviceOnlyHead("cuda:4")
        value = deepcopy(self.original)
        value.update(cpu=self.advanced_cpu_state(), cuda=torch.arange(27, dtype=torch.uint8))
        backend = BackendGenerator(self.original["seed"] + 1)
        before = torch.get_rng_state().clone()
        with mock.patch.object(torch, "Generator", side_effect=self.selective_generator(backend, [])), self.assertRaisesRegex(
            ValueError, "different seed"):
            self.model.load_stochastic_state_dict(value)
        self.assert_stream_equal(self.original, self.model.stochastic_state_dict())
        torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)

    def test_cpu_loaded_stream_is_validated_on_first_cuda_training_use(self):
        value = deepcopy(self.original)
        value["cuda"] = torch.arange(43, dtype=torch.uint8)
        self.model.load_stochastic_state_dict(value)
        backend = BackendGenerator(self.original["seed"])
        devices, writes = [], []
        before = torch.get_rng_state().clone()

        @contextmanager
        def cpu_only_fork(*, devices):
            self.assertEqual(devices, [4])
            outer = torch.get_rng_state().clone()
            try:
                yield
            finally:
                torch.set_rng_state(outer)

        def capture_write(state, device):
            writes.append((state.clone(), device))

        images = types.SimpleNamespace(device=torch.device("cuda:4"))
        with mock.patch.object(torch, "Generator", side_effect=self.selective_generator(backend, devices)), mock.patch.object(
            torch.random, "fork_rng", side_effect=cpu_only_fork
        ), mock.patch.object(torch.cuda, "set_rng_state", side_effect=capture_write), mock.patch.object(
            torch.cuda, "get_rng_state", return_value=value["cuda"].clone()):
            with self.model._stochastic_rng(images):
                pass
        self.assertEqual(devices, [torch.device("cuda:4")])
        torch.testing.assert_close(backend.received, value["cuda"], rtol=0, atol=0)
        self.assertEqual(len(writes), 1)
        self.assertEqual(writes[0][1], 4)
        torch.testing.assert_close(writes[0][0], value["cuda"], rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)

    def test_checkpoint_loader_moves_to_requested_device_before_stream_validation(self):
        events = []
        metadata = {"mode": "full_finetune", "initialization": {"seed": 42},
                    "regularization": {"drop_path_rate": .1}}
        stream = deepcopy(self.original)
        stream["cuda"] = torch.arange(45, dtype=torch.uint8)
        payload = {"format_version": single.FORMAT, "model_metadata": metadata,
                   "config": {"training": {"drop_path_rate": .1}}, "model": {},
                   "student_stochastic_state": stream}

        class MovingStudent:
            device = "cpu"

            def training_metadata(self):
                return metadata

            def load_trainable_state_dict(self, state):
                events.append("weights")

            def to(self, device):
                self.device = str(device)
                events.append(("to", self.device))
                return self

            def load_stochastic_state_dict(self, state):
                if self.device != "cuda:4":
                    raise AssertionError("RNG restoration must use the requested CUDA device")
                events.append("stream")
                torch.testing.assert_close(state["cuda"], stream["cuda"], rtol=0, atol=0)

            def eval(self):
                events.append("eval")
                return self

        model = MovingStudent()
        with mock.patch.object(torch, "load", return_value=payload), mock.patch.object(
            single, "build_vit_tiny_student", return_value=model):
            restored, saved = single.load_student_checkpoint("synthetic.pt", "cuda:4")
        self.assertIs(restored, model)
        self.assertIs(saved, payload)
        self.assertEqual(events, ["weights", ("to", "cuda:4"), "stream", "eval"])

    @unittest.skipUnless(torch.cuda.is_available(), "Native CUDA RNG regression requires a GPU")
    def test_native_cuda_state_roundtrip_reproduces_next_training_forward(self):
        device = torch.device("cuda", torch.cuda.current_device())
        self.model.to(device).train()
        images = torch.zeros(2, 3, 224, 224, device=device)
        with torch.no_grad():
            self.model(images)
        stream = self.model.stochastic_state_dict()
        outer_cpu = torch.get_rng_state().clone()
        outer_cuda = torch.cuda.get_rng_state(device).clone()
        with torch.no_grad():
            expected = self.model(images)
            self.model.load_stochastic_state_dict(stream)
            actual = self.model(images)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(torch.get_rng_state(), outer_cpu, rtol=0, atol=0)
        torch.testing.assert_close(torch.cuda.get_rng_state(device), outer_cuda, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
