"""Shared train-only display jitter on real DICOMs and native last-eight ViT KD."""

from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid
import torch
from torch.nn import functional as F

from foundation_sft import data
from foundation_sft import distillation as single
from foundation_sft import equal_teacher_distillation as equal
from foundation_sft import teacher_weight_training as worker
from foundation_sft import partial_finetune_study as coordinator
from foundation_sft import teacher_weight_study as source_study
from foundation_sft import train as supervised
from foundation_sft.config import DEFAULTS
from foundation_sft.distillation_data import SharedViewDataset
from foundation_sft.equal_teacher_data import MultiTeacherViewDataset
from foundation_sft.models.common import PreprocessSpec
from foundation_sft.models import vit_tiny_student as native
from tests import test_equal_teacher_training as teacher_fixture


class SharedIntensityDataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="shared-intensity-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        path = self.root / "synthetic.dcm"
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        dicom = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
        dicom.SOPClassUID = meta.MediaStorageSOPClassUID
        dicom.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        dicom.Rows = dicom.Columns = 16
        dicom.SamplesPerPixel = 1
        dicom.PhotometricInterpretation = "MONOCHROME2"
        dicom.BitsAllocated = dicom.BitsStored = 16
        dicom.HighBit, dicom.PixelRepresentation = 15, 0
        dicom.PixelData = (np.arange(256, dtype=np.uint16).reshape(16, 16) * 13).tobytes()
        options = ({"enforce_file_format": True} if int(pydicom.__version__.split(".")[0]) >= 3
                   else {"write_like_original": False})
        pydicom.dcmwrite(path, dicom, **options)
        self.cohort = pd.DataFrame([{"sample_id": "synthetic", "patient_id": "fixture",
                                    "path": str(path), "label": 1, "split": "train"}])
        self.spec = PreprocessSpec(16, mean=(0., 0., 0.), std=(1., 1., 1.))

    def dataset(self, **kwargs):
        return data.CXRDataset(self.cohort, self.spec, cache_dir=self.root / "cache", **kwargs)

    def test_formula_follows_geometry_cache_and_clamps_with_independent_factors(self):
        geometry = {"training": True, "rotation_degrees": 5., "horizontal_flip_probability": .5}
        clean = self.dataset(**geometry)
        active = self.dataset(**geometry, brightness_jitter=.1, contrast_jitter=.1)
        with mock.patch.object(data.pydicom, "dcmread", wraps=pydicom.dcmread) as decode:
            with mock.patch.object(torch, "rand", side_effect=[torch.tensor(.8), torch.tensor(.1)]):
                reference = clean[0]["image"]
            with mock.patch.object(torch, "rand", side_effect=[torch.tensor(.8), torch.tensor(.1),
                    torch.tensor(.75), torch.tensor(.25)]) as draws:
                actual = active[0]["image"]
            self.assertEqual(decode.call_count, 1)
            self.assertEqual(draws.call_count, 4)
        expected = ((reference - reference[0].mean()) * 1.05 + reference[0].mean()) * .95
        torch.testing.assert_close(actual, expected.clamp(0, 1), rtol=0, atol=1e-7)
        self.assertFalse(torch.equal(actual, reference))
        self.assertGreaterEqual(actual.min().item(), 0.)
        self.assertLessEqual(actual.max().item(), 1.)
        with mock.patch.object(torch, "rand", side_effect=[torch.tensor(.999), torch.tensor(.999)]):
            clipped = self.dataset(training=True, brightness_jitter=.1, contrast_jitter=.1)[0]["image"]
        self.assertEqual(clipped.min().item(), 0.)
        self.assertEqual(clipped.max().item(), 1.)

    def test_disabled_path_is_exact_and_eval_never_consumes_randomness(self):
        geometry = {"training": True, "rotation_degrees": 5., "horizontal_flip_probability": .5}
        torch.manual_seed(602)
        omitted = self.dataset(**geometry)[0]
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(602)
        explicit = self.dataset(**geometry, brightness_jitter=0., contrast_jitter=0.)[0]
        torch.testing.assert_close(explicit["image"], omitted["image"], rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
        torch.manual_seed(602)
        torch.rand(())
        torch.rand(())
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
        clean = self.dataset()[0]["image"]
        for partition in ("val", "train_eval", "test"):
            with self.subTest(partition=partition), mock.patch.object(torch, "rand",
                    side_effect=AssertionError("Evaluation drew random augmentation")):
                evaluation = self.dataset(training=False, rotation_degrees=5.,
                    horizontal_flip_probability=.5, brightness_jitter=.1, contrast_jitter=.1)[0]
            torch.testing.assert_close(evaluation["image"], clean, rtol=0, atol=0)
        for key in ("brightness_jitter", "contrast_jitter"):
            with self.subTest(key=key), mock.patch.object(torch, "rand", return_value=torch.tensor(.3)) as draws:
                self.dataset(training=True, **{key: .1})[0]
            self.assertEqual(draws.call_count, 1)

    def test_all_three_teachers_and_student_derive_one_augmented_field_once(self):
        base = self.dataset(training=True, brightness_jitter=.1, contrast_jitter=.1)
        shared = SharedViewDataset(base, student_size=12, include_teacher=True)
        specs = {"rad_dino": asdict(self.spec), "chexfound": asdict(PreprocessSpec(20)),
                 "chexzero": asdict(PreprocessSpec(24, mean=(.4, .4, .4), std=(.3, .3, .3)))}
        paired = MultiTeacherViewDataset(shared, specs)
        clean_base = self.dataset()
        clean = MultiTeacherViewDataset(SharedViewDataset(clean_base, student_size=12, include_teacher=True), specs)
        self.assertEqual(paired.metadata(), clean.metadata())
        self.assertEqual(paired.teacher_view_metadata(), clean.teacher_view_metadata())
        original = data.CXRDataset.__getitem__
        calls = []
        def fetch(dataset, index):
            calls.append(index)
            return original(dataset, index)
        with mock.patch.object(data.CXRDataset, "__getitem__", new=fetch), \
                mock.patch.object(data.pydicom, "dcmread", wraps=pydicom.dcmread) as decode, \
                mock.patch.object(torch, "rand", side_effect=[torch.tensor(.75), torch.tensor(.25)]) as draws:
            sample = paired[0]
        self.assertEqual(calls, [0])
        self.assertEqual(decode.call_count, 1)
        self.assertEqual(draws.call_count, 2)
        self.assertIs(sample["teacher_images"]["rad_dino"], sample["teacher_image"])
        field = sample["teacher_image"][0:1]
        for name, image in [("student", sample["image"]), *sample["teacher_images"].items()]:
            spec = ({"image_size": 12, "mean": shared.student_mean.flatten(),
                     "std": shared.student_std.flatten()} if name == "student" else specs[name])
            display = image * torch.as_tensor(spec["std"]).view(3, 1, 1) + torch.as_tensor(spec["mean"]).view(3, 1, 1)
            expected = (field if name == "rad_dino" else F.interpolate(field[None],
                size=(spec["image_size"],) * 2, mode="bicubic", align_corners=False,
                antialias=True)[0].clamp(0, 1))
            torch.testing.assert_close(display, expected.repeat(3, 1, 1), rtol=1e-6, atol=1e-7)

    def test_amplitudes_validate_early_and_defaults_do_not_change_effective_settings(self):
        config = {"student_initialization": "imagenet", "student_checkpoint": "unused",
            "method": "equal", "phase": "pilot", "seed": 42,
            "distillation": deepcopy(worker.LOSS_SETTINGS),
            "training": {"weight_decay": .01, "drop_path_rate": 0., "amp": False},
            "teachers": [{"model": name} for name in worker.TEACHER_NAMES]}
        teacher = {"training": deepcopy(DEFAULTS["training"])}
        baseline = single._training_settings(teacher, config)
        self.assertNotIn("brightness_jitter", baseline)
        self.assertNotIn("contrast_jitter", baseline)
        self.assertIsNone(data.intensity_augmentation())
        for key in ("brightness_jitter", "contrast_jitter"):
            for value in (True, False, None, "0.1", [], -.1, 1., float("inf"), float("nan")):
                changed = deepcopy(config)
                changed["training"][key] = value
                with self.subTest(key=key, value=value):
                    with self.assertRaisesRegex(ValueError, key):
                        self.dataset(**{key: value})
                    with self.assertRaisesRegex(ValueError, key):
                        single._training_settings(teacher, changed)
                    with self.assertRaisesRegex(ValueError, key):
                        worker._validate_config(changed)
        active = deepcopy(config)
        active["training"].update(brightness_jitter=.1, contrast_jitter=.1)
        worker._validate_config(active)
        settings = single._training_settings(teacher, active)
        self.assertEqual(settings["brightness_jitter"], .1)
        proof = data.intensity_augmentation(.1, .1)
        self.assertEqual(proof["brightness_factor_range"], [.9, 1.1])
        self.assertEqual(proof["contrast_factor_range"], [.9, 1.1])
        json.dumps(proof, allow_nan=False)

    def test_single_loader_passes_optional_parameters_only_for_active_training(self):
        config = {"seed": 42, "device": "cpu", "training": {"batch_size": 1, "num_workers": 0,
            "horizontal_flip_probability": 0., "rotation_degrees": 0.}}
        original = single.CXRDataset
        for active in (False, True):
            if active:
                config["training"].update(brightness_jitter=.1, contrast_jitter=.1)
            for partition in ("train", "val", "test"):
                with self.subTest(active=active, partition=partition), \
                        mock.patch.object(single, "CXRDataset", wraps=original) as constructor:
                    single._loader(self.cohort, partition, self.spec, PreprocessSpec(12), config)
                passed = constructor.call_args.kwargs
                for key in ("brightness_jitter", "contrast_jitter"):
                    self.assertEqual(key in passed, active and partition == "train")


class SharedIntensityNativeTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import timm
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.temporary = tempfile.TemporaryDirectory(prefix="intensity-native-weight-")
        cls.source = Path(cls.temporary.name) / "fixture_imagenet.pth"
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(712)
            model = timm.create_model(native.ARCHITECTURE, pretrained=False, num_classes=1000)
            torch.save({"model": model.state_dict()}, cls.source)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()
        torch.set_num_threads(cls.original_threads)

    def setUp(self):
        self.fixture = teacher_fixture.EqualTeacherTrainingTests(methodName="runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.config = deepcopy(self.fixture.config)
        self.config.update(seed=42, student_initialization="imagenet", student_checkpoint=str(self.source),
            method="equal", phase="pilot", output_root=str(self.root / "intensity"),
            student={"mode": "partial_finetune", "train_last_blocks": 8},
            teachers=[{"model": name, "mode": "lora", "seed": 42} for name in worker.TEACHER_NAMES])
        self.config["training"].update(epochs=2, weight_decay=.01, drop_path_rate=0., amp=False,
                                       brightness_jitter=.1, contrast_jitter=.1)
        for target, name, value in (
                (worker, "LOSS_SETTINGS", deepcopy(self.config["distillation"])),
                (native, "sha256_file", lambda _: native.CHECKPOINT_SHA256_PREFIX + "0" * 56)):
            patcher = mock.patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def prepared(self, config):
        return deepcopy(config), self.fixture.teachers, self.fixture.split, self.fixture.infos, self.fixture.anchor

    def test_real_last8_preflight_and_pilot_keep_initialization_frozen_prefix_and_clean_evaluation(self):
        student = worker._build_student(self.config)
        kd = worker._build_loss(student, self.fixture.teachers, self.config)
        initial = worker._initial_proof(student, kd, self.config)
        baseline = deepcopy(self.config)
        del baseline["training"]["brightness_jitter"]
        del baseline["training"]["contrast_jitter"]
        baseline_student = worker._build_student(baseline)
        baseline_kd = worker._build_loss(baseline_student, self.fixture.teachers, baseline)
        old_proof = worker._initial_proof(baseline_student, baseline_kd, baseline)
        self.assertNotIn("intensity_augmentation", old_proof)
        self.assertEqual({key: value for key, value in initial.items() if key != "intensity_augmentation"}, old_proof)
        del student, kd, baseline_student, baseline_kd
        with redirect_stdout(io.StringIO()), mock.patch.object(worker, "_prepare",
                side_effect=lambda c, p: self.prepared(c)):
            directory = worker.check(self.config, self.root)
        check = json.loads((directory / "check.json").read_text())
        self.assertTrue(check["partial_frozen_state_verified"])
        self.assertTrue(check["partial_optimizer_step_verified"])
        self.assertEqual(check["feature_kd_gradients_verified"], {"soft": True, "global": True, "local": True})
        for key, value in initial.items():
            self.assertEqual(check[key], value)
        self.assertEqual(check["weighting_metadata"]["method"], "equal")
        accesses = []
        original = supervised.predict
        def predict(model, loader, device, **kwargs):
            base = getattr(loader.dataset, "base", loader.dataset)
            partition = set(base.cohort["split"])
            accesses.append(partition)
            self.assertFalse(base.training)
            self.assertEqual(base.brightness_jitter, 0.)
            self.assertEqual(base.contrast_jitter, 0.)
            # The actual train_eval and validation pixels match the clean
            # inference path and do not draw from the augmentation stream.
            reference = single.inference_dataset(base.cohort, {"format_version": single.FORMAT,
                "shared_view": loader.dataset.metadata()})
            with mock.patch.object(torch, "rand", side_effect=AssertionError("Evaluation augmentation")):
                actual, expected = loader.dataset[0], reference[0]
            torch.testing.assert_close(actual["image"], expected["image"], rtol=0, atol=0)
            return original(model, loader, device, **kwargs)
        with redirect_stdout(io.StringIO()), mock.patch.object(supervised, "predict", side_effect=predict), \
                mock.patch("foundation_sft.training_history.save_history_plots"):
            result = worker._run_training(*self.prepared(self.config))
        self.assertEqual(accesses, [{"train"}, {"val"}, {"train"}, {"val"}, {"val"}])
        self.assertNotIn("test_patient", result)
        run = Path(result["run"])
        self.assertFalse((run / "test").exists())
        self.assertEqual(worker.verify_completed_run(run, check, {"config": self.config}),
                         json.loads(json.dumps(result)))
        payload = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
        for key, value in initial.items():
            self.assertEqual(result[key], value)
            self.assertEqual(payload[key], value)
        self.assertEqual(payload["shared_view"], check["shared_view"])
        self.assertEqual(payload["teacher_views"], check["teacher_views"])
        frozen = {name: payload["model"][name] for name in result["student_frozen_parameter_names"]}
        self.assertEqual(single._state_hash(frozen), initial["frozen_parameter_sha256"])
        for location in ("check", "result", "payload"):
            altered = deepcopy({"check": check, "result": result, "payload": payload}[location])
            del altered["intensity_augmentation"]
            with self.subTest(location=location), self.assertRaisesRegex(ValueError, "intensity augmentation"):
                if location == "check":
                    worker.verify_completed_run(run, altered, {"config": self.config})
                elif location == "payload":
                    worker.verify_completed_run(run, check, {"config": self.config}, checkpoint_loader=lambda _: altered)
                else:
                    supervised._json(run / "result.json", altered)
                    try:
                        worker.verify_completed_run(run, check, {"config": self.config})
                    finally:
                        supervised._json(run / "result.json", result)
        mismatched = deepcopy(check)
        mismatched["config"]["training"]["brightness_jitter"] = .2
        with self.assertRaisesRegex(ValueError, "effective training"):
            worker.verify_completed_run(run, mismatched, {"config": mismatched["config"]})
        self.fixture.assert_teachers_unchanged()

    def test_real_coordinator_reuses_completed_last8_and_resumes_one_validation_only_candidate(self):
        baseline = deepcopy(self.config)
        for key in ("brightness_jitter", "contrast_jitter"):
            del baseline["training"][key]
        source = self.root / "completed_last8_source"
        source.mkdir()
        baseline["output_root"] = str(source / "runs")
        with redirect_stdout(io.StringIO()), mock.patch.object(worker, "_prepare",
                side_effect=lambda c, p: self.prepared(c)), \
                mock.patch("foundation_sft.training_history.save_history_plots"):
            checked = worker.check(baseline, self.root)
            baseline_check = json.loads((checked / "check.json").read_text())
            pilot = worker._run_training(*self.prepared(baseline))
            main = deepcopy(baseline)
            main.update(phase="main", selected_checkpoint_sha256=pilot["checkpoint_sha256"])
            evaluated = worker.evaluate_selected(pilot["run"], config=main, project=self.root)
        baseline_check["config"] = deepcopy(main)
        self.assertEqual(worker.verify_completed_run(pilot["run"], baseline_check, {"config": main}), evaluated)
        self.assertNotIn("intensity_augmentation", baseline_check)
        source_bytes = {str(path.relative_to(source)): path.read_bytes()
                        for path in source.rglob("*") if path.is_file()}
        changes = {"brightness_jitter": .1, "contrast_jitter": .1}
        candidate = deepcopy(main)
        candidate.pop("selected_checkpoint_sha256")
        candidate["phase"] = "pilot"
        candidate["training"].update(changes)
        plan = {"format_version": coordinator.FORMAT, "study_type": "intensity_augmentation",
            "project": str(self.root), "source_study": str(source),
            "source_protocol_sha256": coordinator.records._hash({"source": pilot["checkpoint_sha256"]}),
            "output_root": str(self.root / "augmentation_study"), "devices": ["cpu"],
            "max_parallel": 1, "cache_dir": None, "seeds": [42],
            "selection_rule": {"metric": "best validation patient ROC-AUC",
                "ties": "retain the existing last8 baseline", "test_metrics_used": False,
                "requires_all_candidates": True, "intensity_augmentation": changes},
            "baseline": {"job": coordinator._job(main, "baseline"), "check": baseline_check,
                "result": evaluated, "source_study": str(source)},
            "jobs": [coordinator._job(candidate, "augmented")], "training_changes": changes,
            "evaluation_policy": "validation_only", "new_optimization_trainings": 1}
        calls, predictions = [], []
        original_predict = supervised.predict
        def predict(model, loader, device, **kwargs):
            base = getattr(loader.dataset, "base", loader.dataset)
            partition = set(base.cohort["split"])
            self.assertNotEqual(partition, {"test"}, "New augmentation study accessed test pixels")
            self.assertFalse(base.training)
            self.assertEqual(base.brightness_jitter, 0.)
            self.assertEqual(base.contrast_jitter, 0.)
            predictions.append(partition)
            return original_predict(model, loader, device, **kwargs)
        def launch(job, action, project, directory):
            calls.append((action, job["id"]))
            self.assertIn(action, ("check", "train"))
            config = deepcopy(job["config"])
            directory.mkdir(parents=True)
            config["output_root"] = str(directory / "outputs")
            if action == "check":
                checked = worker.check(config, project)
                return json.loads((checked / "check.json").read_text())
            self.assertEqual(job["id"], "augmented")
            self.assertEqual(config["phase"], "pilot")
            return worker._run_training(*self.prepared(config))
        with redirect_stdout(io.StringIO()), \
                mock.patch.object(worker, "_prepare", side_effect=lambda c, p: self.prepared(c)), \
                mock.patch.object(source_study, "LOSS_SETTINGS", deepcopy(self.config["distillation"])), \
                mock.patch.object(supervised, "predict", side_effect=predict), \
                mock.patch("foundation_sft.training_history.save_history_plots"):
            directory = coordinator.run_study(plan, launcher=launch, reporter=lambda *_: None)
            saved = coordinator.records._read(directory / "study.json")
            self.assertEqual(saved["status"], "completed")
            self.assertEqual(saved["baseline"]["status"], "reused")
            self.assertEqual(saved["candidates"]["augmented"]["status"], "completed")
            self.assertEqual(saved["selected_evaluation"]["status"], "not_requested")
            self.assertFalse(saved["selection"]["test_metrics_used"])
            self.assertEqual(calls, [("check", "baseline"), ("check", "augmented"), ("train", "augmented")])
            self.assertEqual(predictions, [{"train"}, {"val"}, {"train"}, {"val"}, {"val"}])
            receipt = saved["candidates"]["augmented"]["result"]
            self.assertNotIn("test_patient", receipt)
            self.assertFalse((Path(receipt["run"]) / "test").exists())
            before_resume = len(predictions)
            calls.clear()
            resumed = coordinator.run_study(plan, resume_study=directory,
                                             launcher=launch, reporter=lambda *_: None)
            self.assertEqual(resumed, directory)
            self.assertEqual(calls, [("check", "baseline"), ("check", "augmented")])
            self.assertEqual(len(predictions), before_resume)
        final = coordinator.records._read(directory / "study.json")
        self.assertEqual(final["status"], "completed")
        self.assertEqual(final["candidates"]["augmented"]["status"], "resumed")
        self.assertEqual(final["candidates"]["augmented"]["result"]["checkpoint_sha256"], receipt["checkpoint_sha256"])
        self.assertEqual({str(path.relative_to(source)): path.read_bytes()
                          for path in source.rglob("*") if path.is_file()}, source_bytes)
        self.fixture.assert_teachers_unchanged()


if __name__ == "__main__":
    unittest.main()
