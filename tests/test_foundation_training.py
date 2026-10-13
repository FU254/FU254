"""End-to-end shared trainer tests on synthetic DICOMs and a tiny real encoder.

The locally generated DINOv2 weights validate software, not medical-model
quality. Tests never use clinical data, public pretrained weights or a network.
"""

import copy
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from foundation_sft.config import DEFAULTS, _merge
from foundation_sft.metrics import aggregate_patient_predictions, select_threshold
from foundation_sft.models import build_adapter
from foundation_sft import train as trainer


@unittest.skipUnless(importlib.util.find_spec("transformers"), "Tiny HF integration needs pinned Transformers")
class FoundationTrainingTests(unittest.TestCase):
    def setUp(self):
        from transformers import BitImageProcessor, Dinov2Config, Dinov2Model

        self.temporary = tempfile.TemporaryDirectory(prefix="foundation-train-test-")
        self.root = Path(self.temporary.name)
        dicom_dir = self.root / "dicoms"
        dicom_dir.mkdir()
        rows = []
        rng = np.random.default_rng(920)
        for label in (0, 1):
            for patient_index in range(8):
                patient = f"synthetic-{label}-{patient_index}"
                # Unequal numbers of images exercise real patient aggregation.
                for image_index in range(1 + int(patient_index % 3 == 0)):
                    accession = f"a-{label}-{patient_index}-{image_index}"
                    rows.append({"放射编号": accession, "PatientID": patient, "阻塞性通气障碍": label})
                    path = dicom_dir / f"{accession}.dcm"
                    meta = FileMetaDataset()
                    meta.TransferSyntaxUID = ExplicitVRLittleEndian
                    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
                    meta.MediaStorageSOPInstanceUID = generate_uid()
                    dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
                    dataset.SOPClassUID = meta.MediaStorageSOPClassUID
                    dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
                    dataset.AccessionNumber = accession
                    dataset.PatientID = patient
                    dataset.PatientName = "Synthetic^Only"
                    dataset.Rows = dataset.Columns = 16
                    dataset.SamplesPerPixel = 1
                    dataset.PhotometricInterpretation = "MONOCHROME2"
                    dataset.BitsAllocated = dataset.BitsStored = 16
                    dataset.HighBit = 15
                    dataset.PixelRepresentation = 0
                    grid = np.arange(16, dtype=float)
                    pattern = grid[:, None] if label else grid[None, :]
                    pixels = np.clip(pattern * 40 + rng.normal(100, 15, size=(16, 16)), 0, 65535).astype(np.uint16)
                    dataset.PixelData = pixels.tobytes()
                    options = ({"enforce_file_format": True} if int(pydicom.__version__.split(".")[0]) >= 3
                               else {"write_like_original": False})
                    pydicom.dcmwrite(path, dataset, **options)
        excel = self.root / "labels.xlsx"
        pd.DataFrame(rows).to_excel(excel, index=False)
        torch.manual_seed(901)
        hf_config = Dinov2Config(
            image_size=8, patch_size=4, hidden_size=12,
            num_hidden_layers=1, num_attention_heads=3, mlp_ratio=2,
            hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
        )
        hf_config._attn_implementation = "eager"
        model_dir = self.root / "tiny_hf_snapshot"
        Dinov2Model(hf_config).save_pretrained(model_dir, safe_serialization=True)
        BitImageProcessor(
            size={"height": 8, "width": 8}, do_center_crop=False,
            image_mean=[0.5] * 3, image_std=[0.25] * 3,
        ).save_pretrained(model_dir)
        self.config = _merge(DEFAULTS, {
            "model": "rad_dino", "model_dir": str(model_dir),
            "checkpoint_source": "test-fixture:local-tiny-dinov2",
            "device": "cpu", "seed": 17,
            "output_root": str(self.root / "runs"),
            "split_manifest": str(self.root / "frozen_split.csv"),
            "split_seed": 9, "split_ratios": [0.6, 0.2, 0.2],
            "data": {"excel_path": str(excel), "dicom_dir": str(dicom_dir),
                     "patient_id_source": "excel", "label_definition": "Synthetic software-test labels"},
            "training": {
                "epochs": 2, "patience": 5, "batch_size": 3,
                "accumulation_steps": 3, "num_workers": 0,
                "head_lr": 0.02, "lora_lr": 0.01, "minimum_lr": 1e-5,
                "lora_rank": 2, "lora_alpha": 4.0, "hidden_dims": [8],
                "amp": False, "rotation_degrees": 5.0,
                "horizontal_flip_probability": 0.5,
            },
            "evaluation": {"bootstrap_samples": 8},
        })

    def tearDown(self):
        self.temporary.cleanup()

    def _run_with_first_epoch_checkpoint(self, config, name):
        checkpoint = self.root / name
        original_save = trainer._torch_save

        def capture_first_epoch(path, payload):
            original_save(path, payload)
            if Path(path).name == "last.pt" and payload["epoch"] == 0:
                shutil.copy2(path, checkpoint)

        with redirect_stdout(io.StringIO()), patch.object(
                trainer, "_torch_save", side_effect=capture_first_epoch):
            run = trainer.run_training(config)
        self.assertTrue(checkpoint.is_file())
        return run, checkpoint

    def _assert_state_equal(self, expected, actual):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        elif isinstance(expected, dict):
            self.assertEqual(set(expected), set(actual))
            for key in expected:
                self._assert_state_equal(expected[key], actual[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(expected), len(actual))
            for first, second in zip(expected, actual):
                self._assert_state_equal(first, second)
        else:
            self.assertEqual(expected, actual)

    def test_linear_probe_and_lora_train_save_reload_and_evaluate(self):
        for mode in ("linear_probe", "lora"):
            with self.subTest(mode=mode):
                config = _merge(self.config, {"mode": mode})
                adapter = build_adapter(config)
                frozen = [(parameter, parameter.detach().clone()) for parameter in adapter.parameters()]
                initial = {}
                original_classifier = trainer._classifier

                def capture_classifier(*args):
                    model = original_classifier(*args)
                    initial.update({name: value.clone() for name, value in model.trainable_state_dict().items()})
                    return model

                with redirect_stdout(io.StringIO()), patch.object(trainer, "_classifier", side_effect=capture_classifier), patch.object(
                    trainer, "select_threshold", wraps=select_threshold
                ) as threshold_selector:
                    run = trainer.run_training(config, adapter=adapter)
                self.assertEqual(threshold_selector.call_count, 1)
                self.assertTrue(all(torch.equal(parameter.detach(), before) for parameter, before in frozen))
                best = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
                self.assertEqual(best["model_metadata"]["mode"], mode)
                self.assertEqual(set(initial), set(best["model"]))
                self.assertTrue(any(not torch.equal(value, initial[name]) for name, value in best["model"].items()))
                if mode == "lora":
                    self.assertTrue(any(name.endswith("lora_B") and torch.count_nonzero(value).item() > 0
                                        for name, value in best["model"].items()))
                else:
                    self.assertEqual(set(best["model"]), {"head.weight", "head.bias"})
                selection = json.loads((run / "selection.json").read_text())
                validation = pd.read_csv(run / "validation_predictions.csv")
                patients = aggregate_patient_predictions(validation)
                selected = select_threshold(patients.label, patients.probability)
                self.assertAlmostEqual(selected, selection["threshold"], places=14)
                self.assertEqual(selection["threshold"], best["threshold"])
                test_predictions = pd.read_csv(run / "test" / "predictions.csv")
                self.assertFalse(set(validation.patient_id) & set(test_predictions.patient_id))
                # The selector's only call receives validation patient scores.
                np.testing.assert_array_equal(threshold_selector.call_args.args[0], patients.label)
                np.testing.assert_allclose(threshold_selector.call_args.args[1], patients.probability, rtol=0, atol=1e-14)
                test_metrics = json.loads((run / "test" / "metrics.json").read_text())
                self.assertEqual(test_metrics["patient"]["threshold"], best["threshold"])
                history = json.loads((run / "history.json").read_text())
                self.assertEqual(len(history), 2)
                self.assertTrue(all(np.isfinite(row["val_loss"]) and row["val_loss"] >= 0 for row in history))
                for filename in ("training_loss.png", "validation_auc.png"):
                    self.assertGreater((run / filename).stat().st_size, 1000)
                # Build a fresh real encoder from the local snapshot and reload best.pt.
                with redirect_stdout(io.StringIO()), patch.object(trainer, "select_threshold", wraps=select_threshold) as evaluation_selector:
                    evaluated = trainer.evaluate(config, run / "best.pt")
                self.assertEqual(evaluation_selector.call_count, 0)
                pd.testing.assert_frame_equal(test_predictions, pd.read_csv(evaluated / "predictions.csv"), check_exact=True)
                self.assertEqual(test_metrics, json.loads((evaluated / "metrics.json").read_text()))
                for filename in ("roc_curve.png", "precision_recall_curve.png", "calibration_curve.png"):
                    self.assertGreater((evaluated / filename).stat().st_size, 1000)

    def test_cached_data_matches_uncached_training_exactly(self):
        for mode in ("linear_probe", "lora"):
            with self.subTest(mode=mode):
                config = _merge(self.config, {"mode": mode})
                cached = _merge(config, {"cache_dir": str(self.root / "data_cache")})
                with redirect_stdout(io.StringIO()):
                    direct_run = trainer.run_training(config)
                    cached_run = trainer.run_training(cached)
                direct = torch.load(direct_run / "last.pt", map_location="cpu", weights_only=True)
                restored = torch.load(cached_run / "last.pt", map_location="cpu", weights_only=True)
                self.assertEqual(direct["history"], restored["history"])
                self.assertEqual(direct["cohort_hash"], restored["cohort_hash"])
                for name, parameter in direct["model"].items():
                    torch.testing.assert_close(parameter, restored["model"][name], rtol=0, atol=0)
                pd.testing.assert_frame_equal(
                    pd.read_csv(direct_run / "test" / "predictions.csv"),
                    pd.read_csv(cached_run / "test" / "predictions.csv"), check_exact=True,
                )
        self.assertEqual(len(list((self.root / "data_cache" / "pixels").glob("*.npy"))), 22)

    def test_epoch_checkpoint_resume_matches_uninterrupted_training_exactly(self):
        config = _merge(self.config, {"mode": "lora"})
        epoch_zero = self.root / "epoch_zero.pt"
        original_save = trainer._torch_save

        def capture_first_epoch(path, payload):
            original_save(path, payload)
            if Path(path).name == "last.pt" and payload["epoch"] == 0:
                shutil.copy2(path, epoch_zero)

        with redirect_stdout(io.StringIO()), patch.object(trainer, "_torch_save", side_effect=capture_first_epoch):
            uninterrupted = trainer.run_training(config)
        self.assertTrue(epoch_zero.exists())
        with redirect_stdout(io.StringIO()):
            resumed = trainer.run_training(config, resume=epoch_zero)
        full_last = torch.load(uninterrupted / "last.pt", map_location="cpu", weights_only=True)
        resumed_last = torch.load(resumed / "last.pt", map_location="cpu", weights_only=True)
        self.assertEqual(full_last["history"], resumed_last["history"])
        self.assertEqual(full_last["scheduler"], resumed_last["scheduler"])
        self.assertEqual(full_last["epoch"], resumed_last["epoch"])
        for name, tensor in full_last["model"].items():
            torch.testing.assert_close(tensor, resumed_last["model"][name], rtol=0, atol=0)
        pd.testing.assert_frame_equal(
            pd.read_csv(uninterrupted / "test" / "predictions.csv"),
            pd.read_csv(resumed / "test" / "predictions.csv"), check_exact=True,
        )
        self.assertEqual(json.loads((uninterrupted / "selection.json").read_text()),
                         json.loads((resumed / "selection.json").read_text()))

    def test_amp_change_on_ordinary_resume_is_rejected(self):
        source = _merge(self.config, {"mode": "lora", "training": {"amp": True}})
        _, checkpoint = self._run_with_first_epoch_checkpoint(source, "amp_source.pt")
        full_precision = _merge(source, {"training": {"amp": False}})
        previous_bytes = checkpoint.read_bytes()
        with redirect_stdout(io.StringIO()), patch.object(
                trainer, "_train_epoch", wraps=trainer._train_epoch) as epoch:
            with self.assertRaises(ValueError):
                trainer.run_training(full_precision, resume=checkpoint)
        self.assertEqual(epoch.call_count, 0)
        self.assertEqual(checkpoint.read_bytes(), previous_bytes)

    def test_explicit_full_precision_resume_preserves_continuity_and_provenance(self):
        # CPU execution is FP32 for both flags. This makes any unintended loss
        # of optimizer, scheduler, best-model, augmentation or RNG state visible
        # as an exact difference after the explicit precision transition.
        config = _merge(self.config, {"mode": "lora", "training": {"amp": True}})
        uninterrupted, checkpoint = self._run_with_first_epoch_checkpoint(
            config, "amp_epoch_zero.pt")
        unchanged = copy.deepcopy(config)
        checkpoint_bytes = checkpoint.read_bytes()
        source = torch.load(checkpoint, map_location="cpu", weights_only=True)
        with redirect_stdout(io.StringIO()):
            resumed = trainer.run_training(
                config, resume=checkpoint, resume_full_precision=True)
        self.assertEqual(config, unchanged)
        self.assertEqual(checkpoint.read_bytes(), checkpoint_bytes)
        full_last = torch.load(uninterrupted / "last.pt", map_location="cpu", weights_only=True)
        resumed_last = torch.load(resumed / "last.pt", map_location="cpu", weights_only=True)
        self.assertFalse(resumed_last["config"]["training"]["amp"])
        for field in ("model", "best_model", "optimizer", "scheduler", "rng", "history",
                      "epoch", "best_auc", "bad_epochs", "cohort_hash", "model_metadata"):
            self._assert_state_equal(full_last[field], resumed_last[field])
        self.assertEqual(resumed_last["history"][:len(source["history"])], source["history"])
        self.assertEqual([row["epoch"] for row in resumed_last["history"]], [1, 2])
        self.assertEqual(json.loads((resumed / "config.json").read_text())["training"]["amp"], False)
        pd.testing.assert_frame_equal(
            pd.read_csv(uninterrupted / "test" / "predictions.csv"),
            pd.read_csv(resumed / "test" / "predictions.csv"), check_exact=True,
        )
        self.assertEqual(json.loads((uninterrupted / "selection.json").read_text()),
                         json.loads((resumed / "selection.json").read_text()))
        provenance = json.loads((resumed / "resume.json").read_text())
        self.assertEqual(provenance["checkpoint_sha256"],
                         hashlib.sha256(checkpoint_bytes).hexdigest())
        self.assertEqual(provenance["checkpoint"], str(checkpoint.resolve()))
        self.assertEqual(provenance["epochs_completed_before_resume"], 1)
        self.assertEqual(provenance["next_epoch"], 2)
        self.assertTrue(provenance["source_amp"])
        self.assertFalse(provenance["target_amp"])
        self.assertTrue(provenance["precision_changed"])
        self.assertTrue(provenance["history_before_resume_uses_source_precision"])

    def test_full_precision_resume_does_not_allow_other_training_or_seed_changes(self):
        source = _merge(self.config, {"mode": "lora", "training": {"amp": True}})
        _, checkpoint = self._run_with_first_epoch_checkpoint(source, "strict_amp_source.pt")
        changes = (
            {"training": {"head_lr": 0.03}},
            {"training": {"batch_size": 2}},
            {"training": {"accumulation_steps": 2}},
            {"training": {"epochs": 3}},
            {"seed": source["seed"] + 1},
        )
        for change in changes:
            with self.subTest(change=change), redirect_stdout(io.StringIO()), patch.object(
                    trainer, "_train_epoch", wraps=trainer._train_epoch) as epoch:
                altered = _merge(source, change)
                unchanged = copy.deepcopy(altered)
                with self.assertRaises(ValueError):
                    trainer.run_training(altered, resume=checkpoint, resume_full_precision=True)
                self.assertEqual(epoch.call_count, 0)
                self.assertEqual(altered, unchanged)

    def test_full_precision_resume_requires_checkpoint_and_amp_source(self):
        config = _merge(self.config, {"mode": "lora"})
        with redirect_stdout(io.StringIO()), patch.object(
                trainer, "_train_epoch", wraps=trainer._train_epoch) as epoch:
            with self.assertRaises(ValueError):
                trainer.run_training(config, resume_full_precision=True)
            self.assertEqual(epoch.call_count, 0)
        _, checkpoint = self._run_with_first_epoch_checkpoint(config, "fp32_source.pt")
        previous_bytes = checkpoint.read_bytes()
        with redirect_stdout(io.StringIO()), patch.object(
                trainer, "_train_epoch", wraps=trainer._train_epoch) as epoch:
            with self.assertRaises(ValueError):
                trainer.run_training(config, resume=checkpoint, resume_full_precision=True)
            self.assertEqual(epoch.call_count, 0)
        self.assertEqual(checkpoint.read_bytes(), previous_bytes)

    def test_full_precision_resume_rejects_nonfinite_checkpoint_parameters(self):
        config = _merge(self.config, {"mode": "lora", "training": {"amp": True}})
        _, checkpoint = self._run_with_first_epoch_checkpoint(config, "finite_amp_source.pt")
        original_bytes = checkpoint.read_bytes()
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value):
                damaged = torch.load(checkpoint, map_location="cpu", weights_only=True)
                name = sorted(damaged["model"])[0]
                damaged["model"][name].reshape(-1)[0] = value
                damaged_path = self.root / "invalid_amp_source.pt"
                torch.save(damaged, damaged_path)
                with redirect_stdout(io.StringIO()), patch.object(
                        trainer, "_train_epoch", wraps=trainer._train_epoch) as epoch:
                    with self.assertRaisesRegex(ValueError, "(?i)non.?finite"):
                        trainer.run_training(
                            config, resume=damaged_path, resume_full_precision=True)
                    self.assertEqual(epoch.call_count, 0)
        self.assertEqual(checkpoint.read_bytes(), original_bytes)


class GradientAccumulationTests(unittest.TestCase):
    def test_unequal_minibatches_match_one_full_batch(self):
        class BinaryHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 1)

            def forward(self, images):
                return self.linear(images).squeeze(-1)

        torch.manual_seed(72)
        whole = BinaryHead()
        accumulated = copy.deepcopy(whole)
        images = torch.randn(5, 2)
        labels = torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0])
        samples = [{"image": image, "label": label} for image, label in zip(images, labels)]
        losses = []
        for model, batch_size, accumulation in ((whole, 5, 1), (accumulated, 2, 3)):
            losses.append(trainer._train_epoch(
                model, DataLoader(samples, batch_size=batch_size, shuffle=False),
                torch.optim.SGD(model.parameters(), lr=0.2), trainer.FocalLoss(),
                torch.device("cpu"),
                {"amp": False, "accumulation_steps": accumulation, "gradient_clip": 1e9},
                torch.cuda.amp.GradScaler(enabled=False),
            ))
        self.assertAlmostEqual(losses[0], losses[1], places=7)
        for full_parameter, accumulated_parameter in zip(whole.parameters(), accumulated.parameters()):
            torch.testing.assert_close(full_parameter, accumulated_parameter, rtol=1e-7, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
