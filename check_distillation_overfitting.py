#!/usr/bin/env python3
"""Check six completed seed42 ablations using clean best/last train and val inference."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import sys
import uuid

from run_single_teacher_test_evaluation import _load_payload, _read, _resolve_device, _same, _sha, _write

FORMAT = "distillation_posthoc_overfit_v1"
NOTE = ("Clean hard-label Focal loss excludes KD and regularizers; patient AUCs use mean image "
        "probabilities. Only best/last clean training points are available; original augmented "
        "training objectives are not plotted. Positive train-minus-val AUC or val-minus-train "
        "loss gaps and deteriorating validation after selection are descriptive evidence to "
        "inspect, not a universal cutoff or proof of overfitting. Validation selected best.pt; "
        "this is not an independent generalization estimate. No test images or test metrics are used.")


def _preflight(suite):
    """Validate every source and all twelve payloads before opening image datasets."""
    import pandas as pd
    from foundation_sft.distillation_ablation import FORMAT as SOURCE_FORMAT, GROUPS, WEIGHTS
    from foundation_sft.distillation_suite import protocol_hash, _student_mode, _validate_protocol

    suite = Path(suite).expanduser().resolve()
    state = _read(suite / "suite.json")
    _same("suite format", state.get("format_version"), SOURCE_FORMAT)
    _same("completed suite", state.get("status"), "completed")
    plan = state["plan"]
    _same("frozen protocol", state.get("protocol_sha256"), protocol_hash(plan))
    expected = {name + "_seed42" for name in GROUPS}
    jobs = {job["id"]: job for job in plan["jobs"]}
    _same("planned job count", len(plan["jobs"]), 6)
    _same("planned jobs", set(jobs), expected)
    _same("completed jobs", set(state["jobs"]), expected)
    entries, reference, split_reference = [], None, None
    for group, ablation in zip("ABCDEF", GROUPS):
        job = jobs[ablation + "_seed42"]
        record = state["jobs"][job["id"]]
        if record.get("status") not in {"completed", "reused", "resumed"}:
            raise ValueError("Incomplete group " + group)
        _same("saved job", record["job"], job)
        for name, expected_value in (("group", ablation), ("seed", 42),
                                     ("student_mode", "full_finetune"), ("condition", GROUPS[ablation][0])):
            _same("job " + name, job.get(name), expected_value)
        _same("planned KD weights", tuple(job["config"]["distillation"][key] for key in WEIGHTS), GROUPS[ablation][1:])
        receipt, check = record["result"], record["check"]
        _same("source preflight", check.get("check_passed"), True)
        run = Path(receipt["run"]).expanduser()
        run = (run if run.is_absolute() else suite / run).resolve()
        files = {name: _sha(run / name) for name in ("best.pt", "last.pt", "history.json", "config.json",
                 "result.json", "model_metadata.json", "shared_view.json", "teacher.json", "split.csv", "split_metadata.json")}
        _same("best checkpoint SHA256", files["best.pt"], receipt["checkpoint_sha256"])
        if receipt.get("history_sha256"):
            _same("history SHA256", files["history.json"], receipt["history_sha256"])
        config, result = _read(run / "config.json"), _read(run / "result.json")
        metadata, geometry = _read(run / "model_metadata.json"), _read(run / "shared_view.json")
        teacher = _read(run / "teacher.json")
        _same("saved seed", config.get("seed"), 42)
        _same("full student", _student_mode(config), "full_finetune")
        _same("full metadata", metadata.get("mode", "full_finetune"), "full_finetune")
        _same("ImageNet initialization", metadata.get("initialization", {}).get("pretraining"), "ImageNet-1k")
        _validate_protocol(config, check["config"], job["condition"])
        _same("check seed", check["config"].get("seed"), 42)
        _same("check full student", _student_mode(check["config"]), "full_finetune")
        _same("check condition", check.get("condition"), job["condition"])
        for key, value in job["config"]["training"].items():
            _same("planned training." + key, check["config"]["training"].get(key), value)
        _same("planned distillation", check["config"].get("distillation"), job["config"].get("distillation"))
        for name, value in (("model_metadata", metadata), ("shared_view", geometry)):
            _same("source check " + name, check.get(name), value)
        for name, value in (("status", "completed"), ("condition", job["condition"]), ("seed", 42),
                            ("teacher_unchanged", True), ("cohort_hash", check["cohort_hash"]),
                            ("initial_state_sha256", check["initial_state_sha256"])):
            _same("receipt " + name, receipt.get(name), value)
            _same("result " + name, result.get(name), value)
        teacher_sha = teacher.get("checkpoint_sha256")
        if not teacher_sha:
            raise ValueError("Missing teacher fingerprint")
        _same("frozen teacher fingerprint", teacher_sha, check["teacher"].get("checkpoint_sha256"))
        protocol = {"training": config["training"], "evaluation": config.get("evaluation"),
                    "metadata": metadata, "geometry": geometry, "cohort": check["cohort_hash"],
                    "initial_state": check["initial_state_sha256"], "teacher": teacher_sha}
        reference = protocol if reference is None else reference
        _same("matched A-F scientific protocol", protocol, reference)
        training = config["training"]
        for key in ("focal_gamma", "focal_scale"):
            value = training[key]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (key == "focal_scale" and value == 0):
                raise ValueError("Invalid saved " + key)
        threshold = receipt["threshold"]
        if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Invalid saved validation threshold")
        _same("saved threshold", result.get("threshold"), threshold)
        _same("split cohort", _read(run / "split_metadata.json").get("_manifest_hash"), check["cohort_hash"])
        split = pd.read_csv(run / "split.csv", dtype={"sample_id": str, "patient_id": str}, keep_default_na=False)
        if not {"sample_id", "patient_id", "label", "split", "path"} <= set(split) or split.empty or split.sample_id.duplicated().any():
            raise ValueError("Invalid frozen split columns or duplicate sample IDs")
        if any(split[key].str.strip().eq("").any() for key in ("sample_id", "patient_id")):
            raise ValueError("Empty sample/patient identifier")
        if set(split["split"]) != {"train", "val", "test"} or not split.label.isin([0, 1]).all():
            raise ValueError("Invalid split assignments or labels")
        if (split.groupby("patient_id")["split"].nunique() > 1).any() or (split.groupby("patient_id").label.nunique() > 1).any():
            raise ValueError("Patient overlap or conflicting patient labels")
        normalized = split.sort_values("sample_id").reindex(sorted(split.columns), axis=1).to_dict("records")
        split_reference = normalized if split_reference is None else split_reference
        _same("identical A-F frozen split rows", normalized, split_reference)
        for partition in ("train", "val"):
            _same(partition + " classes", set(split.loc[split["split"] == partition, "label"]), {0, 1})
        history = _read(run / "history.json")
        if not isinstance(history, list) or not history:
            raise ValueError("Missing training history")
        _same("history epochs", [row["epoch"] for row in history], list(range(1, len(history) + 1)))
        for row in history:
            for key in ("val_loss", "val_patient_roc_auc", "val_patient_pr_auc"):
                value = row[key]
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (key != "val_loss" and value > 1):
                    raise ValueError("Invalid history " + key)
        best_epoch = max(range(len(history)), key=lambda index: history[index]["val_patient_roc_auc"]) + 1
        best_auc = history[best_epoch - 1]["val_patient_roc_auc"]
        for value in (receipt, result):
            _same("validation selection", value.get("best_val_patient_roc_auc"), best_auc)
            if "best_epoch" in value:
                _same("saved best epoch", value["best_epoch"], best_epoch)
        for checkpoint in ("best", "last"):
            payload = _load_payload(run / (checkpoint + ".pt"))
            _same("checkpoint format", payload.get("format_version"), "single_teacher_kd_v1")
            for name, value in (("config", config), ("model_metadata", metadata), ("shared_view", geometry),
                                ("condition", job["condition"]), ("cohort_hash", check["cohort_hash"]),
                                ("initial_state_sha256", check["initial_state_sha256"]), ("teacher", teacher)):
                _same(checkpoint + " checkpoint " + name, payload.get(name), value)
            if checkpoint == "best":
                _same("best threshold", payload.get("threshold"), threshold)
                _same("best selected AUC", payload.get("best_val_patient_roc_auc"), best_auc)
                if "best_epoch" in payload:
                    _same("best selected epoch", payload["best_epoch"], best_epoch)
            else:
                _same("last epoch", payload.get("epoch"), len(history) - 1)
                _same("last history", payload.get("history"), history)
                _same("last best AUC", payload.get("best_auc"), best_auc)
            del payload
        entries.append(dict(group=group, ablation=ablation, run=run, config=config, split=split,
                            history=history, best_epoch=best_epoch, threshold=threshold, files=files))
    return dict(suite=suite, entries=entries, suite_sha256=_sha(suite / "suite.json"), protocol_sha256=state["protocol_sha256"])


def _plots(entries, rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for metric, historical, label in (("loss", "val_loss", "Clean image Focal loss"),
            ("roc_auc", "val_patient_roc_auc", "Patient ROC-AUC"),
            ("pr_auc", "val_patient_pr_auc", "Patient PR-AUC (trapezoidal)")):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
        for ax, entry in zip(axes.flat, entries):
            history = entry["history"]
            ax.plot([item["epoch"] for item in history], [item[historical] for item in history], label="Saved validation history", color="tab:blue")
            for checkpoint, marker in (("best", "o"), ("last", "X")):
                row = next(row for row in rows if row["group"] == entry["group"] and row["checkpoint"] == checkpoint)
                ax.scatter(row["epoch"], row["train_" + metric], marker=marker, color="tab:orange", label="Clean train " + checkpoint, zorder=4)
                ax.scatter(row["epoch"], row["val_" + metric], marker=marker, facecolors="none", edgecolors="tab:green", label="Recomputed val " + checkpoint, zorder=3)
            ax.set(title=entry["group"] + " " + entry["ablation"], xlabel="Epoch", ylabel=label)
            ax.grid(alpha=.2)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3)
        fig.suptitle(label + " | clean train: best/last points only; augmented training objective not plotted")
        fig.tight_layout(rect=(0, .09, 1, .94))
        fig.savefig(output / (metric + "_grid.png"), dpi=160)
        plt.close(fig)


def _table(table):
    columns = ["group", "checkpoint", "epoch", "train_loss", "val_loss", "train_roc_auc", "val_roc_auc", "loss_gap", "auc_gap"]
    print(table[columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def audit(suite, device="auto"):
    """Write a unique posthoc report, preserving every source artifact."""
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader
    from foundation_sft.distillation import inference_dataset, load_student_checkpoint
    from foundation_sft.metrics import aggregate_patient_predictions, binary_metrics
    from foundation_sft.train import FocalLoss, predict

    context = _preflight(suite)
    device = _resolve_device(device)
    output = context["suite"] / "report" / ("overfit_check_" + uuid.uuid4().hex)
    output.mkdir(parents=True, exist_ok=False)
    rows, provenance = [], []
    print("Clean best/last inference device: " + str(device), flush=True)
    for entry in context["entries"]:
        training = entry["config"]["training"]
        criterion = FocalLoss(gamma=training["focal_gamma"], scale=training["focal_scale"])
        for checkpoint in ("best", "last"):
            print(f"[{entry['group']} {entry['ablation']}] {checkpoint}: clean train + validation", flush=True)
            path = entry["run"] / (checkpoint + ".pt")
            _same("checkpoint before inference", _sha(path), entry["files"][path.name])
            student, payload = load_student_checkpoint(path, device)
            _same("loaded checkpoint", _sha(path), entry["files"][path.name])
            epoch = entry["best_epoch"] if checkpoint == "best" else len(entry["history"])
            row = dict(group=entry["group"], ablation=entry["ablation"], checkpoint=checkpoint, epoch=epoch)
            for partition in ("train", "val"):
                split = entry["split"].loc[entry["split"]["split"] == partition].copy()
                loader = DataLoader(inference_dataset(split, payload, entry["config"].get("cache_dir")),
                    batch_size=training["batch_size"], shuffle=False, num_workers=training["num_workers"],
                    pin_memory=torch.device(device).type == "cuda")
                predictions, loss = predict(student, loader, torch.device(device), criterion=criterion)
                _same("complete prediction cohort", predictions[["sample_id", "patient_id", "label"]].sort_values("sample_id").to_dict("records"),
                      split[["sample_id", "patient_id", "label"]].sort_values("sample_id").to_dict("records"))
                patients = aggregate_patient_predictions(predictions)
                metrics = binary_metrics(patients.label, patients.probability, threshold=entry["threshold"])
                row.update({partition + "_" + key: metrics[key] for key in ("roc_auc", "pr_auc", "average_precision")})
                row.update({partition + "_loss": loss, partition + "_n_images": len(predictions), partition + "_n_patients": len(patients)})
                predictions.to_csv(output / f"{entry['group']}_{checkpoint}_{partition}_predictions.csv", index=False)
                patients.to_csv(output / f"{entry['group']}_{checkpoint}_{partition}_patient_predictions.csv", index=False)
            row.update(loss_gap=row["val_loss"] - row["train_loss"], auc_gap=row["train_roc_auc"] - row["val_roc_auc"])
            rows.append(row)
            provenance.append(dict(group=entry["group"], ablation=entry["ablation"], checkpoint=checkpoint,
                epoch=epoch, run=str(entry["run"]), files=entry["files"], threshold=entry["threshold"],
                focal_gamma=training["focal_gamma"], focal_scale=training["focal_scale"]))
            del student, payload, loader
    for entry in context["entries"]:
        for name, fingerprint in entry["files"].items():
            _same("source preserved " + name, _sha(entry["run"] / name), fingerprint)
    _same("suite preserved", _sha(context["suite"] / "suite.json"), context["suite_sha256"])
    table = pd.DataFrame(rows)
    table.to_csv(output / "comparison.csv", index=False)
    changes = [{"group": group, **{key + "_last_minus_best": pair[1][key] - pair[0][key]
                for key in ("train_loss", "val_loss", "train_roc_auc", "val_roc_auc", "train_pr_auc", "val_pr_auc", "loss_gap", "auc_gap")}}
               for group in "ABCDEF" for pair in [[row for row in rows if row["group"] == group]]]
    pd.DataFrame(changes).to_csv(output / "best_last_changes.csv", index=False)
    _plots(context["entries"], rows, output)
    _write(output / "provenance.json", dict(format_version=FORMAT, status="completed", device=str(device),
        created_utc=datetime.now(timezone.utc).isoformat(), suite=str(context["suite"]),
        suite_sha256=context["suite_sha256"], protocol_sha256=context["protocol_sha256"],
        train_augmented_original_not_plotted=True, notes=NOTE, checkpoints=provenance))
    (output / "README.txt").write_text(NOTE + "\n", encoding="utf-8")
    _table(table)
    print(NOTE + "\nReport: " + str(output))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, help="Explicit completed six-ablation suite")
    parser.add_argument("--project", help="Project containing foundation_sft (default: this script's folder)")
    parser.add_argument("--device", default="auto", help="auto uses an idle GPU with >=20GB free, otherwise CPU")
    parser.add_argument("--show-report", help="Read an existing report directory without images or Torch")
    args = parser.parse_args(argv)
    project = Path(args.project).expanduser().resolve() if args.project else Path(__file__).resolve().parent
    sys.path.insert(0, str(project))
    suite = Path(args.suite).expanduser()
    suite = (suite if suite.is_absolute() else project / suite).resolve()
    try:
        if args.show_report:
            import pandas as pd
            report = Path(args.show_report).expanduser().resolve()
            provenance = _read(report / "provenance.json")
            _same("report format", provenance.get("format_version"), FORMAT)
            _same("report suite", provenance.get("suite"), str(suite))
            _table(pd.read_csv(report / "comparison.csv"))
            print(provenance["notes"] + "\nReport: " + str(report))
        else:
            audit(suite, device=args.device)
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print("Overfit check failed: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
