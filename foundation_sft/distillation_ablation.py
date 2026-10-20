"""Six loss ablations using one RAD-DINO LoRA teacher and full ViT-Tiny students.

All groups retain the original true-label loss. Only the three fixed KD weights
vary. Planning and result printing are standard-library-only read operations.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

from . import distillation_suite as coordinator


FORMAT = "distillation_ablation_v1"
DEFAULT_CONFIG = "foundation_sft/configs/distillation_ablation.json"
GROUPS = {
    "supervised": ("supervised", 0.0, 0.0, 0.0),
    "soft_only": ("distillation", 0.5, 0.0, 0.0),
    "no_local": ("distillation", 0.5, 0.1, 0.0),
    "no_global": ("distillation", 0.5, 0.0, 1.0),
    "no_soft": ("distillation", 0.0, 0.1, 1.0),
    "full_kd": ("distillation", 0.5, 0.1, 1.0),
}
WEIGHTS = ("soft_weight", "global_weight", "local_weight")


def run_suite(plan, **kwargs):
    """Use the shared checked engine, then add the matched-seed loss contrasts."""
    from .distillation_ablation_report import print_paired_metrics, write_paired_reports

    suite = coordinator.run_suite(plan, **kwargs)
    state = coordinator._read(suite / "suite.json")
    write_paired_reports(suite, state)
    print_paired_metrics(state)
    return suite


def make_plan(config_path, project, *, groups=None, seeds=None, devices=None,
              max_parallel=None, cache_dir=None, reuse_suites=None, no_reuse=False):
    """Copy the original full-finetune protocol, varying only the KD weights."""
    project = Path(project).expanduser().resolve()
    path = coordinator._path(config_path, project)
    settings = coordinator._read(path)
    configured = settings.get("groups", {})
    if not isinstance(configured, dict) or set(configured) != set(GROUPS):
        raise ValueError("The ablation config must define the six fixed loss groups")
    for name, expected in GROUPS.items():
        group = configured[name]
        if (not isinstance(group, dict)
                or set(group) != {"condition", *WEIGHTS}
                or tuple(group.get(key) for key in ("condition", *WEIGHTS)) != expected
                or any(type(group[key]) not in (int, float) for key in WEIGHTS)):
            raise ValueError(f"Group {name} must use the predefined condition and KD weights {expected}")
    selected = list(groups if groups is not None else GROUPS)
    if not selected or len(set(selected)) != len(selected) or any(group not in GROUPS for group in selected):
        raise ValueError("groups must be unique members of " + " ".join(GROUPS))
    chosen_seeds = list(seeds if seeds is not None else settings.get("seeds", [42, 43, 44]))
    if (not chosen_seeds or len(set(chosen_seeds)) != len(chosen_seeds)
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in chosen_seeds)):
        raise ValueError("seeds must be unique integers in [0, 2**32)")
    chosen_devices = list(devices if devices is not None else settings.get("devices", ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]))
    if not chosen_devices or any(not isinstance(device, str) or not device for device in chosen_devices):
        raise ValueError("devices must contain at least one nonempty device name")
    maximum = max_parallel if max_parallel is not None else settings.get("max_parallel", 4)
    if type(maximum) is not int or maximum < 1:
        raise ValueError("max_parallel must be a positive integer")
    template_path = coordinator._path(settings["template"], project)
    template = coordinator._read(template_path)
    if coordinator._student_mode(template) != "full_finetune":
        raise ValueError("Loss ablation uses the full-finetune student; LoRA changes a second factor")
    original_kd = template.get("distillation", {})
    if original_kd.get("temperature") != 2.0 or original_kd.get("pool_grid") != [7, 7]:
        raise ValueError("Loss ablation requires temperature=2.0 and pool_grid=[7, 7]")
    if tuple(original_kd.get(key) for key in WEIGHTS) != GROUPS["full_kd"][1:]:
        raise ValueError("The full template must retain the original three KD weights")
    training = template["training"]
    if (training.get("backbone_lr") != 5e-5 or training.get("head_lr") != 1e-4
            or training.get("projection_lr") != 1e-4 or training.get("amp") is not False):
        raise ValueError("Loss ablation retains the original FP32 full-student training learning rates")
    output_root = coordinator._path(settings.get("output_root", "runs/distillation_ablation"), project)
    jobs = []
    for group in selected:
        index = list(GROUPS).index(group) if devices is None else selected.index(group)
        device = chosen_devices[index % len(chosen_devices)]
        condition, *weights = GROUPS[group]
        for seed in chosen_seeds:
            config = deepcopy(template)
            config.update(seed=seed, device=device, output_root=str(output_root / "children"))
            config["distillation"].update(dict(zip(WEIGHTS, weights)))
            if cache_dir is not None:
                config["cache_dir"] = str(coordinator._path(cache_dir, project))
            jobs.append({"id": f"{group}_seed{seed}", "group": group, "seed": seed,
                         "student_mode": "full_finetune", "condition": condition,
                         "device": device, "config": config})
    if no_reuse and reuse_suites:
        raise ValueError("--no-reuse cannot be combined with --reuse-suite")
    reuse = [] if no_reuse else list(settings.get("reuse_suites", [])) + list(reuse_suites or [])
    # Keep the declared order, remove duplicates, and never discover other suites.
    reuse = list(dict.fromkeys(str(coordinator._path(value, project)) for value in reuse))
    return {"format_version": FORMAT, "project": str(project), "config_path": str(path),
            "output_root": str(output_root), "max_parallel": maximum,
            "groups": selected, "seeds": chosen_seeds, "reuse_suites": reuse, "jobs": jobs}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--groups", choices=list(GROUPS), nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--devices", nargs="+")
    parser.add_argument("--max-parallel", type=int)
    parser.add_argument("--cache-dir")
    parser.add_argument("--reuse-suite", action="append", dest="reuse_suites",
                        help="Append one explicit saved suite; may be repeated (never selected by score)")
    parser.add_argument("--no-reuse", action="store_true", help="Train fresh, ignoring configured saved suites")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--plan", action="store_true", help="Print all resolved jobs without writing files")
    actions.add_argument("--check-only", action="store_true", help="Preflight the selected groups without training")
    actions.add_argument("--show-metrics", action="store_true", help="Read all saved ablation batches without Torch")
    parser.add_argument("--resume-suite")
    parser.add_argument("--suite", help="With --show-metrics, print this saved batch only")
    args = parser.parse_args(argv)
    try:
        project = Path(args.project).expanduser().resolve()
        if args.show_metrics:
            if args.resume_suite:
                raise ValueError("--resume-suite is for training or checking, not --show-metrics")
            settings = coordinator._read(coordinator._path(args.config, project))
            batches = coordinator.saved_batches(
                coordinator._path(settings.get("output_root", "runs/distillation_ablation"), project),
                coordinator._path(args.suite, project) if args.suite else None)
            if not batches:
                print("No saved loss-ablation batches found.")
            coordinator.print_metrics(batches)
            from .distillation_ablation_report import print_paired_metrics
            for batch, state in batches:
                print(f"\nPaired batch: {batch}")
                print_paired_metrics(state)
            return 0
        if args.suite:
            raise ValueError("--suite requires --show-metrics; use --resume-suite to continue a batch")
        plan = make_plan(args.config, project, groups=args.groups, seeds=args.seeds, devices=args.devices,
                         max_parallel=args.max_parallel, cache_dir=args.cache_dir,
                         reuse_suites=args.reuse_suites, no_reuse=args.no_reuse)
        if args.plan:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return 0
        suite = run_suite(plan, check_only=args.check_only, resume_suite=args.resume_suite)
        return 1 if coordinator._read(suite / "suite.json")["status"] in {"partial", "failed"} else 0
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"Loss-ablation workflow failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
