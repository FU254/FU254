"""The new experiment batch preserves the existing frozen data definition."""

from pathlib import Path
import unittest

from foundation_sft.config import load_config
from foundation_sft.workflow import build_plan


class VLMConfigTests(unittest.TestCase):
    def test_four_models_have_two_modes_shared_cohort_and_separate_results(self):
        root = Path(__file__).resolve().parents[1]
        # The command is intentionally run from the repository root, like the
        # server workflow. Avoid changing global cwd during parallel tests.
        plan = build_plan(root / "foundation_sft/configs/experiments_vlm.json")
        previous = load_config(root / "foundation_sft/configs/eva_x.json")
        self.assertEqual(len(plan["jobs"]), 8)
        expected = {"cxr_clip": ("evax_lora", "cuda:0"), "xray_siglip": ("cxr_openclip", "cuda:1"),
                    "biomedclip": ("cxr_openclip", "cuda:2"), "chexzero": ("evax_lora", "cuda:3")}
        for model, (environment, device) in expected.items():
            jobs = [job for job in plan["jobs"] if job["model"] == model]
            self.assertEqual({job["mode"] for job in jobs}, {"linear_probe", "lora"})
            for job in jobs:
                config = job["config"]
                self.assertEqual((job["env"], job["device"], job["seed"]), (environment, device, 42))
                for name in ("data", "split_manifest", "split_seed", "split_ratios"):
                    self.assertEqual(config[name], previous[name])
                self.assertEqual(config["output_root"], "runs/foundation_sft_vlm")
                self.assertIs(config["training"]["amp"], False)
                self.assertEqual(config["training"]["batch_size"] * config["training"]["accumulation_steps"], 64)
                if model in {"xray_siglip", "biomedclip"}:
                    self.assertNotIn("preprocess_override", config)
        self.assertEqual(plan["suite_root"], "runs/foundation_sft_vlm/suites")


if __name__ == "__main__":
    unittest.main()
