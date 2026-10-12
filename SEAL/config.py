import argparse
import os
from pathlib import Path


ROOT = os.environ.get("SEAL_ROOT", str(Path(__file__).resolve().parents[1]))


def parse_args():
    parser = argparse.ArgumentParser(description="Run the SEAL strategy-to-response pipeline.")
    parser.add_argument("--data", default=f"{ROOT}/data/test_samples.json")
    parser.add_argument("--output", default=f"{ROOT}/outputs/seal_test20.json")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--min-response-words", type=int, default=8)
    parser.add_argument(
        "--preserve-order",
        action="store_true",
        help="Use samples in file order; required for a materialized fixed test set.",
    )
    parser.add_argument(
        "--ablation",
        choices=(
            "full",
            "without_behavior_prior",
            "without_esc_stage",
            "without_execution",
            "static_execution",
        ),
        default="full",
    )
    parser.add_argument(
        "--selector-cache",
        default=None,
        help="Reuse sample-aligned selector traces from an existing full SEAL output.",
    )

    parser.add_argument("--behavior-model", default=f"{ROOT}/checkpoints/behavior/best")
    parser.add_argument("--prior-metadata", default=f"{ROOT}/artifacts/clientreaction_prior_counts.json")
    parser.add_argument(
        "--stage-model",
        default="Qwen/Qwen3-8B",
    )
    parser.add_argument("--selector-batch-size", type=int, default=4)
    parser.add_argument("--beta", type=float, default=500.0)
    parser.add_argument("--rank-temperature", type=float, default=4.0)
    parser.add_argument("--alpha-low", type=float, default=0.1)
    parser.add_argument("--alpha-high", type=float, default=1.0)

    parser.add_argument("--execution-checkpoint", default=f"{ROOT}/checkpoints/execution/best")
    parser.add_argument("--execution-bank", default=f"{ROOT}/artifacts/context_strategy_latents.pt")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--lora-adapter", default=f"{ROOT}/checkpoints/generator/last/lora_adapter")
    parser.add_argument("--projector-path", default=f"{ROOT}/checkpoints/generator/last/z_projector.pt")
    parser.add_argument("--max-new-tokens", type=int, default=72)
    parser.add_argument("--min-new-tokens", type=int, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--response-min-words", type=int, default=20)
    parser.add_argument("--response-max-words", type=int, default=30)
    return parser.parse_args()


def configure_environment():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
