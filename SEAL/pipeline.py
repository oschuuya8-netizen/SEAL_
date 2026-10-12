import json
import hashlib
from copy import deepcopy
from collections import Counter
from pathlib import Path

from execution import ExecutionPlanner
from labels import STRATEGIES
from generator import ExecutionConditionedGenerator
from generation_prompts import word_count
from strategy_selector import DynamicStrategySelector
from utils import format_context, load_json, normalize, save_json


class SEALPipeline:
    def __init__(self, args):
        self.args = args
        self.data_sha256 = hashlib.sha256(Path(args.data).read_bytes()).hexdigest()

    def run(self, samples):
        selections = self._select(samples)

        use_execution = self.args.ablation != "without_execution"
        planner = (
            ExecutionPlanner(self.args.execution_checkpoint, self.args.execution_bank)
            if use_execution
            else None
        )
        generator = ExecutionConditionedGenerator(
            self.args,
            planner.z_dim if planner is not None else 0,
            use_execution=use_execution,
        )
        results = []
        for index, (sample, selection) in enumerate(zip(samples, selections), start=1):
            context_text = format_context(sample["context"])
            strategy = selection["selected_strategy"]
            if self.args.ablation == "without_execution":
                execution_vector = None
                neighbor = None
                execution_mode = "none"
            elif self.args.ablation == "static_execution":
                execution_vector = planner.static_plan(strategy)
                neighbor = planner.nearest_execution(execution_vector)
                execution_mode = "strategy_centroid"
            else:
                execution_vector = planner.plan(context_text, strategy)
                neighbor = planner.nearest_execution(execution_vector)
                execution_mode = "sample_specific"
            generation = generator.generate(
                context_text,
                strategy,
                execution_vector,
                sample_seed=self.args.seed + index,
            )
            result = {
                "index": index,
                **sample,
                "selected_strategy": strategy,
                "strategy_correct": strategy == sample.get("reference_strategy"),
                "strategy_top2": selection["strategy_top2"],
                "strategy_top2_correct": sample.get("reference_strategy") in selection["strategy_top2"],
                "selector": selection,
                "execution_mode": execution_mode,
                "nearest_execution": neighbor,
                "generated_response": generation["response"],
                "response_selection": "single_decode",
                "grounding_verified": False,
                "generation_failed": generation["generation_failed"],
            }
            results.append(result)
            print(json.dumps({
                "index": index,
                "id": sample["id"],
                "stage": selection["stage"],
                "alpha": round(selection["alpha"], 4),
                "selected_strategy": strategy,
                "reference_strategy": sample.get("reference_strategy"),
                "generated_response": generation["response"],
            }, ensure_ascii=False), flush=True)
            save_json(self.args.output, self._payload(results, complete=False))
        payload = self._payload(results, complete=True)
        save_json(self.args.output, payload)
        return payload

    def _select(self, samples):
        if self.args.selector_cache:
            payload = load_json(self.args.selector_cache)
            all_cached_rows = payload.get("results", [])
            if len(all_cached_rows) < len(samples):
                raise ValueError(
                    f"Selector cache has {len(all_cached_rows)} rows, need {len(samples)}"
                )
            cached_rows = all_cached_rows[:len(samples)]
            expected = [(row["id"], row.get("source_index")) for row in samples]
            actual = [(row["id"], row.get("source_index")) for row in cached_rows]
            if expected != actual:
                raise ValueError("Selector cache sample keys or order do not match the input")
            selections = [deepcopy(row["selector"]) for row in cached_rows]
        else:
            selector = DynamicStrategySelector(self.args)
            selections = selector.select_batch(samples)
            del selector

        for selection in selections:
            if self.args.ablation == "without_behavior_prior":
                distribution = normalize(selection["stage_strategy_distribution"])
                selection["alpha"] = 0.0
                selection["selection_components"] = "esc_stage_only"
            elif self.args.ablation == "without_esc_stage":
                distribution = normalize(selection["behavior_strategy_distribution"])
                selection["alpha"] = 1.0
                selection["selection_components"] = "behavior_prior_only"
            else:
                distribution = normalize(selection["combined_distribution"])
                selection["selection_components"] = "dynamic_fusion"
            ranking = sorted(
                STRATEGIES,
                key=lambda label: (-distribution[label], STRATEGIES.index(label)),
            )
            selection["combined_distribution"] = distribution
            selection["selected_strategy"] = ranking[0]
            selection["strategy_top2"] = ranking[:2]
            selection["ablation"] = self.args.ablation
        return selections

    def _payload(self, results, complete):
        alphas = [row["selector"]["alpha"] for row in results]
        response_lengths = [word_count(row["generated_response"]) for row in results]
        strategy_metrics = self._strategy_metrics(results)
        summary = {
            "complete": complete,
            "response_selection": "single_decode",
            "grounding_verification": "disabled",
            "ablation": self.args.ablation,
            "selector_cache": self.args.selector_cache,
            "num_results": len(results),
            "data_path": self.args.data,
            "data_sha256": self.data_sha256,
            "seed": self.args.seed,
            "preserve_order": self.args.preserve_order,
            "sample_seed_formula": "seed + 1-based output index",
            "strategy_accuracy": (
                sum(row["strategy_correct"] for row in results) / len(results) if results else None
            ),
            "strategy_macro_f1": strategy_metrics["macro_f1"],
            "strategy_top2_accuracy": strategy_metrics["top2_accuracy"],
            "strategy_per_class_f1": strategy_metrics["per_class_f1"],
            "strategy_label_set": list(STRATEGIES),
            "stage_counts": dict(Counter(row["selector"]["stage"] for row in results)),
            "stage_parse_errors": sum(bool(row["selector"]["stage_error"]) for row in results),
            "alpha_min": min(alphas) if alphas else None,
            "alpha_mean": sum(alphas) / len(alphas) if alphas else None,
            "alpha_max": max(alphas) if alphas else None,
            "average_response_words": (
                sum(response_lengths) / len(response_lengths) if response_lengths else None
            ),
            "responses_in_target_range": sum(
                self.args.response_min_words <= count <= self.args.response_max_words
                for count in response_lengths
            ),
            "generation_failures": sum(row["generation_failed"] for row in results),
            "selector_parameters": {
                "beta": self.args.beta,
                "rank_temperature": self.args.rank_temperature,
                "alpha_low": self.args.alpha_low,
                "alpha_high": self.args.alpha_high,
            },
            "generation_parameters": {
                "min_new_tokens": self.args.min_new_tokens,
                "max_new_tokens": self.args.max_new_tokens,
                "do_sample": self.args.do_sample,
                "repetition_penalty": self.args.repetition_penalty,
                "no_repeat_ngram_size": self.args.no_repeat_ngram_size,
                "num_return_sequences": 1,
                "temperature": self.args.temperature if self.args.do_sample else None,
                "top_p": self.args.top_p if self.args.do_sample else None,
                "response_min_words": self.args.response_min_words,
                "response_max_words": self.args.response_max_words,
            },
        }
        return {"summary": summary, "results": results}

    @staticmethod
    def _strategy_metrics(results):
        per_class = {}
        for label in STRATEGIES:
            true_positive = sum(
                row.get("reference_strategy") == label and row["selected_strategy"] == label
                for row in results
            )
            false_positive = sum(
                row.get("reference_strategy") != label and row["selected_strategy"] == label
                for row in results
            )
            false_negative = sum(
                row.get("reference_strategy") == label and row["selected_strategy"] != label
                for row in results
            )
            denominator = 2 * true_positive + false_positive + false_negative
            per_class[label] = 2 * true_positive / denominator if denominator else 0.0
        return {
            "macro_f1": sum(per_class.values()) / len(STRATEGIES),
            "top2_accuracy": (
                sum(row["strategy_top2_correct"] for row in results) / len(results)
                if results else None
            ),
            "per_class_f1": per_class,
        }
