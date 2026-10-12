import gc
import math

import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from labels import STAGES, STRATEGIES
from prompts import build_stage_prompt
from utils import extract_json, format_context, load_json, normalize, normalized_confidence


class DynamicStrategySelector:
    def __init__(self, args):
        self.args = args
        self.global_prior, self.behavior_priors = self._build_priors(load_json(args.prior_metadata))

    def _build_priors(self, metadata):
        global_prior = normalize({label: metadata["global_strategy_counts"].get(label, 0) for label in STRATEGIES})
        priors = {}
        for behavior in metadata["behaviors"]:
            counts = metadata["behavior_strategy_counts"].get(behavior, {})
            support = sum(counts.values())
            denominator = support + self.args.beta
            priors[behavior] = {
                label: (counts.get(label, 0) + self.args.beta * global_prior[label]) / denominator
                for label in STRATEGIES
            }
        return global_prior, priors

    def _behavior_posteriors(self, samples):
        tokenizer = AutoTokenizer.from_pretrained(self.args.behavior_model, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.args.behavior_model, local_files_only=True
        ).to("cuda").eval()
        labels = [model.config.id2label[index] for index in range(model.config.num_labels)]
        posteriors = []
        for offset in range(0, len(samples), 16):
            batch = samples[offset:offset + 16]
            histories, current = [], []
            for sample in batch:
                turns = sample["context"]
                current_turn = turns[-1]
                histories.append(format_context(turns[max(0, len(turns) - 7):-1]))
                current.append(current_turn.get("utterance") or current_turn.get("content") or "")
            encoded = tokenizer(
                histories, current, truncation=True, max_length=256, padding=True, return_tensors="pt"
            ).to("cuda")
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                probabilities = model(**encoded).logits.softmax(-1).float().cpu().tolist()
            posteriors.extend(dict(zip(labels, row)) for row in probabilities)
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        return posteriors

    def _stage_rankings(self, samples):
        tokenizer = AutoTokenizer.from_pretrained(self.args.stage_model, local_files_only=True)
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            self.args.stage_model,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to("cuda").eval()

        results = []
        for offset in range(0, len(samples), self.args.selector_batch_size):
            batch = samples[offset:offset + self.args.selector_batch_size]
            texts = []
            for sample in batch:
                messages = [
                    {"role": "system", "content": "You are a careful emotional-support dialogue analyst."},
                    {"role": "user", "content": build_stage_prompt(format_context(sample["context"]))},
                ]
                texts.append(tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                ))
            encoded = tokenizer(
                texts, padding=True, truncation=True, max_length=1792, return_tensors="pt"
            ).to("cuda")
            with torch.inference_mode():
                outputs = model.generate(
                    **encoded,
                    max_new_tokens=160,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            prompt_length = encoded["input_ids"].shape[1]
            for output in outputs:
                raw = tokenizer.decode(output[prompt_length:], skip_special_tokens=True).strip()
                try:
                    parsed = extract_json(raw)
                    stage = str(parsed["stage"]).strip().lower()
                    ranking = [str(parsed[f"strategy_top_{rank}"]).strip() for rank in (1, 2, 3)]
                    if stage not in STAGES or any(label not in STRATEGIES for label in ranking):
                        raise ValueError("closed-set label violation")
                    error = None
                except Exception as exc:
                    stage = "exploration"
                    ranking = [
                        "Inquire Subjective Information",
                        "Affirmation and Reassurance",
                        "Restatement",
                    ]
                    error = str(exc)
                results.append({"stage": stage, "ranking": ranking, "raw": raw, "error": error})
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        return results

    def _behavior_strategy_distribution(self, behavior_posterior):
        result = {label: 0.0 for label in STRATEGIES}
        for behavior, probability in behavior_posterior.items():
            prior = self.behavior_priors.get(behavior, self.global_prior)
            for label in STRATEGIES:
                result[label] += float(probability) * prior[label]
        return normalize(result)

    def _rank_distribution(self, ranking):
        logits = {label: 0.0 for label in STRATEGIES}
        for label, value in zip(ranking, (3.0, 2.0, 1.0)):
            logits[label] = max(logits[label], value)
        return normalize({
            label: math.exp(value / self.args.rank_temperature) for label, value in logits.items()
        })

    def select_batch(self, samples):
        behavior_posteriors = self._behavior_posteriors(samples)
        stage_outputs = self._stage_rankings(samples)
        selections = []
        for behavior_posterior, stage_output in zip(behavior_posteriors, stage_outputs):
            behavior_distribution = self._behavior_strategy_distribution(behavior_posterior)
            stage_distribution = self._rank_distribution(stage_output["ranking"])
            confidence = normalized_confidence(behavior_distribution)
            alpha = self.args.alpha_low + (self.args.alpha_high - self.args.alpha_low) * confidence
            combined = normalize({
                label: alpha * behavior_distribution[label] + (1.0 - alpha) * stage_distribution[label]
                for label in STRATEGIES
            })
            selections.append({
                "selected_strategy": max(combined, key=combined.get),
                "stage": stage_output["stage"],
                "stage_ranking": stage_output["ranking"],
                "stage_raw": stage_output["raw"],
                "stage_error": stage_output["error"],
                "behavior_prediction": max(behavior_posterior, key=behavior_posterior.get),
                "behavior_posterior": behavior_posterior,
                "behavior_strategy_distribution": behavior_distribution,
                "stage_strategy_distribution": stage_distribution,
                "strategy_prior_confidence": confidence,
                "alpha": alpha,
                "combined_distribution": combined,
            })
        return selections
