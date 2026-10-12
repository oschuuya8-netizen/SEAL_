#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "seal"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_linear_schedule_with_warmup

from generation_prompts import build_generator_prompt, target_is_safe, word_count
from prompts import EXEC_TOKEN

EXECUTION_FIELDS = ["target", "actions", "grounding", "avoid"]


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_items(items):
    if isinstance(items, str):
        items = [items]
    if not isinstance(items, list):
        return []
    return [str(item).strip() for item in items if str(item).strip()]


def valid_execution(execution):
    if not isinstance(execution, dict):
        return False
    return all(clean_items(execution.get(field)) for field in EXECUTION_FIELDS)


def format_execution(execution):
    return "\n\n".join(
        [
            "[TARGET]\n" + "; ".join(clean_items(execution["target"])),
            "[ACTION]\n" + "; ".join(clean_items(execution["actions"])),
            "[GROUNDING]\n" + "; ".join(clean_items(execution["grounding"])),
            "[CONSTRAINT]\n" + "; ".join(clean_items(execution["avoid"])),
        ]
    )


def format_planner_input(sample, mode="context_strategy"):
    context = "[CONTEXT]\n" + str(sample.get("context", "")).strip()
    behavior = "[CLIENT_BEHAVIOR]\n" + str(sample.get("behavior", "")).strip()
    strategy = "[THERAPIST_STRATEGY]\n" + str(sample.get("strategy", "")).strip()
    if mode == "full":
        return "\n\n".join([context, behavior, strategy])
    return "\n\n".join([context, strategy])


def build_prompt(sample, min_words=16, max_words=25):
    return build_generator_prompt(
        str(sample.get("context", "")).strip(),
        str(sample.get("strategy", "")).strip(),
        min_words,
        max_words,
    )


class TextProjectionEncoder(nn.Module):
    def __init__(self, model_name, projection_dim, dropout):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        hidden_size = self.backbone.config.hidden_size
        self.projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, projection_dim),
            nn.Tanh(),
            nn.Linear(projection_dim, projection_dim),
        )

    def mean_pool(self, outputs, attention_mask):
        token_embeddings = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
        summed = (token_embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-6)
        return summed / counts

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.mean_pool(outputs, attention_mask)
        return F.normalize(self.projection(pooled), dim=-1)


class DualExecutionModel(nn.Module):
    def __init__(self, model_name, projection_dim, dropout, share_backbone=False):
        super().__init__()
        self.planner = TextProjectionEncoder(model_name, projection_dim, dropout)
        self.execution = self.planner if share_backbone else TextProjectionEncoder(model_name, projection_dim, dropout)


def load_execution_model(checkpoint_dir, device):
    checkpoint_path = Path(checkpoint_dir) / "pytorch_model.bin"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    ckpt_args = checkpoint["args"]
    model = DualExecutionModel(
        ckpt_args["model_name"],
        ckpt_args["projection_dim"],
        ckpt_args.get("dropout", 0.1),
        ckpt_args.get("share_backbone", False),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    tokenizer = AutoTokenizer.from_pretrained(ckpt_args["model_name"], trust_remote_code=True)
    return model, tokenizer, ckpt_args


class SFTDataset(Dataset):
    def __init__(self, samples, latents=None):
        self.samples = samples
        self.latents = latents

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = {"sample": self.samples[index]}
        if self.latents is not None:
            item["z"] = self.latents[index]
        return item


def collate_fn(batch, tokenizer, max_length, min_words=16, max_words=25):
    input_ids = []
    attention_masks = []
    labels = []
    samples = [item["sample"] for item in batch]
    for sample in samples:
        prompt = build_prompt(sample, min_words, max_words)
        response = str(sample.get("response", "")).strip() + "<|im_end|>"
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
        ids = prompt_ids + response_ids
        label = [-100] * len(prompt_ids) + response_ids
        if len(ids) > max_length:
            ids = ids[-max_length:]
            label = label[-max_length:]
        input_ids.append(torch.tensor(ids, dtype=torch.long))
        attention_masks.append(torch.ones(len(ids), dtype=torch.long))
        labels.append(torch.tensor(label, dtype=torch.long))

    input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_masks = nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    result = {"input_ids": input_ids, "attention_mask": attention_masks, "labels": labels, "samples": samples}
    if "z" in batch[0]:
        result["z"] = torch.stack([item["z"] for item in batch])
    return result


@torch.no_grad()
def compute_z(batch_samples, stage, execution_model, execution_tokenizer, device, max_length, planner_input_mode):
    if stage == "ze":
        texts = [format_execution(sample["execution"]) for sample in batch_samples]
        encoder = execution_model.execution
    elif stage == "zp":
        texts = [format_planner_input(sample, planner_input_mode) for sample in batch_samples]
        encoder = execution_model.planner
    else:
        raise ValueError(f"unknown stage: {stage}")
    inputs = execution_tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)
    return encoder(**inputs)


class ExecutionConditionedLM(nn.Module):
    def __init__(self, base_model, z_dim, exec_token_id, eos_penalty_weight):
        super().__init__()
        self.base_model = base_model
        hidden_size = base_model.config.hidden_size
        self.z_projector = nn.Sequential(
            nn.Linear(z_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.exec_token_id = exec_token_id
        self.eos_penalty_weight = eos_penalty_weight

    def forward(self, input_ids, attention_mask, labels, z):
        inputs_embeds = self.base_model.get_input_embeddings()(input_ids)
        exec_mask = input_ids.eq(self.exec_token_id)
        projected = self.z_projector(z).to(inputs_embeds.dtype)
        if not exec_mask.any(dim=1).all():
            missing = (~exec_mask.any(dim=1)).nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(f"missing execution token in rows: {missing}")
        row_idx, token_idx = exec_mask.nonzero(as_tuple=True)
        inputs_embeds[row_idx, token_idx] = projected[row_idx]
        outputs = self.base_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        eos_penalty = outputs.loss.new_zeros(())
        if self.eos_penalty_weight > 0:
            shifted_labels = labels[:, 1:]
            active = shifted_labels.ne(-100) & shifted_labels.ne(self.base_model.config.eos_token_id)
            if active.any():
                active_logits = outputs.logits[:, :-1][active].float()
                eos_probabilities = active_logits.softmax(dim=-1)[
                    :, self.base_model.config.eos_token_id
                ]
                token_penalties = -torch.log1p(-eos_probabilities.clamp(max=1.0 - 1e-6))
                eos_penalty = token_penalties.mean().to(outputs.loss.dtype)
                outputs.loss = outputs.loss + self.eos_penalty_weight * eos_penalty
        return outputs, eos_penalty


def load_samples(path, limit=None):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    samples = [item for item in data if valid_execution(item.get("execution")) and str(item.get("response", "")).strip()]
    return samples[:limit] if limit else samples


def save_model(output_dir, model, tokenizer, execution_checkpoint, args, step):
    output_dir.mkdir(parents=True, exist_ok=True)
    model.base_model.save_pretrained(output_dir / "lora_adapter")
    tokenizer.save_pretrained(output_dir / "lora_adapter")
    torch.save(model.z_projector.state_dict(), output_dir / "z_projector.pt")
    metadata = {
        "step": step,
        "execution_checkpoint": execution_checkpoint,
        "args": vars(args),
    }
    (output_dir / "sft_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Execution-conditioned SFT for Qwen.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--execution-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--stage", choices=["ze", "zp"], required=True)
    parser.add_argument("--latent-cache")
    parser.add_argument("--resume-lora")
    parser.add_argument("--resume-projector")
    parser.add_argument("--planner-input-mode", choices=["context_strategy", "full"], default="context_strategy")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--eos-penalty-weight", type=float, default=0.05)
    parser.add_argument("--min-target-words", type=int, default=16)
    parser.add_argument("--max-target-words", type=int, default=25)
    parser.add_argument("--drop-unsafe-targets", action="store_true")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--z-max-length", type=int, default=512)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--max-optimizer-steps", type=int)
    parser.add_argument("--skip-final-save", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    latent_cache = None
    if args.latent_cache:
        latent_cache = torch.load(args.latent_cache, map_location="cpu")
        samples = latent_cache["samples"]
        z_dim = latent_cache["z_e"].shape[-1]
        execution_model = None
        execution_tokenizer = None
        execution_args = {"projection_dim": z_dim}
    else:
        execution_model, execution_tokenizer, execution_args = load_execution_model(args.execution_checkpoint, device)
        z_dim = execution_args["projection_dim"]

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.add_special_tokens({"additional_special_tokens": [EXEC_TOKEN]})
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    exec_token_id = tokenizer.convert_tokens_to_ids(EXEC_TOKEN)

    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    lm = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        quantization_config=quantization_config,
        trust_remote_code=True,
    )
    lm.resize_token_embeddings(len(tokenizer))
    lm.config.use_cache = False
    if args.load_in_4bit:
        lm = prepare_model_for_kbit_training(lm, use_gradient_checkpointing=args.gradient_checkpointing)
    if args.gradient_checkpointing and hasattr(lm, "gradient_checkpointing_enable"):
        lm.gradient_checkpointing_enable()

    if args.resume_lora:
        lm = PeftModel.from_pretrained(lm, args.resume_lora, is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        lm = get_peft_model(lm, lora_config)

    model = ExecutionConditionedLM(lm, z_dim, exec_token_id, args.eos_penalty_weight).to(device)
    if args.resume_projector:
        model.z_projector.load_state_dict(torch.load(args.resume_projector, map_location="cpu"))
    model.train()

    if latent_cache is None:
        samples = load_samples(args.data, args.limit)
        latents = None
    else:
        latents = latent_cache["z_e"] if args.stage == "ze" else latent_cache["z_p"]
        if args.limit:
            samples = samples[:args.limit]
            latents = latents[:args.limit]
    before_filter = len(samples)
    keep_indices = []
    for index, sample in enumerate(samples):
        response = str(sample.get("response", "")).strip()
        count = word_count(response)
        if not args.min_target_words <= count <= args.max_target_words:
            continue
        if args.drop_unsafe_targets and not target_is_safe(response):
            continue
        keep_indices.append(index)
    samples = [samples[index] for index in keep_indices]
    if latents is not None:
        latents = latents[keep_indices]
    if not samples:
        raise ValueError("no samples remain after length and safety filtering")
    combined = list(zip(samples, latents)) if latents is not None else [(sample, None) for sample in samples]
    random.Random(args.seed).shuffle(combined)
    samples = [sample for sample, _ in combined]
    latents = torch.stack([z for _, z in combined]) if latents is not None else None
    loader = DataLoader(
        SFTDataset(samples, latents),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(
            batch,
            tokenizer,
            args.max_length,
            args.min_target_words,
            args.max_target_words,
        ),
    )

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    update_steps = max(1, (len(loader) * args.epochs) // args.grad_accum)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(update_steps * 0.03)),
        num_training_steps=update_steps,
    )

    run_meta = {
        "num_samples": len(samples),
        "stage": args.stage,
        "device": str(device),
        "z_dim": z_dim,
        "exec_token_id": exec_token_id,
        "num_samples_before_filter": before_filter,
        "num_samples_after_filter": len(samples),
        "target_word_range": [args.min_target_words, args.max_target_words],
        "args": vars(args),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(run_meta, ensure_ascii=False, indent=2), flush=True)

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    stop_training = False
    for epoch in range(1, args.epochs + 1):
        running = 0.0
        running_eos = 0.0
        for step, batch in enumerate(loader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            z = compute_z(
                batch["samples"],
                args.stage,
                execution_model,
                execution_tokenizer,
                device,
                args.z_max_length,
                args.planner_input_mode,
            ) if "z" not in batch else batch["z"].to(device)
            outputs, eos_penalty = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                z=z,
            )
            loss = outputs.loss / args.grad_accum
            loss.backward()
            running += loss.item() * args.grad_accum
            running_eos += eos_penalty.detach().float().item()

            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % args.log_every == 0:
                    print(
                        json.dumps(
                            {
                                "epoch": epoch,
                                "loader_step": step,
                                "global_step": global_step,
                                "loss": running / (args.log_every * args.grad_accum),
                                "eos_penalty": running_eos / (args.log_every * args.grad_accum),
                                "lr": scheduler.get_last_lr()[0],
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    running = 0.0
                    running_eos = 0.0
                if args.save_every and global_step % args.save_every == 0:
                    save_model(output_dir / f"checkpoint-{global_step}", model, tokenizer, args.execution_checkpoint, args, global_step)
                if args.max_optimizer_steps and global_step >= args.max_optimizer_steps:
                    stop_training = True
                    break

        remainder = len(loader) % args.grad_accum
        if remainder and not stop_training:
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
        if not args.skip_final_save:
            save_model(output_dir / "last", model, tokenizer, args.execution_checkpoint, args, global_step)
            print(json.dumps({"epoch": epoch, "global_step": global_step, "saved": str(output_dir / "last")}), flush=True)
        if stop_training:
            break


if __name__ == "__main__":
    main()
