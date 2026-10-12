#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


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
    sections = []
    sections.append("[TARGET]\n" + "; ".join(clean_items(execution["target"])))
    sections.append("[ACTION]\n" + "; ".join(clean_items(execution["actions"])))
    sections.append("[GROUNDING]\n" + "; ".join(clean_items(execution["grounding"])))
    sections.append("[CONSTRAINT]\n" + "; ".join(clean_items(execution["avoid"])))
    return "\n\n".join(sections)


def format_planner_input(sample, mode="full"):
    context = "[CONTEXT]\n" + str(sample.get("context", "")).strip()
    behavior = "[CLIENT_BEHAVIOR]\n" + str(sample.get("behavior", "")).strip()
    strategy = "[THERAPIST_STRATEGY]\n" + str(sample.get("strategy", "")).strip()
    response = "[THERAPIST_RESPONSE]\n" + str(sample.get("response", "")).strip()
    if mode == "strategy_only":
        return strategy
    if mode == "context_strategy":
        return "\n\n".join([context, strategy])
    if mode == "behavior_strategy":
        return "\n\n".join([behavior, strategy])
    if mode == "context_strategy_response":
        return "\n\n".join([context, strategy, response])
    return "\n\n".join([context, behavior, strategy])


class ExecutionDataset(Dataset):
    def __init__(self, samples, planner_input_mode):
        self.samples = samples
        self.planner_input_mode = planner_input_mode

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        return {
            "planner_text": format_planner_input(sample, self.planner_input_mode),
            "execution_text": format_execution(sample["execution"]),
            "index": index,
        }


def collate_fn(batch):
    return {
        "planner_text": [item["planner_text"] for item in batch],
        "execution_text": [item["execution_text"] for item in batch],
        "index": torch.tensor([item["index"] for item in batch], dtype=torch.long),
    }


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
    def __init__(self, model_name, projection_dim, dropout, share_backbone):
        super().__init__()
        self.planner = TextProjectionEncoder(model_name, projection_dim, dropout)
        if share_backbone:
            self.execution = self.planner
        else:
            self.execution = TextProjectionEncoder(model_name, projection_dim, dropout)

    def forward(self, planner_inputs, execution_inputs):
        z_p = self.planner(**planner_inputs)
        z_e = self.execution(**execution_inputs)
        return z_p, z_e


def contrastive_loss(z_p, z_e, temperature):
    logits = z_p @ z_e.T / temperature
    labels = torch.arange(z_p.size(0), device=z_p.device)
    loss_pe = F.cross_entropy(logits, labels)
    loss_ep = F.cross_entropy(logits.T, labels)
    return (loss_pe + loss_ep) / 2


@torch.no_grad()
def encode_all(model, tokenizer, texts, branch, device, batch_size, max_length):
    model.eval()
    vectors = []
    encoder = model.execution if branch == "execution" else model.planner
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        vectors.append(encoder(**inputs).cpu())
    return torch.cat(vectors, dim=0)


@torch.no_grad()
def retrieval_eval(model, tokenizer, samples, planner_input_mode, device, batch_size, max_length):
    planner_texts = [format_planner_input(sample, planner_input_mode) for sample in samples]
    execution_texts = [format_execution(sample["execution"]) for sample in samples]
    z_p = encode_all(model, tokenizer, planner_texts, "planner", device, batch_size, max_length)
    z_e = encode_all(model, tokenizer, execution_texts, "execution", device, batch_size, max_length)
    sim = z_p @ z_e.T
    ranks = torch.argsort(sim, dim=1, descending=True)
    gold = torch.arange(len(samples)).unsqueeze(1)
    top1 = (ranks[:, :1] == gold).any(dim=1).float().mean().item()
    top5 = (ranks[:, :5] == gold).any(dim=1).float().mean().item()
    mrr = []
    for i in range(len(samples)):
        rank = (ranks[i] == i).nonzero(as_tuple=False)[0].item() + 1
        mrr.append(1.0 / rank)
    return {"top1": top1, "top5": top5, "mrr": sum(mrr) / len(mrr)}


def load_samples(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [item for item in data if valid_execution(item.get("execution"))]


def split_samples(samples, val_ratio, seed):
    indices = list(range(len(samples)))
    random.Random(seed).shuffle(indices)
    val_size = max(1, int(len(samples) * val_ratio))
    val_set = set(indices[:val_size])
    train = [sample for i, sample in enumerate(samples) if i not in val_set]
    val = [sample for i, sample in enumerate(samples) if i in val_set]
    return train, val


def save_checkpoint(path, model, tokenizer, args, metrics, epoch):
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "metrics": metrics,
            "epoch": epoch,
        },
        path / "pytorch_model.bin",
    )
    tokenizer.save_pretrained(path)
    (path / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Train Execution Encoder and Planner with contrastive learning.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--share-backbone", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--eval-limit", type=int, default=512)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--planner-input-mode",
        choices=["full", "strategy_only", "context_strategy", "behavior_strategy", "context_strategy_response"],
        default="full",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(args.data)
    if args.limit:
        samples = samples[:args.limit]
    train_samples, val_samples = split_samples(samples, args.val_ratio, args.seed)
    eval_samples = val_samples[: args.eval_limit]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = DualExecutionModel(args.model_name, args.projection_dim, args.dropout, args.share_backbone)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(
        ExecutionDataset(train_samples, args.planner_input_mode),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * 0.06)),
        num_training_steps=total_steps,
    )

    metadata = {
        "num_loaded_valid_samples": len(samples),
        "num_train": len(train_samples),
        "num_val": len(val_samples),
        "device": str(device),
        "args": vars(args),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)

    best_top1 = -math.inf
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader, start=1):
            planner_inputs = tokenizer(
                batch["planner_text"],
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            ).to(device)
            execution_inputs = tokenizer(
                batch["execution_text"],
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            ).to(device)
            z_p, z_e = model(planner_inputs, execution_inputs)
            loss = contrastive_loss(z_p, z_e, args.temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            running += loss.item()
            global_step += 1
            if step % args.log_every == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "step": step,
                            "global_step": global_step,
                            "loss": running / args.log_every,
                            "lr": scheduler.get_last_lr()[0],
                        }
                    ),
                    flush=True,
                )
                running = 0.0

        metrics = retrieval_eval(
            model,
            tokenizer,
            eval_samples,
            args.planner_input_mode,
            device,
            args.eval_batch_size,
            args.max_length,
        )
        metrics.update({"epoch": epoch, "global_step": global_step})
        print(json.dumps({"eval": metrics}, ensure_ascii=False), flush=True)
        save_checkpoint(output_dir / "last", model, tokenizer, args, metrics, epoch)
        if metrics["top1"] > best_top1:
            best_top1 = metrics["top1"]
            save_checkpoint(output_dir / "best", model, tokenizer, args, metrics, epoch)


if __name__ == "__main__":
    main()
