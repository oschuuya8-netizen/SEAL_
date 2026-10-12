from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from prompts import build_planner_input


class TextProjectionEncoder(nn.Module):
    def __init__(self, model_name, projection_dim, dropout):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
        hidden_size = self.backbone.config.hidden_size
        self.projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, projection_dim),
            nn.Tanh(),
            nn.Linear(projection_dim, projection_dim),
        )

    def forward(self, input_ids, attention_mask, **_kwargs):
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        return F.normalize(self.projection(pooled), dim=-1)


class DualExecutionModel(nn.Module):
    def __init__(self, model_name, projection_dim, dropout, share_backbone=False):
        super().__init__()
        self.planner = TextProjectionEncoder(model_name, projection_dim, dropout)
        self.execution = self.planner if share_backbone else TextProjectionEncoder(
            model_name, projection_dim, dropout
        )


class ExecutionPlanner:
    def __init__(self, checkpoint_dir, bank_path):
        self.device = torch.device("cuda")
        checkpoint = torch.load(Path(checkpoint_dir) / "pytorch_model.bin", map_location="cpu")
        saved_args = checkpoint["args"]
        self.max_length = int(saved_args.get("max_length", 512))
        self.model = DualExecutionModel(
            saved_args["model_name"],
            int(saved_args["projection_dim"]),
            float(saved_args.get("dropout", 0.1)),
            bool(saved_args.get("share_backbone", False)),
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.tokenizer = AutoTokenizer.from_pretrained(
            saved_args["model_name"], trust_remote_code=True, local_files_only=True
        )
        self.z_dim = int(saved_args["projection_dim"])

        bank = torch.load(bank_path, map_location="cpu")
        self.bank_samples = bank["samples"]
        self.bank_vectors = F.normalize(bank["z_e"].float(), dim=-1)
        self.planner_bank_vectors = F.normalize(bank["z_p"].float(), dim=-1)
        self.strategy_centroids = self._build_strategy_centroids()

    def _build_strategy_centroids(self):
        grouped = {}
        for index, sample in enumerate(self.bank_samples):
            strategy = str(sample.get("strategy") or "").strip()
            if strategy:
                grouped.setdefault(strategy, []).append(index)
        centroids = {}
        for strategy, indices in grouped.items():
            centroid = self.planner_bank_vectors[indices].mean(dim=0, keepdim=True)
            centroids[strategy] = F.normalize(centroid, dim=-1)
        return centroids

    @torch.inference_mode()
    def plan(self, context_text, strategy):
        encoded = self.tokenizer(
            [build_planner_input(context_text, strategy)],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        return self.model.planner(**encoded)

    def static_plan(self, strategy):
        if strategy not in self.strategy_centroids:
            available = ", ".join(sorted(self.strategy_centroids))
            raise ValueError(
                f"No execution-bank centroid for strategy {strategy!r}; available: {available}"
            )
        return self.strategy_centroids[strategy].to(self.device)

    def nearest_execution(self, vector):
        query = F.normalize(vector.detach().cpu().float(), dim=-1)
        scores = (query @ self.bank_vectors.T).squeeze(0)
        index = int(torch.argmax(scores))
        sample = self.bank_samples[index]
        execution = sample.get("execution") or {}
        return {
            "similarity": float(scores[index]),
            "execution": execution,
            "source": {
                "source_file": sample.get("source_file"),
                "round_index": sample.get("round_index"),
                "strategy": sample.get("strategy"),
                "response": sample.get("response"),
            },
        }
