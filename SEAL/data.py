import random
import re

from utils import clean_text, load_json


def word_count(text):
    return len(re.findall(r"[A-Za-z0-9']+", clean_text(text)))


def normalize_sample(item, index):
    return {
        "id": item.get("id", f"sample_{index}"),
        "source_index": item.get("source_index", index),
        "context": item.get("context") or item.get("dialogue") or [],
        "reference_strategy": (
            item.get("reference_strategy") or item.get("strategy") or item.get("label")
        ),
        "reference_response": (
            item.get("reference_response") or item.get("response") or item.get("target") or ""
        ),
    }


def load_samples(path, limit, seed, min_response_words=0, preserve_order=False):
    raw = load_json(path)
    if isinstance(raw, dict):
        raw = raw.get("samples") or raw.get("data") or raw.get("results") or []
    samples = []
    for index, item in enumerate(raw):
        sample = normalize_sample(item, index)
        if not sample["context"]:
            continue
        if sample["reference_strategy"] == "Other":
            continue
        if min_response_words and word_count(sample["reference_response"]) < min_response_words:
            continue
        samples.append(sample)
    if not preserve_order:
        random.Random(seed).shuffle(samples)
    return samples[:limit]
