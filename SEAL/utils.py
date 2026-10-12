import json
import math
import re
from pathlib import Path


def clean_text(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize(values):
    total = sum(max(0.0, float(value)) for value in values.values())
    if total <= 0:
        return {key: 1.0 / len(values) for key in values}
    return {key: max(0.0, float(value)) / total for key, value in values.items()}


def normalized_confidence(values):
    probabilities = list(normalize(values).values())
    entropy = -sum(p * math.log(max(p, 1e-12)) for p in probabilities)
    return 1.0 - entropy / math.log(len(probabilities))


def format_context(context):
    lines = []
    for turn in context:
        speaker = clean_text(turn.get("speaker") or turn.get("role"))
        utterance = clean_text(turn.get("utterance") or turn.get("content"))
        if speaker and utterance:
            lines.append(f"[{speaker}] {utterance}")
    return "\n".join(lines)


def extract_json(text):
    match = re.search(r"\{.*\}", text, flags=re.S)
    return json.loads(match.group(0) if match else text)
