# SEAL

Online response generation and execution planner training code for SEAL.

```text
seal/                    Online inference
  main.py                Entry point for generating final responses
  pipeline.py            Strategy selection, execution planning, and response generation
  strategy_selector.py   Dynamic fusion of behavior priors and ESC stages
  execution.py           Planner loading and execution vector prediction
  generator.py           Execution injection and response generation
  prompts.py             Strategy selection prompts and Planner input formatting
  generation_prompts.py  Response prompts and training target filtering rules
  config.py              Model paths and runtime arguments
  labels.py              Strategy and stage labels
  data.py / utils.py      Data loading and utility functions required for inference
train/
  train_planner.py        Contrastive training of the Planner and execution encoder
  train_projection.py    Two-stage ze/zp joint training of the projector and LoRA
  train_projection_length.py  Continued projector and LoRA training with length preferences
```

## Online Inference

Inference requires a CUDA environment and a trained behavior classifier, Planner,
latent bank, LoRA adapter, and projector. The base models are RoBERTa-large,
BGE-base-en-v1.5, Qwen3-8B, and Qwen2.5-7B-Instruct.
Model weights and real dialogue data are not included in this repository.

Run from the repository root:

```bash
python -m pip install -r requirements.txt
python seal/main.py --data data/test_samples.json --output outputs/responses.json --limit 20 --preserve-order
```

## Planner Training

```bash
python train/train_planner.py --data data/execution_supervised_data.json \
  --output-dir checkpoints/execution --model-name BAAI/bge-base-en-v1.5 \
  --planner-input-mode context_strategy --projection-dim 256 \
  --batch-size 16 --epochs 3 --lr 2e-5 --temperature 0.07 \
  --max-length 512 --val-ratio 0.1 --eval-limit 512 --seed 42
```

Training data is a JSON array. Each sample contains string fields `context` and
`strategy`, plus an `execution` object with four nonempty string-list fields:
`target`, `actions`, `grounding`, and `avoid`. The Planner receives context plus
strategy, while the execution encoder receives the four execution fields.
The two BGE encoders have independent parameters and are jointly trained with
bidirectional InfoNCE. Do not enable `--share-backbone`.
Samples are split into 90% training and 10% validation. The best checkpoint is
selected by validation Top-1 accuracy. Both encoders are frozen during inference.

## Projector Training

The Planner's 256-dimensional projection head is trained by `train/train_planner.py`.
The generator's `z_projector` is a separate module that maps `z_p` into its embedding
space. It must be trained jointly with LoRA during response training; randomly
initialized projector weights should not be used for inference.
Online inference loads the trained `z_projector.pt` without updating its parameters.
`train/train_projection.py` jointly trains the projector and LoRA, saving
`lora_adapter/` and `z_projector.pt`:

```bash
python train/train_projection.py --data data/execution_supervised_data.json \
  --execution-checkpoint checkpoints/execution/best --output-dir checkpoints/generator_ze \
  --stage ze --planner-input-mode context_strategy --epochs 1 --batch-size 1 \
  --grad-accum 8 --lr 2e-4 --max-length 768 --seed 42 --load-in-4bit

```
