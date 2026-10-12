import torch
import torch.nn as nn
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from generation_prompts import build_generator_prompt
from prompts import EXEC_TOKEN


class ExecutionConditionedGenerator:
    def __init__(self, args, execution_dim, use_execution=True):
        self.args = args
        self.use_execution = use_execution
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.base_model, trust_remote_code=True, local_files_only=True
        )
        self.tokenizer.add_special_tokens({"additional_special_tokens": [EXEC_TOKEN]})
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            local_files_only=True,
            dtype=torch.bfloat16,
            quantization_config=quantization,
            device_map="auto",
            trust_remote_code=True,
        )
        base.resize_token_embeddings(len(self.tokenizer))
        self.model = PeftModel.from_pretrained(base, args.lora_adapter).eval()
        self.embedding_device = self.model.get_input_embeddings().weight.device

        self.projector = None
        if self.use_execution:
            hidden_size = self.model.config.hidden_size
            self.projector = nn.Sequential(
                nn.Linear(execution_dim, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
            )
            self.projector.load_state_dict(torch.load(args.projector_path, map_location="cpu"))
            self.projector.to(self.embedding_device).eval()
            for parameter in self.projector.parameters():
                parameter.requires_grad = False
        self.exec_token_id = self.tokenizer.convert_tokens_to_ids(EXEC_TOKEN)

    @torch.inference_mode()
    def _generate_ids(self, input_ids, attention_mask, embeddings):
        kwargs = {
            "attention_mask": attention_mask,
            "max_new_tokens": self.args.max_new_tokens,
            "do_sample": self.args.do_sample,
            "num_return_sequences": 1,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.args.min_new_tokens > 0:
            kwargs["min_new_tokens"] = self.args.min_new_tokens
        if self.args.repetition_penalty != 1.0:
            kwargs["repetition_penalty"] = self.args.repetition_penalty
        if self.args.no_repeat_ngram_size > 0:
            kwargs["no_repeat_ngram_size"] = self.args.no_repeat_ngram_size
        if self.args.do_sample:
            kwargs.update(
                temperature=self.args.temperature,
                top_p=self.args.top_p,
            )
        output = self.model.generate(
            input_ids=input_ids,
            inputs_embeds=embeddings,
            **kwargs,
        )
        return output[:, input_ids.shape[-1]:]

    @torch.inference_mode()
    def generate(self, context_text, strategy, execution_vector, sample_seed):
        torch.manual_seed(sample_seed)
        torch.cuda.manual_seed_all(sample_seed)
        prompt = build_generator_prompt(
            context_text,
            strategy,
            self.args.response_min_words,
            self.args.response_max_words,
            include_execution=self.use_execution,
        )
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.embedding_device)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        embeddings = self.model.get_input_embeddings()(input_ids).clone()
        if self.use_execution:
            if execution_vector is None:
                raise ValueError("execution vector is required when execution injection is enabled")
            positions = input_ids.eq(self.exec_token_id).nonzero(as_tuple=True)
            if not positions[0].numel():
                raise ValueError("execution token missing from generator prompt")
            projected = self.projector(execution_vector.to(self.embedding_device)).to(embeddings.dtype)
            embeddings[positions] = projected[positions[0]]

        generated = self._generate_ids(input_ids, attention_mask, embeddings)
        text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        text = text.split("<|im_end|>")[0].strip()
        return {"response": text, "generation_failed": not bool(text)}
