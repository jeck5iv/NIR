import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch


def load_config(config_path: str):
    with open(config_path, "r") as f:
        return json.load(f)


def load_tokenizer(model_dir: str):
    return AutoTokenizer.from_pretrained(model_dir)


def load_model(model_dir: str, model_config: dict = None):
    if model_config.get("quantized", False):
        model_dir = model_config["quantized_model"]
    else:
        model_dir = model_config["original_model"]
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    if model_config.get("quantized", False):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=model_config.get("load_in_4bit", True),
            bnb_4bit_use_double_quant=model_config.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_quant_type=model_config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=getattr(torch, model_config.get("bnb_4bit_compute_dtype", "float16")),
            llm_int8_enable_fp32_cpu_offload=model_config.get("llm_int8_enable_fp32_cpu_offload", False),
            bnb_4bit_group_size=model_config.get("bnb_4bit_group_size", 128),
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=None,
        )

    return model, tokenizer