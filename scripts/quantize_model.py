from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from pathlib import Path
import torch
import gc

def run(model_config, paths):
    """Квантование модели с использованием конфигурации"""
    print("\n=== Quantizing model ===")
    
    original_dir = paths.original_model
    quant_dir = paths.quantized_model

    print("Clearing memory...")
    torch.cuda.empty_cache()
    gc.collect()

    print("Creating quantization config...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=model_config.get("load_in_4bit", True),
        bnb_4bit_use_double_quant=model_config.get("bnb_4bit_use_double_quant", True),
        bnb_4bit_quant_type=model_config.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=getattr(torch, model_config.get("bnb_4bit_compute_dtype", "float16")),
        llm_int8_enable_fp32_cpu_offload=model_config.get("llm_int8_enable_fp32_cpu_offload", False)
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(original_dir)

    try:
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            original_dir,
            quantization_config=bnb_config,
            device_map="auto",
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

    print(f"Saving quantized model to {quant_dir}...")
    quant_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(quant_dir)
    model.save_pretrained(quant_dir)

    print("Quantization complete")

if __name__ == "__main__":
    from utils.io import load_config
    from utils.paths import Paths
    config = load_config("configs/base.json")
    run(config["model"], Paths(config))