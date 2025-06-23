import os
import datetime
import shutil
import json
from peft import PeftModel
from utils.io import load_model, load_tokenizer


def generate_model_name(model_cfg, lora_cfg):
    base_name = model_cfg["name"]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    lora_name = (lora_cfg or {}).get("r", "nolora")
    quant_name = model_cfg.get("quant_name", "fp16")
    return f"{base_name}_{quant_name}_lora{lora_name}_{timestamp}"


def save_model(model, tokenizer, model_cfg, lora_cfg, paths, train_cfg):
    name = generate_model_name(model_cfg, lora_cfg)
    model_cfg["full_name"] = name
    target_dir = os.path.join(paths.resolve("model_weights_dir"), name)
    os.makedirs(target_dir, exist_ok=True)

    output_dir = paths.resolve(train_cfg["output_dir"])

    if isinstance(model, PeftModel):
        checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
        if not checkpoints:
            raise RuntimeError("Не найдено checkpoint'ов для сохранения.")
        last_ckpt = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
        src_dir = os.path.join(output_dir, last_ckpt)

        for filename in os.listdir(src_dir):
            shutil.copy2(os.path.join(src_dir, filename), os.path.join(target_dir, filename))
    else:
        model.save_pretrained(target_dir)

    tokenizer.save_pretrained(target_dir)
    return name


def load_model_adapted_with_lora(model_cfg, paths):
    name = model_cfg["full_name"]
    adapter_path = os.path.join(paths.resolve("model_weights_dir"), name)
    adapter_config_path = os.path.join(adapter_path, "adapter_config.json")

    with open(adapter_config_path, "r") as f:
        adapter_config = json.load(f)
    base_model_path = adapter_config["base_model_name_or_path"]

    base_model, tokenizer = load_model(base_model_path, model_cfg)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer
