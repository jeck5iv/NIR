import os
from utils.io import load_config, load_model
from utils.paths import Paths
from scripts.download_model import run as download_model
from scripts.quantize_model import run as quantize_model
from scripts.telegram_parse import run_parser as telegram_parse
from scripts.prepare_data import prepare_data
from scripts.finetune import run as finetune
from scripts.evaluate import evaluate
from scripts.save_load_model import save_model
from scripts.metrics import evaluate_metrics
from scripts.metrics import save_metrics_csv
from scripts.metrics import build_prompt_from_train_data

USE_PROMPT = False

def main():
    base_cfg = load_config("configs/BASE.json")
    base_model_cfg = load_config("configs/models_configs/RefalMachineRuadaptQwen2.5-7B-Lite-Beta.json")
    quant_cfg = None #load_config("configs/quantisation_configs/quant_fp4.json")
    lora_cfg = load_config("configs/lora_configs/lora_optimal.json")
    train_cfg = load_config("configs/train_configs/train_shared.json")
    telegram_cfg = load_config("configs/telegram_config.json")

    model_cfg = {**base_model_cfg, **(quant_cfg or {}), **base_cfg}
    print(model_cfg)
    paths = Paths(model_cfg, telegram_cfg)
    
    os.makedirs(paths.data_dir, exist_ok=True)
    os.makedirs(paths.original_model.parent, exist_ok=True)
    os.makedirs(paths.model_weights_dir, exist_ok=True)


    # Пайплайн
    print("=== Downloading model ===")
    download_model(base_model_cfg)
    
    print("\n=== Quantizing model ===")
    if quant_cfg is None:
        print("=== Empty quant_cfg, skipping quantizing model ===")
    else:
        quantize_model(model_cfg, paths)
        model_cfg["quantized"] = True
    
    # print("\n=== Parsing Telegram ===")
    # telegram_parse(telegram_cfg, paths)
    
    # print("\n=== Preparing data ===")
    # prepare_data(paths)

    print("\n=== Finetuning ===")
    if lora_cfg is None:
        print("lora_cfg is None — Finetuning будет пропущен")
        from utils.io import load_model
        model, tokenizer = load_model(paths.quantized_model if model_cfg.get("quantized") else paths.original_model, model_cfg)
    else:
        print("Train config:", train_cfg)
        print(lora_cfg, "lora_cfg")
        model, tokenizer = finetune(model_cfg, lora_cfg, train_cfg, paths)

    print("\n=== Saving model ===")
    model_name = save_model(model, tokenizer, model_cfg, lora_cfg or {}, paths, train_cfg)

    print("\n=== Evaluating ===")
    evaluate(model_cfg, paths)

    print("\n=== Evaluating metrics ===")
    prompt = build_prompt_from_train_data(paths.train_data) if USE_PROMPT else None
    metrics = evaluate_metrics(model_cfg, paths, model_name, prompt=prompt)
    save_metrics_csv(metrics, model_cfg, lora_cfg or {}, paths, with_prompt=USE_PROMPT)


if __name__ == "__main__":
    main()