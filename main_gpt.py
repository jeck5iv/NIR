import os
import sys
from pathlib import Path
from utils.io import load_config
from utils.paths import Paths
from scripts.telegram_parse import run_parser as telegram_parse
from scripts.prepare_data import prepare_data
from scripts.metrics import evaluate_metrics, save_metrics_csv, build_prompt_from_train_data

USE_PROMPT = True
NUM_SAMPLES = 100

def main():
    base_model_cfg = load_config("configs/models_configs/gpt-3.5-turbo.json")
    telegram_cfg = load_config("configs/telegram_config.json")

    model_cfg = base_model_cfg
    paths = Paths(model_cfg, telegram_cfg)

    os.makedirs(paths.data_dir, exist_ok=True)

    # print("=== Parsing Telegram ===")
    # telegram_parse(telegram_cfg, paths)

    # print("=== Preparing data ===")
    # prepare_data(paths)

    print("\n=== Evaluating metrics with GPT ===")
    prompt = build_prompt_from_train_data(paths.train_data) if USE_PROMPT else None
    metrics = evaluate_metrics(model_cfg, paths, model_name=model_cfg["name"], prompt=prompt, use_gpt_api=True)
    save_metrics_csv(metrics, model_cfg, lora_cfg={}, paths=paths, with_prompt=USE_PROMPT)

if __name__ == "__main__":
    main()
