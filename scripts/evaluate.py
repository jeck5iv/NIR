from utils.paths import Paths
from scripts.save_load_model import load_model_adapted_with_lora
import torch
import pandas as pd


def evaluate(model_cfg: dict, paths: Paths, num_samples: int = 10):
    """Запуск оценки модели"""
    try:
        model, tokenizer = load_model_adapted_with_lora(model_cfg, paths)
    except FileNotFoundError:
        print("LoRA adapter not found. Loading base model only.")
        from utils.io import load_model
        model, tokenizer = load_model(paths.original_model, model_cfg)

    model.eval()

    data = pd.read_csv(paths.test_data)
    sample_texts = data['cleaned_text'].dropna().tolist()[:num_samples]

    print(f"\nEvaluating on {len(sample_texts)} samples...")
    with torch.no_grad():
        for i, text in enumerate(sample_texts):
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(model.device)

            output = model.generate(
                **inputs,
                max_new_tokens=50,
                pad_token_id=tokenizer.eos_token_id
            )
            decoded = tokenizer.decode(
                output[0],
                skip_special_tokens=True
            )
            print(f"=== Sample {i} ===\n{decoded}\n")


if __name__ == "__main__":
    from utils.io import load_config
    from utils.paths import Paths

    config = load_config("configs/base.json")
    model_cfg = config["model"]
    paths = Paths(config)

    model_name = "checkpoint-834"
    evaluate(model_cfg, paths, model_name)
