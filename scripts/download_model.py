from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import torch

def run(model_config):
    print(model_config)
    model_name = model_config.get("name")
    save_dir = Path(model_config.get("original_model"))

    if not model_name:
        raise ValueError("Missing 'name' in model_config")

    if save_dir.exists():
        print(f"✅ Модель уже существует в {save_dir}")
        return

    print(f"⬇️ Скачиваем {model_name} в {save_dir}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16
        )

        save_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(save_dir)
        model.save_pretrained(save_dir)

        print("✅ Скачивание завершено.")
    except Exception as e:
        print(f"❌ Ошибка при скачивании модели: {e}")
        raise
