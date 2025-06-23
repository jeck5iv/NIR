from pathlib import Path
from datetime import datetime
import evaluate
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import Counter
from tqdm import tqdm
import math
import json
import csv
import os

from utils.paths import Paths
from scripts.save_load_model import load_model_adapted_with_lora
from utils.gpt_wrapper import GPTGenerator
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()


def calculate_bleu(predictions, references):
    metric = evaluate.load("bleu")
    return metric.compute(predictions=predictions, references=[[r] for r in references])["bleu"]


def calculate_rouge(predictions, references):
    metric = evaluate.load("rouge")
    return metric.compute(predictions=predictions, references=references)


def calculate_bertscore(predictions, references):
    metric = evaluate.load("bertscore")
    return metric.compute(predictions=predictions, references=references, lang="en")


def calculate_perplexity(model, tokenizer, texts):
    model.eval()
    losses = []
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            output = model(**enc, labels=enc["input_ids"])
        losses.append(output.loss.item())
    return math.exp(sum(losses) / len(losses))


def token_repetition_ratio(text):
    tokens = text.split()
    counter = Counter(tokens)
    repeated = sum(v > 1 for v in counter.values())
    return repeated / len(counter) if counter else 0.0


def build_prompt_from_train_data(train_path: Path, num_messages: int = 5) -> str:
    df = pd.read_csv(train_path)
    texts = df["cleaned_text"].dropna().sample(n=num_messages, random_state=42).tolist()
    intro = (
        "Сейчас я передам тебе несколько сообщений из телеграм-канала. "
        "Я хочу, чтобы ты дальше продолжала писать посты в таком же стиле.\n\n"
    )
    return intro + "\n\n".join(texts) + "\n\nОсновываясь на стиле предыдущих сообщений из телеграм-канала, закончи сообщение:"


def evaluate_metrics(model_cfg: dict, paths: Paths, model_name: str, num_samples: int = 100,
                     prompt: str | None = None, use_gpt_api: bool = False):

    if use_gpt_api:
        config_path = paths.resolve("configs/gpt_api_config.json")
        with open(config_path, "r") as f:
            gpt_cfg = json.load(f)
        api_key = gpt_cfg["api_key"]
        gpt = GPTGenerator(api_key=api_key)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = None
    else:
        try:
            model, tokenizer = load_model_adapted_with_lora(model_cfg, paths)
        except FileNotFoundError:
            from utils.io import load_model
            model, tokenizer = load_model(paths.original_model, model_cfg)

    if model:
        model.eval()

    data = pd.read_csv(paths.test_data)
    sample_texts = data["cleaned_text"].dropna().tolist()[:num_samples]

    prompts, targets, generations = [], [], []

    for text in tqdm(sample_texts):
        tokens = tokenizer.tokenize(text)
        if len(tokens) < 5:
            continue
        cutoff = int(len(tokens) * 0.8)
        prompt_tokens = tokens[:cutoff]
        target_tokens = tokens[cutoff:]
        prompt_text = tokenizer.convert_tokens_to_string(prompt_tokens)
        target_text = tokenizer.convert_tokens_to_string(target_tokens)
        full_prompt = (prompt or '') + "\n\n" + prompt_text

        if use_gpt_api:
            generated_text = gpt.generate(full_prompt, max_tokens=len(target_tokens))
        else:
            inputs = tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=tokenizer.model_max_length
            ).to(model.device)

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=len(target_tokens),
                    pad_token_id=tokenizer.eos_token_id
                )
            decoded = tokenizer.decode(output[0], skip_special_tokens=True)
            generated_text = decoded[len(full_prompt):].strip()

        prompts.append(prompt_text)
        targets.append(target_text)
        generations.append(generated_text)

    filtered = [(gen, tgt) for gen, tgt in zip(generations, targets) if tgt.strip() and gen.strip()]
    generations, targets = zip(*filtered) if filtered else ([], [])

    if not generations:
        return {}

    bleu = calculate_bleu(generations, targets)
    rouge = calculate_rouge(generations, targets)
    bertscore = calculate_bertscore(generations, targets)
    perplexity = None if use_gpt_api else calculate_perplexity(model, tokenizer, sample_texts)
    repetition = sum(token_repetition_ratio(g) for g in generations) / len(generations)

    return {
        "bleu": bleu,
        "rougeL": rouge["rougeL"],
        "bertscore": sum(bertscore["f1"]) / len(bertscore["f1"]),
        "perplexity": perplexity if perplexity is not None else 0.0,
        "repetition": repetition,
        "source": "gpt" if use_gpt_api else "local"
    }


def save_metrics_csv(metrics: dict, model_cfg: dict, lora_cfg: dict, paths, with_prompt: bool):
    csv_path = paths.resolve("metrics_log.csv")
    is_new_file = not os.path.exists(csv_path)

    model_name = model_cfg["name"]
    quant_name = model_cfg.get("quant_name") if model_cfg else "None"
    lora_name = lora_cfg.get("name") if lora_cfg else "None"
    source = metrics.get("source", "local")

    with open(csv_path, mode="a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new_file:
            writer.writerow([
                "timestamp",
                "model_name",
                "quant_cfg_name",
                "lora_cfg_name",
                "bleu",
                "rougeL",
                "bertscore",
                "perplexity",
                "repetition",
                "with_prompt",
                "source"
            ])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model_name,
            quant_name,
            lora_name,
            f"{metrics['bleu']:.4f}",
            f"{metrics['rougeL']:.4f}",
            f"{metrics['bertscore']:.4f}",
            f"{metrics['perplexity']:.2f}",
            f"{metrics['repetition']:.4f}",
            "yes" if with_prompt else "no",
            source
        ])
