from datasets import Dataset
from transformers import TrainingArguments, Trainer, default_data_collator
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from utils.io import load_model, load_tokenizer
import pandas as pd
import torch

def run(model_cfg, lora_cfg, train_cfg, paths):
    print("Train config:", train_cfg)
    print("=== Загружаем модель и токенизатор ===")
    if model_cfg.get("quantized", False):
        model, tokenizer = load_model(paths.quantized_model, model_cfg)
    else:
        model, tokenizer = load_model(paths.original_model, model_cfg)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch):
        result = tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        result["labels"] = result["input_ids"].clone()
        result = {k: v.squeeze(0) if v.dim() == 2 else v for k, v in result.items()}
        return result

    print("=== Настраиваем LoRA ===")
    if model_cfg.get("quantized", False):
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg["task_type"]
    )
    model = get_peft_model(model, lora_config)

    model.train()
    print("Model is in training mode:", model.training)

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, 'config'):
        model.config.use_cache = False

    print("=== Загружаем обучающую выборку ===")
    train_df = pd.read_csv(paths.train_data)
    train_texts = train_df['cleaned_text'].fillna('').tolist()
    train_dataset = Dataset.from_dict({'text': train_texts})
    train_dataset = train_dataset.map(tokenize, batched=True)

    print("=== Загружаем валидационную выборку ===")
    val_df = pd.read_csv(paths.val_data)
    val_texts = val_df['cleaned_text'].fillna('').tolist()
    eval_dataset = Dataset.from_dict({'text': val_texts})
    eval_dataset = eval_dataset.map(tokenize, batched=True)

    print("=== Запускаем обучение ===")
    training_args = TrainingArguments(
        output_dir=paths.resolve(train_cfg["output_dir"]),
        per_device_train_batch_size=train_cfg["batch_size"],
        num_train_epochs=train_cfg["num_train_epochs"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        fp16=train_cfg["fp16"],
        learning_rate=train_cfg["learning_rate"],
        logging_steps=train_cfg["logging_steps"],
        eval_strategy=train_cfg["eval_strategy"],
        save_strategy=train_cfg["save_strategy"],
        logging_dir=paths.resolve(train_cfg["logging_dir"]),
        save_total_limit=train_cfg["save_total_limit"],
        report_to="none"
    )

    print("=== Проверка backward на одном примере ===")
    model.train()
    sample = tokenizer("пример текста для обучения", return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    sample["labels"] = sample["input_ids"].clone()
    sample = {k: v.to(model.device) for k, v in sample.items()}

    print("Model device:", model.device)
    print("Input IDs device:", sample["input_ids"].device)
    print("Labels device:", sample["labels"].device)
    print("Input IDs requires_grad:", sample["input_ids"].requires_grad)
    print("Labels requires_grad:", sample["labels"].requires_grad)

    output = model(**sample)
    loss = output.loss

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator
    )

    torch.cuda.empty_cache()
    trainer.train()
    
    return model, tokenizer