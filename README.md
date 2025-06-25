
# LoRA Fine-Tuning of Mistral-7B and Qwen2.5-7B for Text Generation in a Personal Communication Style

Проект исследует возможность дообучения LLM (например, Mistral 7B) под стиль конкретного человека на основе данных из Telegram канала. Для адаптации используется LoRA (Low-Rank Adaptation), позволяющая эффективно дообучать модель даже при ограниченном объёме данных.

Этот репозиторий - пайплайн, позволяющий для выбранного канала, выбранной модели, выбранного конфига квантизации и выбранного LoRa-конфига,
распарсить данные из тг канала, сквантовать и затюнить модель, а затем измерить на ней метрики.

Также тут есть модифицированный пайплайн для замера метрик на моделях через OpenAI API (например gpt3.5-turbo), для последующего сравнения результатов с локально зафайнтюнеными моделями.

---

## Установка

```bash
git clone git@github.com:jeck5iv/NIR.git
cd NIR
pip install -r requirements.txt
````

---

## Структура проекта

```
configs/
├── base.json
├── telegram_config.json
├── gpt_api_config.json
├── models_configs/
├── train_configs/
├── lora_configs/
├── quantisation_configs/

scripts/ и utils/                  # основной код всего пайплайна
data/                              # собранные и подготовленные данные в результате парсинга
models/                            # скаченные локально модели и квантизации
checkpoints/ и model_weights_dir   # обученные веса
metrics_log.csv                    # собранные метрики
```

---

## Конфигурация запуска

Все настройки задаются через конфиги в `configs/` которые передаются в `main.py`. Вы можете варьировать:

### 1. Выбор модели

В `configs/models_configs/` находятся описания моделей. Примеры:

* `Mistral-7B-v0.1.json`
* `Qwen2.5-1.5B-Instruct.json`
* `RefalMachineRuadaptQwen2.5-7B-Lite-Beta.json`
* `gpt-3.5-turbo.json` (для сравнения через API)

```json
{
  "name": "mistralai/Mistral-7B-v0.1",
  "original_model": "models/Mistral-7B-v0.1",
  "train_data": "data/train_data.csv",
  "val_data": "data/val_data.csv",
  "test_data": "data/test_data.csv",
  "model_weights_dir": "checkpoints"
}
```
```python
base_model_cfg = load_config("configs/models_configs/Mistral-7B-v0.1.json")
```

### 2. Квантование (опционально)

Можно указать конфиг квантования из `configs/quantisation_configs/`, например:

```json
{
  "quant_name": "quant_fp4",
  "quantized": true,
  "base_model_id": "mistralai/Mistral-7B-v0.1",
  "original_model": "models/Mistral-7B-v0.1",
  "quantized_model": "./models/mistral-7b-fp4",
  "load_in_4bit": true,
  "bnb_4bit_use_double_quant": false,
  "bnb_4bit_quant_type": "fp4",
  "bnb_4bit_compute_dtype": "bfloat16",
  "bnb_4bit_group_size": 32,
  "llm_int8_enable_fp32_cpu_offload": false
}
```

```python
quant_cfg = load_config("configs/quantisation_configs/quant_fp4.json")
```

Или отключить:

```python
quant_cfg = None
```

### 3. LoRA-конфиг (опционально)

```json
{
  "name": "lora_max_quality",
  "r": 32,
  "lora_alpha": 64,
  "lora_dropout": 0.05,
  "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
  "bias": "none",
  "task_type": "CAUSAL_LM"
}

```

```python
lora_cfg = load_config("configs/lora_configs/lora_optimal.json")
```

Или отключить LoRA:

```python
lora_cfg = None
```

### 4. Подсказка (USE\_PROMPT)

В `main.py`:

```python
USE_PROMPT = False
```

Если True, будет использоваться небольшой вспомогательный промпт на основе train-данных при генерации.

---

## Добавление приватных ключей

### Telegram API

Файл `configs/telegram_config.json`:

```json
{
  "tg_messages": "data/tg_messages.csv",
  "api_id": 12345678,
  "api_hash": "your_api_hash",
  "session_name": "parser_nir_session",
  "channel_name": "channel_name",
  "phone": "",
  "code": "",
  "password": ""
}
```

Password это пароль при двухфакторной аутендификации. Code - проверочный код, который прийдет в сообщениие при попытке спарсить канал через ваш тг-api.

### OpenAI API

Файл `configs/gpt_api_config.json`:

```json
{
  "api_key": "sk-..."
}
```

---

## Запуск пайплайна

```bash
python main.py
```

Внутри он вызывает:

* скачивание модели,
* квантование (если задано),
* LoRA-файнтюнинг (если задано),
* сохранение весов,
* генерация и оценка метрик.

---

## Сравнение с GPT (API)

Для сравнения метрик с GPT-3.5-turbo:

```bash
python main_gpt.py
```

Требуется наличие API-ключа в `configs/gpt_api_config.json`.

---

## Метрики

Пайплайн оценивает:

* **BLEU** — n-грамм совпадения
* **ROUGE-L** — совпадение длинных подпоследовательностей
* **BERTScore** — семантическое сходство (самая важная метрика в контексте стилевого совпадения)
* **Perplexity** — логарифмическая уверенность
* **Repetition** — количество повторов в генерации



