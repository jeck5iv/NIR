import pandas as pd
import re
from pathlib import Path
from utils.paths import Paths

def clean_text(text: str) -> str:
    """Очистка текста от лишних символов"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'[^\w\s,]', '', text)
    return text.strip()

def prepare_data(paths: Paths):
    """Основная функция подготовки данных"""
    print(f"\nPreparing data from {paths.tg_messages}")

    data = pd.read_csv(paths.tg_messages)
    data['cleaned_text'] = data['text'].fillna('').apply(clean_text)

    data = data[data['cleaned_text'].str.len() >= 100].copy()

    train_val = data.sample(frac=0.9, random_state=1986)
    test_data = data.drop(train_val.index)

    train_data = train_val.sample(frac=8/9, random_state=1986)
    val_data = train_val.drop(train_data.index)

    paths.data_dir.mkdir(exist_ok=True)
    train_data.to_csv(paths.train_data, index=False)
    val_data.to_csv(paths.val_data, index=False)
    test_data.to_csv(paths.test_data, index=False)

    print("Data prepared:")
    print(f"- Train: {paths.train_data} ({len(train_data)} rows)")
    print(f"- Val:   {paths.val_data} ({len(val_data)} rows)")
    print(f"- Test:  {paths.test_data} ({len(test_data)} rows)")

