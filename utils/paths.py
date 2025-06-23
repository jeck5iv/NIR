from pathlib import Path
from typing import Dict

class Paths:
    def __init__(self, model_config: Dict, telegram_config: Dict):
        self.model_config = model_config
        self.telegram_config = telegram_config
        self.base_dir = Path(__file__).parent.parent

    def resolve(self, path: str) -> Path:
        return self.base_dir / path

    @property
    def original_model(self) -> Path:
        return self.resolve(self.model_config["original_model"])

    @property
    def quantized_model(self) -> Path:
        return self.resolve(self.model_config["quantized_model"])

    @property
    def data_dir(self) -> Path:
        return self.resolve("data")

    @property
    def train_data(self) -> Path:
        return self.resolve(self.model_config["train_data"])

    @property
    def test_data(self) -> Path:
        return self.resolve(self.model_config["test_data"])

    @property
    def tg_messages(self) -> Path:
        return self.resolve(self.telegram_config["tg_messages"])

    @property
    def val_data(self) -> Path:
        return self.resolve(self.model_config["val_data"])

    @property
    def model_weights_dir(self) -> Path:
        return self.resolve(self.model_config["model_weights_dir"])
