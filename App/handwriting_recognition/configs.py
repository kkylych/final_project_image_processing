import os
from datetime import datetime

from mltu.configs import BaseModelConfigs

_HERE = os.path.dirname(os.path.abspath(__file__))

class ModelConfigs(BaseModelConfigs):
    def __init__(self):
        super().__init__()
        self.model_path = os.path.join(_HERE, "Models", "handwriting_recognition", datetime.strftime(datetime.now(), "%Y%m%d%H%M"))
        self.vocab = ""
        self.height = 32
        self.width = 128
        self.max_text_length = 0
        self.batch_size = 16
        self.learning_rate = 0.0005
        self.train_epochs = 5
        self.train_workers = 4