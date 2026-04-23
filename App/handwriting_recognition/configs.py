"""
configs.py — Model hyperparameters and training configuration.

ModelConfigs inherits from mltu.BaseModelConfigs which provides
save()/load() helpers (YAML format) so the exact config used for
each training run is stored alongside the model weights.
"""

import os
from datetime import datetime

from mltu.configs import BaseModelConfigs

_HERE = os.path.dirname(os.path.abspath(__file__))

class ModelConfigs(BaseModelConfigs):
    """Training configuration for the handwriting recognition model.

    Attributes:
        model_path (str): Directory where model weights, logs, and CSVs are saved.
        vocab (str): String of all unique characters seen in the training set.
        height (int): Image height after resizing (pixels).
        width (int): Image width after resizing (pixels).
        max_text_length (int): Maximum label length (used for CTC padding).
        batch_size (int): Number of samples per training batch.
        learning_rate (float): Initial Adam learning rate.
        train_epochs (int): Maximum number of training epochs.
        train_workers (int): Number of parallel data-loading workers.
    """
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