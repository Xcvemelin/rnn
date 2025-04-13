
from dataclasses import dataclass

@dataclass
class ModelConfig:
    model_name: str = ""
    embedding_dim: int = 0
    rnn_units: int = 0
    batch_size: int = 0
    stateful: bool = False
    epochs: int = 0

    def __post_init__(self):
        presets = ModelPresets()
        model_preset = getattr(presets, self.model_name.upper(), None)
        if model_preset:
            for key, value in model_preset.items():
                if getattr(self, key) is None:
                    setattr(self, key, value)

