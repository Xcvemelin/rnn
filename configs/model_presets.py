from dataclasses import dataclass

@dataclass
class ModelPresets:
    RNN = {
        "embedding_dim": 128,
        "rnn_units": 128,
        "batch_size": 256,
        "stateful": True,
        "epochs": 15,
        "learning_rate": 0.01
    }
    GRU = {
        "embedding_dim": 256,
        "rnn_units": 512,
        "batch_size": 64,
        "stateful": True,
        "epochs": 10,
        "learning_rate": 0.001
    }

    LSTM = {
        "embedding_dim": 256,
        "rnn_units": 512,
        "batch_size": 64,
        "stateful": True,
        "epochs": 10,
        "learning_rate": 0.001
    }