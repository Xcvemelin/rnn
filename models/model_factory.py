from .variants import ModelRNN, ModelGRU, ModelLSTM
from configs.model_config import ModelConfig

class ModelFactory():
    @staticmethod
    def create_model(vocab_size, dataset, config: ModelConfig):
        ''' Factory function to create RNN models using structural pattern matching '''
        match config.model_name.lower():
            case "simplernn" | "rnn":
                return ModelRNN(vocab_size, dataset, config)
            case "gru":
                return ModelGRU(vocab_size, dataset, config)
            case "lstm":
                return ModelLSTM(vocab_size, dataset, config)
            case _:
                available = ["SimpleRNN", "GRU", "LSTM"]
                raise ValueError(
                    f"Unknown model: {config.model_name}. "
                    f"Available models: {available}"
                )

    @staticmethod
    def build_model(path, vocab_size, dataset, config: ModelConfig):
        ''' Create and train model '''
        model = ModelFactory.create_model(vocab_size, dataset, config)
        return model.load_or_train_model(path)
