import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from configs.logging import logger
from data.dataset_generator import DatasetGenerator
from utils.text_generator import TextGenerator
from models.model_factory import ModelFactory
from configs.model_config import ModelConfig
from configs.paths import TRAINED_MODELS_DIR

def main():
    ''' 
    Model types: SimpleRNN, GRU, LSTM
    Text generations: 
        text_generation, 
        text_generation_with_sampling, 
        beam_search, 
        visualize_logits_probs
    '''
    try:
        config = ModelConfig(
            model_name="rnn",
            embedding_dim=128, 
            rnn_units=256, 
            batch_size=64, 
            stateful=True,
            epochs=10
        )
        logger.info("Starting application...")
        
        dataset_gen = DatasetGenerator()
        text = dataset_gen.load_text()
        text_as_int, char2idx, idx2char = dataset_gen.tokenization(text)
        dataset = dataset_gen.prepare_dataset(text_as_int)

        model_path = TRAINED_MODELS_DIR / f"{config.model_name}_model.keras"
        logger.info(f"Creating {config.model_name} model")
        model_gen = ModelFactory.create_model(len(char2idx), dataset, config)
        model = model_gen.load_or_train_model(path=model_path)
        
        text_generation = TextGenerator(char2idx, idx2char)

        logger.info("Generating text with sampling...")
        generated_text = text_generation.with_sampling(model, "ROMEO", num_generate=300, temperature=0.7, method='top_k', k=5)
        logger.info(f"Generated text (sampling):\n{generated_text}")

        logger.info("Generating text with beam search...")
        generated_text = text_generation.beam_search(model, "ROMEO")
        logger.info(f"Generated text (beam search):\n{generated_text}")

        logger.info("Visualizing logits and probs...")
        text_generation.visualize_logits_probs(model, "ROMEO")
        
        logger.success("Application finished successfully")
    except Exception as e:
        logger.error(f"Application failed: {e}")
        raise

if __name__ == '__main__':
    main()