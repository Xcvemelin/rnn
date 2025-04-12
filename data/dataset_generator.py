import numpy as np
import tensorflow as tf
from configs.logging import logger
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

class DatasetGenerator():
    ''' A class for dataset generation '''
    def __init__(self):
        ''' Initialize the TextGenerator variables '''
        self.chars = []
        self.char2idx = {}
        self.idx2char = np.array([])
        self.vocab_size = 0
        self.batch_size = 64

    def load_text(self, url:str='https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt') -> str:
        ''' Load text from link '''
        path_to_file = tf.keras.utils.get_file('shakespeare.txt', url)
        with open(path_to_file, 'rb') as file:
            text = file.read().decode(encoding='utf-8')
        logger.info(f'Text length: {len(text)} characters')
        return text

    def tokenization(self, text:str):
        ''' Text tokenization '''
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        self.char2idx = {c: i for i, c in enumerate(self.chars)}
        self.idx2char = np.array(self.chars)
        text_as_int = np.array([self.char2idx[c] for c in text])
        logger.debug(f"Vocabulary size: {self.vocab_size}")
        return text_as_int, self.char2idx, self.idx2char

    def prepare_dataset(self, text_as_int):
        ''' Prepare dataset for training '''
        seq_length = 20
        char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
        sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

        def split_input_target(chunk):
            input_text = chunk[:-1]
            target_text = chunk[1:]
            return input_text, target_text
        
        dataset = sequences.map(split_input_target)
        dataset = (
            dataset.shuffle(1000)
            .batch(self.batch_size, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )
        logger.debug(f"Dataset prepared: {dataset}")
        return dataset

if __name__ == '__main__':
    generator = DatasetGenerator()
    text = generator.load_text()
    text_as_int = generator.tokenization(text)
    dataset = generator.prepare_dataset(text_as_int)
    logger.info(dataset)