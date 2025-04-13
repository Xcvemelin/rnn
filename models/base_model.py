import os
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.saving import register_keras_serializable
from configs.logging import logger
from configs.model_config import ModelConfig

@register_keras_serializable(package='CustomLosses')
def rnn_loss(labels, logits):
    return SparseCategoricalCrossentropy(from_logits=True)(labels, logits)

class BaseModel():
    ''' A class for rnn models '''
    def __init__(self, vocab_size, dataset, config: ModelConfig):
        '''Initialize the ModelGenerator variables'''
        self.vocab_size = vocab_size
        self.dataset = dataset
        self.config = config
        logger.info(f"Initializing: {config}")

    def create_model(self):
        '''Create a model (to be implemented in child classes)'''
        raise NotImplementedError("This method should be implemented in child classes")

    def train_model(self, model):
        ''' Train the model '''
        logger.info(f"Starting training for {self.config.epochs} epochs")
        model.fit(self.dataset, epochs=self.config.epochs)
        logger.success("Training completed")

    def __save_model(self, model, path):
        ''' Save trained model '''
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(path)
        logger.info(f"Model saved to {path}")

    def load_or_train_model(self, path='outputs/trained_models/model.keras'):
        ''' Load existing model or train a new one '''
        need_train = not os.path.exists(path)
        model = None

        if not need_train:
            logger.info(f"Loading trained model from {path}...")
            try:
                model = tf.keras.models.load_model(path, custom_objects={'rnn_loss': rnn_loss})
                logger.success("Model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                need_train = True

        if need_train:
            logger.info("Training new model...")
            model = self.create_model()
            self.train_model(model)
            self.__save_model(model, path)

        if self.config.stateful and self.config.batch_size != 1:
            logger.info("Creating generation model with batch_size=1...")
            original_batch_size = self.config.batch_size
            self.config.batch_size = 1
            gen_model = self.create_model()
            gen_model.build(input_shape=(1, None))
            gen_model.set_weights(model.get_weights())
            self.config.batch_size = original_batch_size
            return gen_model

        return model