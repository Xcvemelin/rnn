import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding, LSTM, GRU, Input
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.saving import register_keras_serializable
from configs.logging import logger
from pathlib import Path

@register_keras_serializable(package='CustomLosses')
def rnn_loss(labels, logits):
    return SparseCategoricalCrossentropy(from_logits=True)(labels, logits)

class ModelGenerator():
    ''' A class for rnn models '''
    def __init__(self, model_name, vocab_size, dataset, embedding_dim=64, rnn_units=512, batch_size=64, stateful=False):
        '''Initialize the ModelGenerator variables'''
        self.model_name = model_name
        self.vocab_size = vocab_size
        self.dataset = dataset
        self.embedding_dim = embedding_dim
        self.rnn_units = rnn_units
        self.batch_size = batch_size
        self.stateful = stateful
        logger.info(f"Initializing {model_name} model with vocab_size={vocab_size}, rnn_units={rnn_units}")

    def create_model(self):
        '''Create a model (to be implemented in child classes)'''
        raise NotImplementedError("This method should be implemented in child classes")

    def train_model(self, model, epochs=50):
        ''' Train the model '''
        logger.info(f"Starting training for {epochs} epochs")
        model.fit(self.dataset, epochs=epochs)
        logger.success("Training completed")

    def __save_model(self, model, path):
        ''' Save trained model '''
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(path)
        logger.info(f"Model saved to {path}")

    def load_or_train_model(self, path='outputs/trained_models/model.keras', epochs=50):
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
            self.train_model(model, epochs=epochs)
            self.__save_model(model, path)

        # For stateful models, we need to recreate the model with batch_size=1 for generation
        if self.stateful and self.batch_size != 1:
            logger.info("Creating generation model with batch_size=1...")
            original_batch_size = self.batch_size
            self.batch_size = 1
            gen_model = self.create_model()
            gen_model.build(input_shape=(1, None))
            gen_model.set_weights(model.get_weights())
            self.batch_size = original_batch_size
            return gen_model

        return model

class ModelRNN(ModelGenerator):
    ''' A class for simple rnn model '''
    def create_model(self):
        ''' Create a rnn model '''
        model = Sequential([
            Embedding(self.vocab_size, self.embedding_dim),
            SimpleRNN(self.rnn_units, return_sequences=True, stateful=self.stateful, recurrent_initializer='glorot_uniform'),
            Dense(self.vocab_size)
        ])
        model.compile(optimizer='adam', loss=rnn_loss)
        return model

class ModelGRU(ModelGenerator):
    ''' A class for gru model '''
    def create_model(self):
        ''' Create a GRU model '''
        model = Sequential([
            Input(batch_shape=(self.batch_size, None)),
            Embedding(self.vocab_size, self.embedding_dim),
            GRU(self.rnn_units, return_sequences=True, stateful=self.stateful, recurrent_initializer='glorot_uniform'),
            Dense(self.vocab_size)
        ])
        model.compile(
            optimizer='adam',
            loss=SparseCategoricalCrossentropy(from_logits=True)
        )
        return model

class ModelLSTM(ModelGenerator):
    ''' A class for lstm model '''
    def create_model(self):
        ''' Create a lstm model '''
        input_shape = (self.batch_size, None) if self.stateful else (None, None)
        model = Sequential([
            Input(batch_shape=input_shape if self.stateful else None, shape=(None,) if not self.stateful else None),
            Embedding(self.vocab_size, self.embedding_dim),
            LSTM(self.rnn_units, 
                 return_sequences=True, 
                 stateful=self.stateful,
                 recurrent_initializer='glorot_uniform'),
            Dense(self.vocab_size)
        ])
        model.compile(
            optimizer='adam', 
            loss=rnn_loss
        )
        return model

def create_model(model_name, vocab_size, dataset, **kwargs):
    ''' Factory function to create RNN models using structural pattern matching '''
    match model_name.lower():
        case "simplernn" | "rnn":
            return ModelRNN(model_name, vocab_size, dataset, **kwargs)
        case "gru":
            return ModelGRU(model_name, vocab_size, dataset, **kwargs)
        case "lstm":
            return ModelLSTM(model_name, vocab_size, dataset, **kwargs)
        case _:
            available = ["SimpleRNN", "GRU", "LSTM"]
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available models: {available}"
            )

def create_model_dict(model_name, vocab_size, dataset, **kwargs):
    ''' Factory function to create RNN models using structural pattern matching '''
    model_classes = {
        "simplernn": ModelRNN,
        "gru": ModelGRU,
        "lstm": ModelLSTM
    }
    model_name = model_name.lower()
    if model_name not in model_classes:
        raise ValueError(f'Model {model_name} not found. Available models: {list(model_classes.keys())}')
    return model_classes[model_name](vocab_size, dataset, **kwargs)