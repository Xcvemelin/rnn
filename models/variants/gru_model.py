from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Embedding, SimpleRNN, LSTM, GRU
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from models.base_model import BaseModel, rnn_loss

class ModelGRU(BaseModel):
    ''' A class for gru model '''
    def create_model(self):
        ''' Create a GRU model '''
        model = Sequential([
            Input(batch_shape=(self.config.batch_size, None)),
            Embedding(self.vocab_size, self.config.embedding_dim),
            GRU(self.config.rnn_units, return_sequences=True, stateful=self.config.stateful, recurrent_initializer='glorot_uniform'),
            Dense(self.vocab_size)
        ])
        model.compile(
            optimizer='adam',
            loss=SparseCategoricalCrossentropy(from_logits=True)
        )
        return model