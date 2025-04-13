from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Embedding, SimpleRNN, LSTM, GRU
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from models.base_model import BaseModel, rnn_loss

class ModelRNN(BaseModel):
    ''' A class for simple rnn model '''
    def create_model(self):
        ''' Create a rnn model '''
        model = Sequential([
            Embedding(self.vocab_size, self.config.embedding_dim),
            SimpleRNN(self.config.rnn_units, return_sequences=True, stateful=self.config.stateful, recurrent_initializer='glorot_uniform'),
            Dense(self.vocab_size)
        ])
        model.compile(optimizer='adam', loss=rnn_loss)
        return model