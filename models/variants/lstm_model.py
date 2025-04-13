from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Embedding, SimpleRNN, LSTM, GRU
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from models.base_model import BaseModel, rnn_loss

class ModelLSTM(BaseModel):
    ''' A class for lstm model '''
    def create_model(self):
        ''' Create a lstm model '''
        input_shape = (self.config.batch_size, None) if self.config.stateful else (None, None)
        model = Sequential([
            Input(batch_shape=input_shape if self.config.stateful else None, shape=(None,) if not self.config.stateful else None),
            Embedding(self.vocab_size, self.config.embedding_dim),
            LSTM(self.config.rnn_units, 
                 return_sequences=True, 
                 stateful=self.config.stateful,
                 recurrent_initializer='glorot_uniform'),
            Dense(self.vocab_size)
        ])
        model.compile(
            optimizer='adam', 
            loss=rnn_loss
        )
        return model