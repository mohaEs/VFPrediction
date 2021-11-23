#!/usr/bin/env python

from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Input, Dense
from tensorflow.keras.activations import sigmoid, relu


def ModelOfPaper():
    timesteps=6
    LSTMcells=6
    Num_features=108
    InputLayer=Input(shape=(timesteps,Num_features))
    L1=LSTM(LSTMcells, activation='tanh', recurrent_activation='sigmoid', use_bias=True)(InputLayer)
    Output=Dense(52, activation='tanh', use_bias=True)(L1)      
    model=Model(inputs=InputLayer,outputs=Output)
    return model
    

    
