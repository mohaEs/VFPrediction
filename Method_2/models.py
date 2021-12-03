#!/usr/bin/env python

from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Input, Dense
from tensorflow.keras.activations import sigmoid, relu
import tensorflow as tf

def ModelOfPaper():
    timesteps=6
    LSTMcells=6
    Num_features=108

    reg_recur = tf.keras.regularizers.l1(0.01)

    InputLayer=Input(shape=(timesteps,Num_features))
    L1=LSTM(LSTMcells, activation='relu', recurrent_activation='sigmoid', 
    use_bias=False, dropout= 0.1,
    recurrent_dropout= 0.1,        
          kernel_regularizer= None,
    recurrent_regularizer= reg_recur,
            kernel_initializer='glorot_uniform',
    recurrent_initializer='glorot_uniform')(InputLayer)
    Output=Dense(52, activation='tanh', kernel_regularizer= None,
    kernel_initializer='orthogonal', use_bias=False)(L1)      
    model=Model(inputs=InputLayer,outputs=Output)
    lr=0.001
    optimizer = tf.keras.optimizers.Adam(lr= lr )
    loss = tf.keras.losses.MeanAbsoluteError()
    model.compile(loss=loss, optimizer=optimizer, )
    return model
    

    
