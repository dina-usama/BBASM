
# @Reference: https://github.com/14H034160212/HHH-An-Online-Question-Answering-System-for-Medical-Questions


import tensorflow as tf

from tensorflow.keras.models import load_model, Model
from tensorflow.python.keras import backend as K

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
K.set_session(tf.compat.v1.Session(config=config))


from keras import Input
import keras
from keras.optimizers import Adam
import keras.backend as K
import numpy as np
from keras.layers import *
from keras.layers.core import Dense, Dropout
from keras.models import Sequential, Model
from keras.layers.recurrent import LSTM
import pandas as pd






from nltk import tokenize
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical




class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(** kwargs)
    
    def build(self, input_shape):
        assert len(input_shape)==3
        
        self.W = self.add_weight(name='att_weight', 
                                 shape=(input_shape[1], input_shape[1]),
                                 initializer='uniform',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs, mask=None):
       
        x = K.permute_dimensions(inputs, (0, 2, 1))
       
        a = K.softmax(K.tanh(K.dot(x, self.W)))
        a = K.permute_dimensions(a, (0, 2, 1))
        outputs = a * inputs
        outputs = K.sum(outputs, axis=1)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]
		
