
# @Reference1 : https://zhuanlan.zhihu.com/p/31638132
# @Reference2 : https://github.com/likejazz/Siamese-LSTM
# @Reference3 : https://github.com/LuJunru/Sentences_Pair_Similarity_Calculation_Siamese_LSTM


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys




import tensorflow  as tf


from time import time
import pandas as pd
from sklearn.model_selection import train_test_split
import keras
import numpy as np
from gensim.models import KeyedVectors
from keras import initializers as initializers, regularizers, constraints
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, Flatten, Activation, RepeatVector, Permute, Lambda, \
    Bidirectional, TimeDistributed, Dropout, Conv1D, GlobalMaxPool1D
from keras.layers.merge import multiply, concatenate
import keras.backend as K


from AttentionLayer import AttentionLayer
from util import make_w2v_embeddings, split_and_zero_padding, ManDist




TRAIN_CSV = 'C:/Users/dina_/Desktop/final/HHH-An-Online-Question-Answering-System-for-Medical-Questions/Data/Model_train_dev_test_dataset/Other_model_train_dev_test_dataset/train.csv'

flag = 'en'
embedding_path = 'C:/Users/dina_/Desktop/final/HHH-An-Online-Question-Answering-System-for-Medical-Questions/wiki-news-300d-1M.vec'
embedding_dim = 300
max_seq_length = 10
savepath = 'C:/Users/dina_/Desktop/Embeddings results/FastText embeddings/model.h5'

train_df = pd.read_csv(TRAIN_CSV, encoding='gb18030')

embedding_dict = KeyedVectors.load_word2vec_format(embedding_path)


for q in ['question1', 'question2']:
    train_df[q + '_n'] = train_df[q]




print(train_df['question1_n'], train_df['question1_n'])
print(train_df.head())
train_df = train_df

train_df, embeddings = make_w2v_embeddings(flag, embedding_dict, train_df, embedding_dim=embedding_dim)
# print('after emedding',len(embeddings))
# print('size of embedding',len(train_df['question1_n'][0]))
# print('size of embedding',train_df['question1_n'][0])

X = train_df[['question1_n', 'question2_n']]
Y = train_df['is_duplicate']
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.2)
print(X_train.head())
X_train = split_and_zero_padding(X_train, max_seq_length)
X_validation = split_and_zero_padding(X_validation, max_seq_length)

Y_train = Y_train.values
Y_validation = Y_validation.values


assert X_train['left'].shape == X_train['right'].shape
assert len(X_train['left']) == len(Y_train)



def shared_model_HBDA(_input):
   
    embedding_layer = Embedding(len(embeddings) + 1,
                                embedding_dim,
                                input_length=max_seq_length)

    print(embedding_layer)
    print(type(embedding_layer))

    embedded_sequences = embedding_layer(_input)
    print('embedded sequence is ', embedded_sequences)
    print('this is the first embedding layer', embedded_sequences)
   
    l_lstm = Bidirectional(LSTM(100, return_sequences=True))(embedded_sequences)
    l_dense = TimeDistributed(Dense(200))(l_lstm)
    l_att = AttentionLayer()(l_dense)

    

    return l_att



if __name__ == '__main__':
   
    batch_size = 1024
    n_epoch = 9
    n_hidden = 50

    left_input = Input(shape=(max_seq_length,), dtype='float32')
    print('this is left inout', left_input)
   
    right_input = Input(shape=(max_seq_length,), dtype='float32')
    left_sen_representation = shared_model_HBDA(left_input)
    print('left snetcen presentation', left_sen_representation)
    right_sen_representation = shared_model_HBDA(right_input)

    man_distance = ManDist()([left_sen_representation, right_sen_representation])
    sen_representation = concatenate([left_sen_representation, right_sen_representation, man_distance])
    similarity = Dense(1, activation='sigmoid')(Dense(2)(Dense(4)(Dense(16)(sen_representation))))
    model = Model(inputs=[left_input, right_input], outputs=[similarity])

    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    model.summary()

    training_start_time = time()
    malstm_trained = model.fit( [ X_train['left'], X_train['right']], Y_train,
                               batch_size=batch_size, epochs=n_epoch,
                               validation_data=(
                               [X_validation['left'], X_validation['right']], Y_validation))
    training_end_time = time()
    print("Training time finished.\n%d epochs in %12.2f" % (n_epoch, training_end_time - training_start_time))

    # Plot accuracy
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.subplot(211)
    plt.plot(malstm_trained.history['acc'])
    plt.plot(malstm_trained.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot loss
    plt.subplot(212)
    plt.plot(malstm_trained.history['loss'])
    plt.plot(malstm_trained.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    plt.tight_layout(h_pad=1.0)
    plt.savefig('C:/Users/dina_/Desktop/Embeddings results/FastText embeddings/history-graph.png')

    model.save(savepath)
    print(str(malstm_trained.history['val_acc'][-1])[:6] +
          "(max: " + str(max(malstm_trained.history['val_acc']))[:6] + ")")
    print("Done.")
