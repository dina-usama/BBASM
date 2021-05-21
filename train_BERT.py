
# @Reference1 : https://zhuanlan.zhihu.com/p/31638132
# @Reference2 : https://github.com/likejazz/Siamese-LSTM
# @Reference3 : https://github.com/LuJunru/Sentences_Pair_Similarity_Calculation_Siamese_LSTM
# @Reference4: https://github.com/hanxiao/bert-as-service


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys

import tensorflow as tf
from time import time
import pandas as pd
from sklearn.model_selection import train_test_split
import keras
import numpy as np
from gensim.models import KeyedVectors
from keras import initializers as initializers, regularizers, constraints
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, Flatten, Activation, RepeatVector, Permute, Lambda, Bidirectional, TimeDistributed, Dropout, Conv1D, GlobalMaxPool1D
from keras.layers.merge import multiply, concatenate
import keras.backend as K
from keras.engine import Layer

import keras.layers as layers




from AttentionLayer import AttentionLayer
from util import make_w2v_embeddings, split_and_zero_padding, ManDist
from bert_serving.client import BertClient

import tensorflow_hub as hub


TRAIN_CSV = 'C:/Users/dina_/Desktop/final/HHH-An-Online-Question-Answering-System-for-Medical-Questions/Data/Model_train_dev_test_dataset/Other_model_train_dev_test_dataset/train.csv'

flag = 'en'
embedding_path = 'C:/Users/dina_/Desktop/final/HHH-An-Online-Question-Answering-System-for-Medical-Questions/GoogleNews-vectors-negative300 .bin.gz'
# max_seq_length = 10
max_seq_length = 25
savepath = 'C:/Users/dina_/Desktop/final/HHH-An-Online-Question-Answering-System-for-Medical-Questions/Evaluation/model1.h5'

train_df = pd.read_csv(TRAIN_CSV, encoding='gb18030')




X = train_df[['question1', 'question2']]

Y = train_df['is_duplicate']
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.2)
# print(X_train.head())

X= X_train[['question1', 'question2']]
X_v= X_validation[['question1', 'question2']]

Y_train=Y_train

Y_validation=Y_validation





#train_question1
bc = BertClient()
BERT_train_question1=[]
k=0



for x in X['question1']:



    
    f = bc.encode([x])

    f = tf.convert_to_tensor(f)
    

    BERT_train_question1.append(f[0])
    k=k+1
    print("BERT_train_question1: point ",k)


BERT_train_question1 = tf.stack(BERT_train_question1)
print("Done BERT_train_question1",BERT_train_question1.shape)


#train_question2
k=0
BERT_train_question2=[]
for x in X['question2']:
    f = bc.encode([x])

    f = tf.convert_to_tensor(f)
    # print("after",f)

    BERT_train_question2.append(f[0])



    k = k + 1
    print("BERT_train_question2: point ", k)

BERT_train_question2 = tf.stack(BERT_train_question2)
print("Done BERT_train_question2",BERT_train_question2.shape)


#test_question1
k=0
BERT_test_question1=[]
for x in X_v['question1']:
    f = bc.encode([x])

    f = tf.convert_to_tensor(f)
    # print("after",f)

    BERT_test_question1.append(f[0])

    k = k + 1
    print("BERT_test_question1: point ", k)

BERT_test_question1 = tf.stack(BERT_test_question1)
print("Done BERT_test_question1",BERT_test_question1.shape)
# print("Done BERT_test_question1",np.shape(BERT_test_question1))

#test_question2
k=0
BERT_test_question2=[]
for x in X_v['question2']:
    f = bc.encode([x])

    f = tf.convert_to_tensor(f)
    # print("after",f)

    BERT_test_question2.append(f[0])

    k = k + 1

    print("BERT_test_question2: point ", k)

BERT_test_question2 = tf.stack(BERT_test_question2)
print("Done BERT_test_question2",BERT_test_question2.shape)
# print("Done BERT_test_question2",np.shape(BERT_test_question2))


Y_train = Y_train.values
Y_validation = Y_validation.values

assert X['question1'].shape == X['question2'].shape
assert len(X['question1']) == len(Y_train)


def shared_model_BBASM(_input):

    print(_input.shape)
    # out = Lambda(lambda x: x[0])(_input)
    # print(out.shape)

    print("Embedding Layer")


    l_lstm = Bidirectional(LSTM(100, return_sequences=True))(_input)
    print("Bi-Directional LSTM Layer")
    l_dense = TimeDistributed(Dense(200))(l_lstm)

    l_att = AttentionLayer()(l_dense)
    print("Attetion Layer")


    return l_att



if __name__ == '__main__':
 
    b = 1
    n_epoch = 100
    n_hidden = 50
    left_input = Input(shape=(max_seq_length,768,), dtype="float32",name="Input_layer")
    print(left_input.shape)
    # left_input = Input(shape=(max_sequence_length,), dtype='float32')
    # print('this is left inout', left_input)
    right_input = Input(shape=(max_seq_length,768,), dtype="float32")
    print(left_input.shape)
    # print('this is right inout', right_input)
    # right_input = Input(shape=(1,), dtype='float32')
    left_sen_representation = shared_model_BBASM(left_input)
    # print('left snetcen presentation', left_sen_representation)
    right_sen_representation = shared_model_BBASM(right_input)

    man_distance = ManDist()([left_sen_representation, right_sen_representation])
    sen_representation = concatenate([left_sen_representation, right_sen_representation, man_distance])
    similarity = Dense(1, activation='sigmoid')(Dense(2)(Dense(4)(Dense(16)(sen_representation))))
    model = Model(inputs=[left_input, right_input], outputs=[similarity])

    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    model.summary()

    training_start_time = time()
    
    malstm_trained = model.fit([BERT_train_question1, BERT_train_question2], Y_train,
                               steps_per_epoch=b, epochs=n_epoch,
                               validation_data=(
                                   [ BERT_test_question1, BERT_test_question2], Y_validation),validation_steps=1)
    training_end_time = time()
    print("Training time finished.\n%d epochs in %12.2f" % (n_epoch, training_end_time - training_start_time))
    yhat_probs = model.predict([BERT_test_question1, BERT_test_question2], steps=1, verbose=0)
    # predict crisp classes for test set
    yhat_classes = np.argmax(yhat_probs, axis=1)
    # yhat_classes = model.predict_classes([BERT_test_question1, BERT_test_question2], verbose=0)
    yhat_probs = yhat_probs[:, 0]
    # yhat_classes = yhat_classes[:, 0]
    yhat_classes = yhat_probs > 0.5
    accuracy = accuracy_score(Y_validation, yhat_classes)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(Y_validation, yhat_classes)
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(Y_validation, yhat_classes)
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(Y_validation, yhat_classes)
    print('F1 score: %f' % f1)
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
    plt.savefig('C:/Users/dina_/Desktop/final/HHH-An-Online-Question-Answering-System-for-Medical-Questions/Evaluation/history-graph.png')

    model.save(savepath)
    print(str(malstm_trained.history['val_acc'][-1])[:6] +
          "(max: " + str(max(malstm_trained.history['val_acc']))[:6] + ")")
    print("Done.")


