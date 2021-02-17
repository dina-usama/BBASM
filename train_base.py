# -*- coding: utf-8 -*-
# @Author  : Bill Bao
# @File    : train.py
# @Software: PyCharm and Spyder
# @Environment : Python 3.6+
# @Reference1 : https://zhuanlan.zhihu.com/p/31638132
# @Reference2 : https://github.com/likejazz/Siamese-LSTM
# @Reference3 : https://github.com/LuJunru/Sentences_Pair_Similarity_Calculation_Siamese_LSTM

# 基础包
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys

sys.path.append('C:/Users/dina_/Desktop/final/HHH-An-Online-Question-Answering-System-for-Medical-Questions/Bert')

sys.path.append(
    'C:/Users/dina_/Desktop/final/HHH-An-Online-Question-Answering-System-for-Medical-Questions/Bert Fixed/bert-sentence-encoder')

# import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import tensorflow  as tf

# ؟tf.compat.v1.disable_eager_execution()
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


# from tensorflow.keras import backend
# from tensorflow.keras import backend as k
# from tensorflow.keras import backend

# from BertTuned import get_features


# from AttentionLayer import AttentionLayer


from AttentionLayer import AttentionLayer
from util import make_w2v_embeddings, split_and_zero_padding, ManDist
# from BertEncoder import BertSentenceEncoder

'''pip
本配置文件用于训练孪生网络
'''

# ------------------预加载------------------ #

TRAIN_CSV = 'C:/Users/dina_/Desktop/final/HHH-An-Online-Question-Answering-System-for-Medical-Questions/Data/Model_train_dev_test_dataset/Other_model_train_dev_test_dataset/train.csv'

flag = 'en'
embedding_path = 'C:/Users/dina_/Desktop/final/HHH-An-Online-Question-Answering-System-for-Medical-Questions/GoogleNews-vectors-negative300 .bin.gz'
embedding_dim = 300
max_seq_length = 10
savepath = 'C:/Users/dina_/Desktop/final/HHH-An-Online-Question-Answering-System-for-Medical-Questions./data/model1.h5'

train_df = pd.read_csv(TRAIN_CSV, encoding='gb18030')
# adds all the sentences in one structure to get bert embeddings

# trainTotal = []
# train_question1 = train_df['question1']
# print('this is first set', len(train_question1))
# train_question2 = train_df['question2']
# print('this is sec set', len(train_question2))
# for i in range(2 * len(train_question1)):
#     if i < len(train_question1):
#         trainTotal.append(train_question1[i])
#         x = i + 1
#     else:
#         trainTotal.append(train_question2[i - len(train_question1)])
# print('this is total', len(trainTotal))
# # print (trainTotal)


# word_encodings = get_features(trainTotal)
# print('Size of the list of encondings',len(word_encodings))
# print('Dimension of enconding a single question',word_encodings[0].shape)
# print('Dimension of the full word_encodings',word_encodings[0].shape)



# 加载词向量
print("Loading word2vec model(it may takes 2-3 mins) ...")
embedding_dict = KeyedVectors.load_word2vec_format(embedding_path, binary=True)

# 读取并加载训练集
#
# print(train_df['question1'])
for q in ['question1', 'question2']:
    train_df[q + '_n'] = train_df[q]


# for x in range(0, 2 * train_df.shape[0]):
#     if x < train_df.shape[0]:
#
#         train_df['question1_n'] = x
#     else:
#         train_df['question2_n'] = x

print(train_df['question1_n'], train_df['question1_n'])
print(train_df.head())
train_df = train_df
# 将训练集词向量化
train_df, embeddings = make_w2v_embeddings(flag, embedding_dict, train_df, embedding_dim=embedding_dim)
# print('after emedding',len(embeddings))
# print('size of embedding',len(train_df['question1_n'][0]))
# print('size of embedding',train_df['question1_n'][0])

# 分割训练集
X = train_df[['question1_n', 'question2_n']]
Y = train_df['is_duplicate']
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.2)
print(X_train.head())
X_train = split_and_zero_padding(X_train, max_seq_length)
X_validation = split_and_zero_padding(X_validation, max_seq_length)

# 将标签转化为数值
Y_train = Y_train.values
Y_validation = Y_validation.values


# 确认数据准备完毕且正确
assert X_train['left'].shape == X_train['right'].shape
assert len(X_train['left']) == len(Y_train)

# -----------------基础函数------------------ #


def shared_model_HBDA(_input):
    # 词向量化
    # embedded = Embedding(len(embeddings), embedding_dim, weights=[embeddings], input_shape=(max_seq_length,),
    #                     trainable=False)(_input)

    embedding_layer = Embedding(len(embeddings) + 1,
                                embedding_dim,
                                input_length=max_seq_length)

    print(embedding_layer)
    print(type(embedding_layer))

    embedded_sequences = embedding_layer(_input)
    print('embedded sequence is ', embedded_sequences)
    print('this is the first embedding layer', embedded_sequences)
    # print('this is size of the first embedding layer', embedded_sequences.shape())

    l_lstm = Bidirectional(LSTM(100, return_sequences=True))(embedded_sequences)
    l_dense = TimeDistributed(Dense(200))(l_lstm)
    l_att = AttentionLayer()(l_dense)

    # 单层Bi-LSTM
    # activations = Bidirectional(LSTM(n_hidden, return_sequences=True), merge_mode='concat')(embedded)

    # dropout
    # activations = Dropout(0.5)(activations)

    # Words level attention model
    # word_dense = Dense(1, activation='relu', name='word_dense')(activations)
    # word_att,word_coeffs = AttentionLayer(EMBED_SIZE,True,name='word_attention')(word_dense)

    # Attention
    # attention = TimeDistributed(Dense(1, activation='tanh'))(activations)
    # attention = Flatten()(attention)
    # attention = Activation('softmax')(attention)
    # attention = RepeatVector(n_hidden * 2)(attention)
    # attention = Permute([2, 1])(attention)
    # sent_representation = dot([activations, attention],axes=1)
    # dropout
    # sent_representation = Dropout(0.1)(sent_representation)

    return l_att







# -----------------主函数----------------- #

if __name__ == '__main__':
    # 超参
    batch_size = 1024
    n_epoch = 9
    n_hidden = 50
    # left_input = Input(shape=(1,), dtype='float32')
    left_input = Input(shape=(max_seq_length,), dtype='float32')
    print('this is left inout', left_input)
    # right_input = Input(shape=(1,), dtype='float32')
    right_input = Input(shape=(max_seq_length,), dtype='float32')
    left_sen_representation = shared_model_HBDA(left_input)
    print('left snetcen presentation', left_sen_representation)
    right_sen_representation = shared_model_HBDA(right_input)

    # 引入曼哈顿距离，把得到的变换concat上原始的向量再通过一个多层的DNN做了下非线性变换、sigmoid得相似度
    # 没有使用https://zhuanlan.zhihu.com/p/31638132中提到的马氏距离，尝试了曼哈顿距离、点乘和cos，效果曼哈顿最好
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
    # plt.savefig('./data/history-graph.png')

    model.save(savepath)
    print(str(malstm_trained.history['val_acc'][-1])[:6] +
          "(max: " + str(max(malstm_trained.history['val_acc']))[:6] + ")")
    print("Done.")
