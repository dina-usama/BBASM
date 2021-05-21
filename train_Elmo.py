
# @Reference1 : https://zhuanlan.zhihu.com/p/31638132
# @Reference2 : https://github.com/likejazz/Siamese-LSTM
# @Reference3 : https://github.com/LuJunru/Sentences_Pair_Similarity_Calculation_Siamese_LSTM


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

import tensorflow_hub as hub



TRAIN_CSV = 'C:/Users/dina_/Desktop/final/HHH-An-Online-Question-Answering-System-for-Medical-Questions/Data/Model_train_dev_test_dataset/Other_model_train_dev_test_dataset/train.csv'

flag = 'en'
embedding_path = 'C:/Users/dina_/Desktop/final/HHH-An-Online-Question-Answering-System-for-Medical-Questions/GoogleNews-vectors-negative300 .bin.gz'


max_seq_length = 72
savepath = 'C:/Users/dina_/Desktop/Embeddings results/Elmo embeddings/model1.h5'

train_df = pd.read_csv(TRAIN_CSV, encoding='gb18030')
print('loaded data')

elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)


def text_to_word_list(flag, text):  
    text = str(text)
    text = text.lower()

    if flag == 'cn':
        pass
    else:
     
        import re
        text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "cannot ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r",", " ", text)
        text = re.sub(r"\.", " ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\/", " ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"\-", " - ", text)
        text = re.sub(r"\=", " = ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r" u s ", " american ", text)
        text = re.sub(r"\0s", "0", text)
        text = re.sub(r" 9 11 ", "911", text)
        text = re.sub(r"e - mail", "email", text)
        text = re.sub(r"j k", "jk", text)
        text = re.sub(r"\s{2,}", " ", text)

    text = text.split()

    return text



def ELMoEmbedding(input_text):
    # print("inside elmoembedding function")
    return elmo(tf.reshape(tf.cast(input_text, tf.string), [-1]), signature="default", as_dict=True)["elmo"]

def elmo_vectors(x):
  embeddings = elmo(x, signature="default", as_dict=True)["elmo"]

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    # return average of ELMo features
    return sess.run(tf.reduce_mean(embeddings,1))


X = train_df[['question1', 'question2']]

Y = train_df['is_duplicate']
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.2)
# print(X_train.head())

X= X_train[['question1', 'question2']][0:320]# dataset shortened for trials
X_v= X_validation[['question1', 'question2']][0:80]# dataset shortened for trials

print("loaded training and validation data")
Y_train=Y_train[0:320]
# print(Y_train)
Y_validation=Y_validation[0:80]

# print(X)



#train_question1
elmo_train_question1=[]
k=0

for x in X['question1']:
    d=[]

    x = text_to_word_list("eng",x)
    d.append(x)
    leng=[len(d[0])]

    print("text is",x)
    # f= elmo([x], signature="default", as_dict=True)["elmo"]   #ater fixing it dontforget to change it to the main dataset
    # print ("before",f)
    f = elmo(
        inputs={
            "tokens": d,
            "sequence_len":leng
        },
        signature="tokens",
        as_dict=True)["elmo"]
    print("emdeding of text",f)

    paddingCons= int(f.shape[1])
    paddingCons=max_seq_length-paddingCons
    paddings= tf.constant([[0,0 ], [0, paddingCons], [0,0]])
    f= tf.pad(f, paddings, "CONSTANT")
    # print("after",f)

    elmo_train_question1.append(f[0])
    k=k+1
    print("elmo_train_question1: point ",k)


elmo_train_question1 = tf.stack(elmo_train_question1)
print("Done Elmo_train_question1",elmo_train_question1)

#train_question2
k=0
elmo_train_question2=[]
for x in X['question2']:
    d = []

    x = text_to_word_list("eng", x)
    d.append(x)
    leng = [len(d[0])]

    print("text is", x)
    # f= elmo([x], signature="default", as_dict=True)["elmo"]   #ater fixing it dontforget to change it to the main dataset
    # print ("before",f)
    f = elmo(
        inputs={
            "tokens": d,
            "sequence_len": leng
        },
        signature="tokens",
        as_dict=True)["elmo"]
    print("emdeding of text", f)

    paddingCons = int(f.shape[1])
    paddingCons = max_seq_length - paddingCons
    paddings = tf.constant([[0, 0], [0, paddingCons], [0, 0]])
    f = tf.pad(f, paddings, "CONSTANT")
    elmo_train_question2.append(f[0])

    k = k + 1
    print("elmo_train_question2: point ", k)

elmo_train_question2 = tf.stack(elmo_train_question2)
print("Done Elmo_train_question2",elmo_train_question2.shape)


#test_question1
k=0
elmo_test_question1=[]
for x in X_v['question1']:
    d = []

    x = text_to_word_list("eng", x)
    d.append(x)
    leng = [len(d[0])]

    print("text is", x)
    # f= elmo([x], signature="default", as_dict=True)["elmo"]   #ater fixing it dontforget to change it to the main dataset
    # print ("before",f)
    f = elmo(
        inputs={
            "tokens": d,
            "sequence_len": leng
        },
        signature="tokens",
        as_dict=True)["elmo"]
    print("emdeding of text", f)

    paddingCons = int(f.shape[1])
    paddingCons = max_seq_length - paddingCons
    paddings = tf.constant([[0, 0], [0, paddingCons], [0, 0]])
    f = tf.pad(f, paddings, "CONSTANT")
    elmo_test_question1.append(f[0])
    k = k + 1
    print("elmo_test_question1: point ", k)

elmo_test_question1 = tf.stack(elmo_test_question1)
print("Done Elmo_test_question1",elmo_test_question1.shape)

#test_question2
k=0
elmo_test_question2=[]
for x in X_v['question2']:
    d = []

    x = text_to_word_list("eng", x)
    d.append(x)
    leng = [len(d[0])]

    print("text is", x)
    # f= elmo([x], signature="default", as_dict=True)["elmo"]   #ater fixing it dontforget to change it to the main dataset
    # print ("before",f)
    f = elmo(
        inputs={
            "tokens": d,
            "sequence_len": leng
        },
        signature="tokens",
        as_dict=True)["elmo"]
    print("emdeding of text", f)

    paddingCons = int(f.shape[1])
    paddingCons = max_seq_length - paddingCons
    paddings = tf.constant([[0, 0], [0, paddingCons], [0, 0]])
    f = tf.pad(f, paddings, "CONSTANT")
    elmo_test_question2.append(f[0])
    k = k + 1
    print("elmo_test_question2: point ", k)

elmo_test_question2 = tf.stack(elmo_test_question2)
print("Done Elmo_test_question2",elmo_test_question2.shape)





Y_train = Y_train.values
Y_validation = Y_validation.values

assert X['question1'].shape == X['question2'].shape
assert len(X['question1']) == len(Y_train)

# class ElmoEmbeddingLayer ( Layer ):
#     def __init__(self, **kwargs):
#         self.dimensions = 1024
#         self.trainable=True
#         super(ElmoEmbeddingLayer, self).__init__(**kwargs)
#
#     def build(self, input_shape):
#         self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable,
#                                name="{}_module".format(self.name))
#
#         self.trainable_weights += K.tf.trainable_variables(scope="^{}_module/.*".format(self.name))
#         super(ElmoEmbeddingLayer, self).build(input_shape)
#
#     def call(self, x, mask=None):
#         result = self.elmo(K.squeeze(K.cast(x, tf.string), axis=1),
#                       as_dict=True,
#                       signature='default',
#                       )['default']
#         return result
#
#     def compute_mask(self, inputs, mask=None):
#         return K.not_equal(inputs, '--PAD--')
#
#     def compute_output_shape(self, input_shape):
#         return (input_shape[0], self.dimensions)


def shared_model_Elmo(_input):



    # print('this is size of the first embedding layer', embedded_sequences.shape())
    # out = Lambda(lambda x: x[0])(_input)
    print("Embedding Layer")
    l_lstm = Bidirectional(LSTM(100, return_sequences=True))(_input)
    print("Bi-Directional LSTM Layer")
    l_dense = TimeDistributed(Dense(200))(l_lstm)

    l_att = AttentionLayer()(l_dense)
    print("Attetion Layer")


    return l_att



if __name__ == '__main__':
 
    batch_size = 1
    n_epoch = 110
    n_hidden = 50
    left_input = Input(shape=(max_seq_length,1024,), dtype="float32",name="Input_layer")

    # left_input = Input(shape=(max_sequence_length,), dtype='float32')
    # print('this is left inout', left_input)
    right_input = Input(shape=(max_seq_length,1024,), dtype="float32")
    # print('this is right inout', right_input)
    # right_input = Input(shape=(1,), dtype='float32')
    left_sen_representation = shared_model_Elmo(left_input)
    # print('left snetcen presentation', left_sen_representation)
    right_sen_representation = shared_model_Elmo(right_input)


    man_distance = ManDist()([left_sen_representation, right_sen_representation])
    sen_representation = concatenate([left_sen_representation, right_sen_representation, man_distance])
    similarity = Dense(1, activation='sigmoid')(Dense(2)(Dense(4)(Dense(16)(sen_representation))))
    model = Model(inputs=[left_input, right_input], outputs=[similarity])

    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    model.summary()

    training_start_time = time()
    

    malstm_trained = model.fit([elmo_train_question1, elmo_train_question2], Y_train,
                               steps_per_epoch=batch_size, epochs=n_epoch,
                               validation_data=(
                                   [elmo_test_question1, elmo_test_question2], Y_validation), validation_steps=1)
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
    plt.savefig('C:/Users/dina_/Desktop/Embeddings results/Elmo embeddings/history-graph.png')

    model.save(savepath)
    print(str(malstm_trained.history['val_acc'][-1])[:6] +
          "(max: " + str(max(malstm_trained.history['val_acc']))[:6] + ")")
    print("Done.")


