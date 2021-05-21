
						BERT BiLSTM-Attention Similarity Model

This respository contains the source codes for the experiments of the Bert-BiLSTM Attention Similarity Model (BBASM). There is also the requirements.txt that contains the versions of all the packages needed to run the files. This model uses BERT as an embedding layer to convert the questions to their respective embedding, and uses BiLSTM-Attention for feature extraction, giving more weight to important parts in the embeddings. 
The function of one over the exponential function of the Manhattan distance is used to calculate the semantic similarity score. The model achieves an accuracy  of 84.45% in determining whether two questions from the Quora duplicate dataset are similar or not.

There are three main experiments that lead to this similarity model as shown below. 

I- Exploration of different word embeddings

Experiment Set Up: In order to allow comparison, the
training and testing for all three embedding approaches were
done on the same datasets as the HBAM model discussed
earlier. This is basically a subset from the Quora duplicate
dataset consisting of 10,000 questions. The dataset was divided
9:1 for training and testing. For the FastText, a pretrained
model called wiki news that was trained on Wikipedia 2017
was used. Regarding ELMo, the pretrained model was available
on TensorflowHub. Finally, the pretrained model used for
BERT embeddings was Uncased L-12 H-768 A-12 (12-layer,
768-hidden, 12-heads, 110M parameters) , which was accessed
using BERT as a service. The embedding dimensions for each
of the models were FastText: 300, ELMo: 1024, Word2vec:
300 and Bert is 768. The max sequence length for each of
the models were FastText: 10, ELMo: 72, Word2vec: 10 and
Bert: 25. After tuning, the number of epochs used in each of
the embeddings were as follows: FastText: 9, ELMo: 110, and
BERT: 100.

1) train_Elmo.py: uses Elmo embeddings followed by BiLSTM and Attention layer as feature extraction.
2) train_FastText.py: uses FastText embeddings followed by BiLSTM and Attention layer as feature extraction.
3) train_BERT.py: uses BERT embeddings followed by BiLSTM and Attention layer as feature extraction.
4) train_word2vec.py: uses word2vec embeddings followed by BiLSTM and Attention layer as feature extraction. (HABM)

Results

Embedding Layer                     Accuracy
HBAM(Word2Vec)                      81.2%
FastText                            81.44%
Elmo                                72.5%
BERT                                84.45%






II- Determining best feature extractor for text similarity model

Experimental Setup: The training and testing were
done on the same Quora Duplicate dataset used in HBAM to
allow comparison. The dataset was divided 9:1 for training
and testing. The training section was further divided 8:2
for training and validation. The embedding layer remained
constant to enable comparison (Word2Vec- Google’s word2vec
pretrained model). The hyperparameters used in both were as
follows: n epochs=9, max seq length=10.
Both the HBAM and the BiGRU+Attention ran on a machine
with the following specifications to enable the speed
comparison as well:
* 2.2 GHz Intel Core i7-8750H Six-Core
* 8GB of DDR4 RAM — 1TB HDD + 128GB SSD
* NVIDIA GeForce 1050 Ti (4GB GDDR5)

1) train_word2vec.py: uses word2vec embeddings followed by BiLSTM and Attention layer as feature extraction. (HABM)
2) train_BIGRU.py: uses word2vec embeddings followed by BiGRU and Attention layer as feature extraction. 


Results 

Feature Extractor                  Evaluation Accuracy     Speed of Training (Seconds for 9 epochs) 
Bi-GRU+Attention                   82%                      39.8
Bi-LSTM+Attention                  81.2%                    50





III- Determing whether BERT-BiGRU or BERT-BiLSTM is more accurate (BERT BiLSTM Attention Similarity Model (BBASM) )

Experimental Setup: Based on the aforementioned
two experiments, two conclusions were reached. The first one
is using BERT as the embedding layer which is then fed to
the BiLSTM-Attention layer for feature extraction yields a
higher accuracy compared to the other mentioned embedding
techniques. The second conclusion is using BiGRU-Attention
as feature extractor, compared to BiLSTM-Attention, results
in faster training and slight improvement in the accuracy.
Based on these conclusions, our initial hypothesis was that
a similarity model that incorporates BERT as an embedding
layer, followed by BiGRU-attention for feature extraction,
should yield a higher accuracy compared to HBAM and
BERT+BiLSTM similarity models. In this experiment, the
aforementioned hypothesis was tested. The training and testing
were done on the same Quora Duplicate dataset used in
HBAM to facilitate comparison. The dataset was divided 9:1
for training and testing.

1) BERT_BiGRU.py : uses BERT embeddings followed by BiGRU and Attention layer as feature extraction. 
2) train_BERT.py: uses BERT embeddings followed by BiLSTM and Attention layer as feature extraction.

Results

Model Used        Accuracy       Precision     Recall      F1 Score
HBAM              81.2%          78.87%        84.93%      81.79%
BERT-BiGRU        83.3%          86.37%        77.02%      81.43%
BBASM             84.45%         81.96%        87.44%      84.61%