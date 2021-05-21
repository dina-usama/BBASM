BERT BiLSTM-Attention Similarity Model
======================================


This respository contains the source codes for the experiments of the Bert-BiLSTM Attention Similarity Model (BBASM). There is also the requirements.txt that contains the versions of all the packages needed to run the files. This model uses BERT as an embedding layer to convert the questions to their respective embedding, and uses BiLSTM-Attention for feature extraction, giving more weight to important parts in the embeddings.  The function of one over the exponential function of the Manhattan distance is used to calculate the semantic similarity score. The model achieves an accuracy  of 84.45% in determining whether two questions from the Quora duplicate dataset are similar or not.

A- Experiments
--------------

I- Exploration of different word embeddings
--------------------------------------------

Experiment Set Up: In order to allow comparison, the training and testing for all three embedding approaches were done on the same datasets as the HBAM model discussed
earlier. This is basically a subset from the Quora duplicate dataset consisting of 10,000 questions. The dataset was divided 9:1 for training and testing. For the FastText, a pretrained model called wiki news that was trained on Wikipedia 2017 was used. Regarding ELMo, the pretrained model was available on TensorflowHub. Finally, the pretrained model used for BERT embeddings was Uncased L-12 H-768 A-12 (12-layer, 768-hidden, 12-heads, 110M parameters) , which was accessed using BERT as a service. The embedding dimensions for each of the models were FastText: 300, ELMo: 1024, Word2vec: 300 and Bert is 768. The max sequence length for each of the models were FastText: 10, ELMo: 72, Word2vec: 10 and Bert: 25. After tuning, the number of epochs used in each of the embeddings were as follows: FastText: 9, ELMo: 110, and BERT: 100.

1) train_Elmo.py: uses Elmo embeddings followed by BiLSTM and Attention layer as feature extraction.
2) train_FastText.py: uses FastText embeddings followed by BiLSTM and Attention layer as feature extraction.
3) train_BERT.py: uses BERT embeddings followed by BiLSTM and Attention layer as feature extraction.
4) train_word2vec.py: uses word2vec embeddings followed by BiLSTM and Attention layer as feature extraction. (HABM)

The results are summarized in the following table. 

|Embedding Layer| Accuracy|
| -----------   | --------|
|HBAM(Word2Vec)  |  81.2% |
| FastText      |  81.44% |
| Elmo     |   72.5% |
 | BERT      | 84.45% |

