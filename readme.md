BERT BiLSTM-Attention Similarity Model
======================================


This respository contains the source codes for the experiments of the Bert-BiLSTM Attention Similarity Model (BBASM). There is also the requirements.txt that contains the versions of all the packages needed to run the files. This model uses BERT as an embedding layer to convert the questions to their respective embedding, and uses BiLSTM-Attention for feature extraction, giving more weight to important parts in the embeddings.  The function of one over the exponential function of the Manhattan distance is used to calculate the semantic similarity score. The model achieves an accuracy  of 84.45% in determining whether two questions from the Quora duplicate dataset are similar or not.

A- Experiments
--------------

