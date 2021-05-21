This respository contains the source codes for the experiments of the Bert-BiLSTM Attention Similarity Model (BBASM).
There are three main experiments that lead to this similarity model.

The first is determining the best embedding for a similarity model:
1) train_Elmo.py: uses Elmo embeddings followed by BiLSTM and Attention layer as feature extraction.
2) train_FastText.py: uses FastText embeddings followed by BiLSTM and Attention layer as feature extraction.
3) train_BERT.py: uses BERT embeddings followed by BiLSTM and Attention layer as feature extraction.
4) train_word2vec.py: uses word2vec embeddings followed by BiLSTM and Attention layer as feature extraction. (HABM)


The second is determining the best feature extraction for a similarity model:
1) train_word2vec.py: uses word2vec embeddings followed by BiLSTM and Attention layer as feature extraction. (HABM)
2) train_BIGRU.py: uses word2vec embeddings followed by BiGRU and Attention layer as feature extraction. 

The third is determing whether BERT-BiGRU or BERT-BiLSTM is more accurate:
1) BERT_BiGRU.py : uses BERT embeddings followed by BiGRU and Attention layer as feature extraction. 
2) train_BERT.py: uses BERT embeddings followed by BiLSTM and Attention layer as feature extraction.
