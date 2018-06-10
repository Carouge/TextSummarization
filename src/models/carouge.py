from __future__ import division
import numpy as np
from nltk.tokenize import RegexpTokenizer
import scipy 

filename = 'glove.6B.50d.txt'
def loadGloVe(filename):
    vocab = []
    embd = []
    file = open(filename,'r')
    for line in file.readlines():
        row = line.strip().split(' ')
        vocab.append(row[0])
        embd.append(row[1:])
    print('Loaded GloVe!')
    file.close()
    return vocab,embd
vocab, embd = loadGloVe(filename)

embedding = np.asarray(embd)
embedding = embedding.astype(np.float32)

def word2vec(word):
    if word in vocab:
        return embedding[vocab.index(word)]
    else:
        return embedding[vocab.index('unk')]

cosine_distance_max = 2

def word_distance_normalized(vec1, vec2):
    return scipy.spatial.distance.cosine(vec1, vec2)/cosine_distance_max

def carouge(summary_generated, summary_actual):
    tokenizer = RegexpTokenizer(r'\w+')
    tokenized_summary_generated = tokenizer.tokenize(summary_generated)
    tokenized_summary_actual = tokenizer.tokenize(summary_actual)
    summary_generated_embeddings = [word2vec(word) for word in  tokenized_summary_generated ]
    summary_actual_embeddings = [word2vec(word) for word in  tokenized_summary_actual ]


    score = np.sum([ 
                        1 - np.min([
                                word_distance_normalized(act_word_emb, gen_word_emb)
                                for  act_word_emb in summary_actual_embeddings
                            ]) 

                        for gen_word_emb in summary_generated_embeddings
                   ])

    return score/len(summary_actual_embeddings)

