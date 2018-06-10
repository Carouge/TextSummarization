"""
This model on based on the work of Jishnu Ray Chowdhury
Source: https://github.com/JRC1995/Abstractive-Summarization
"""
import numpy as np
from __future__ import division

filename = 'glove.6B.50d.txt' 
# (glove data set from: https://nlp.stanford.edu/projects/glove/)


def loadGloVe(filename):
    vocab = []
    embd = []
    file = open(filename,'r')
    for line in file.readlines():
        row = line.strip().split(' ')
        vocab.append(row[0])
        embd.append(row[1:])
    print('GloVe Loaded.')
    file.close()
    return vocab,embd

# Pre-trained GloVe embedding
vocab,embd = loadGloVe(filename)

embedding = np.asarray(embd)
embedding = embedding.astype(np.float32)

word_vec_dim = len(embd[0]) # word_vec_dim = dimension of each word vectors

import csv
import sys
import nltk as nlp
from nltk import word_tokenize
import string
from tqdm import tqdm_notebook as tqdm

csv.field_size_limit(sys.maxsize)

summaries = []
texts = []

def clean(text):
    text = text.lower()
    printable = set(string.printable)
    return filter(lambda x: x in printable, text) #filter funny characters, if any. 
    

with open('dataset.csv', 'rb') as csvfile: #Data from https://www.kaggle.com/snap/amazon-fine-food-reviews
    Reviews = csv.DictReader(csvfile)
    for row in tqdm(Reviews):
        clean_text = clean(row['abstract'])
        clean_summary = clean(row['title'])
        summaries.append(word_tokenize(clean_summary))
        texts.append(word_tokenize(clean_text))


import random

index = random.randint(0,len(texts)-1)

print "SAMPLE CLEANED & TOKENIZED TEXT: \n\n"+str(texts[index])
print "\nSAMPLE CLEANED & TOKENIZED SUMMARY: \n\n"+str(summaries[index])


def np_nearest_neighbour(x):
    #returns array in embedding that's most similar (in terms of cosine similarity) to x
        
    xdoty = np.multiply(embedding,x)
    xdoty = np.sum(xdoty,1)
    xlen = np.square(x)
    xlen = np.sum(xlen,0)
    xlen = np.sqrt(xlen)
    ylen = np.square(embedding)
    ylen = np.sum(ylen,1)
    ylen = np.sqrt(ylen)
    xlenylen = np.multiply(xlen,ylen)
    cosine_similarities = np.divide(xdoty,xlenylen)

    return embedding[np.argmax(cosine_similarities)]
    


def word2vec(word):  # converts a given word into its vector representation
    if word in vocab:
        return embedding[vocab.index(word)]
    else:
        return embedding[vocab.index('unk')]

def vec2word(vec):   # converts a given vector representation into the represented word 
    for x in xrange(0, len(embedding)):
            if np.array_equal(embedding[x],np.asarray(vec)):
                return vocab[x]
    return vec2word(np_nearest_neighbour(np.asarray(vec)))


word = "unk"
print "Vector representation of '"+str(word)+"':\n"
print word2vec(word)


#REDUCE DATA (FOR SPEEDING UP THE NEXT STEPS)

MAXIMUM_DATA_NUM = 2000

texts = texts[0:MAXIMUM_DATA_NUM]
summaries = summaries[0:MAXIMUM_DATA_NUM]


vocab_limit = []
embd_limit = []

i=0
for i in tqdm(range(len(texts))):
    text  = texts[i]
    for word in text:
        if word not in vocab_limit:
            if word in vocab:
                vocab_limit.append(word)
                embd_limit.append(word2vec(word))


for i in tqdm(range(len(summaries))):
    summary = summaries[i]
    for word in summary:
        if word not in vocab_limit:
            if word in vocab:
                vocab_limit.append(word)
                embd_limit.append(word2vec(word))


if 'eos' not in vocab_limit:
    vocab_limit.append('eos')
    embd_limit.append(word2vec('eos'))
if 'unk' not in vocab_limit:
    vocab_limit.append('unk')
    embd_limit.append(word2vec('unk'))

null_vector = np.zeros([word_vec_dim])

vocab_limit.append('<PAD>')
embd_limit.append(null_vector)    


vec_summaries = []

for summary in tqdm(summaries):
    
    vec_summary = []
    
    for word in summary:
        vec_summary.append(word2vec(word))
            
    vec_summary.append(word2vec('eos'))
    
    vec_summary = np.asarray(vec_summary)
    vec_summary = vec_summary.astype(np.float32)
    
    vec_summaries.append(vec_summary)


vec_texts = []

for i in tqdm(range(len(texts))):
    text = texts[i]
    vec_text = []
    
    for word in text:
        vec_text.append(word2vec(word))
    
    vec_text = np.asarray(vec_text)
    vec_text = vec_text.astype(np.float32)
    
    vec_texts.append(vec_text)    


#Saving processed data in another file.

import pickle
with open('vocab_limit', 'wb') as fp:
    pickle.dump(vocab_limit, fp)
with open('embd_limit', 'wb') as fp:
    pickle.dump(embd_limit, fp)
with open('vec_summaries', 'wb') as fp:
    pickle.dump(vec_summaries, fp)
with open('vec_texts', 'wb') as fp:
    pickle.dump(vec_texts, fp)
