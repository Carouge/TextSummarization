from gensim.summarization.summarizer import summarize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge

# Summarization of one document using TextRank-based model from gensim library and calculating BLEU & ROUGE scores


def summarize_doc(content, len_words):
    summarized = summarize(content,  word_count=30)
    words = summarized.split(' ')
    tokenizer = RegexpTokenizer(r'\w+')
    filtered_words = [word for word in words if word not in stopwords.words('english')]
    filtered_words = tokenizer.tokenize(' '.join(filtered_words))
    return summarized, filtered_words

dataset = pd.read_csv('./dataset.csv')
ind = 5   # paper id, position in dataset
content = dataset['text'][ind]

num_words = len(dataset['abstract'][ind].split(' '))
sum_text, filtered_words = summarize_doc(content, num_words)

abstract = dataset['abstract'][ind].split()
tokenizer = RegexpTokenizer(r'\w+')
filtered_abstract = [word for word in abstract if word not in stopwords.words('english')]
filtered_abstract = tokenizer.tokenize(' '.join(filtered_abstract))


bleu_score = sentence_bleu(sum_text.split(' '), abstract)
rouge = Rouge()
rouge_score = rouge.get_scores(sum_text, ' '.join(abstract))

print(bleu_score, rouge_score)
print(dataset['abstract'][ind])
print("\n")
print(sum_text)
