from rake_nltk import Rake
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge

# Rapid Automatic Keyword Extraction algorithm
# Summarization of one document using RAKE and calculating BLEU & ROUGE scores


def summarize_doc(content, length):
    r = Rake()
    r.extract_keywords_from_text(content)
    # summarized = r.get_ranked_phrases_with_scores()
    summarized = ' '.join(r.get_ranked_phrases()).split(' ')[:length]
    return summarized


dataset = pd.read_csv('../../../data/papers-2K.csv')
ind = 9   # paper id, position in dataset
content = dataset['text'][ind]
abstract = dataset['abstract'][ind].split()

sum_text = summarize_doc(content, len(abstract))

print(len(sum_text))
print(len(abstract))

bleu_score = sentence_bleu(sum_text, abstract)
rouge = Rouge()
rouge_score = rouge.get_scores(' '.join(sum_text), ' '.join(abstract))

print(bleu_score, rouge_score)
print(dataset['abstract'][ind])
print("\n")
print(" ".join(sum_text))
