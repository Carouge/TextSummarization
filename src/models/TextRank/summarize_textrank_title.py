from gensim.summarization.summarizer import summarize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from rouge import Rouge
import pandas as pd


def summarize_doc(content, len_words):
    """
    Summarization of one document using TextRank-based model from gensim library.
    :param content: text in string format
    :param len_words: number of words to 'generate', output summary size
    :return: summary and list of cleaned words from summary
    """
    summarized = summarize(content,  word_count=len_words)
    words = summarized.split(' ')
    tokenizer = RegexpTokenizer(r'\w+')
    filtered_words = [word for word in words if word not in stopwords.words('english')]
    filtered_words = tokenizer.tokenize(' '.join(filtered_words))
    return summarized, filtered_words

dataset = pd.read_csv('../../../data/papers-2K.csv')
testrank_res = pd.DataFrame(dataset['id'])
testrank_res['BLEU_unfilter'], testrank_res['ROUGE2_f_unfilter'], testrank_res['ROUGE1_f_unfilter'], testrank_res['ROUGE1_p_unfilter'], testrank_res['ROUGE2_p_unfilter'] = None, None, None, None, None
testrank_res['Summary'] = None
testrank_res['ROUGEl_p_unfilter'],testrank_res['ROUGEl_f_unfilter'] = None, None

for index, paper in dataset.iterrows():
    try:
        content = paper['abstract']

        num_words = len(paper['title'].split(' '))
        sum_text, filtered_words = summarize_doc(content, num_words)

        rouge_unfilter = Rouge()
        rouge_score_unfilter = rouge_unfilter.get_scores(sum_text, paper['title'])

        # print(len(sum_text), len(paper['abstract']), len(' '.join(filtered_abstract)), len(' '.join(filtered_words)))
        testrank_res['ROUGE2_f_unfilter'].iloc[index] = rouge_score_unfilter[0]['rouge-2']['f']
        testrank_res['ROUGE1_f_unfilter'].iloc[index] = rouge_score_unfilter[0]['rouge-1']['f']
        testrank_res['ROUGEl_f_unfilter'].iloc[index] = rouge_score_unfilter[0]['rouge-l']['f']
        testrank_res['Summary'].iloc[index] = ' '.join(sum_text)

        print("Iteration: ", index)
    except:
        pass

print(testrank_res.head(5))
testrank_res.to_csv('textrank_scores_title.csv', index=False)
