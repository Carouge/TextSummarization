from gensim.summarization.summarizer import summarize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.translate.bleu_score import sentence_bleu
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
testrank_res['BLEU'], testrank_res['ROUGE2_f'], testrank_res['ROUGE1_f'], testrank_res['ROUGE1_p'], testrank_res['ROUGE2_p'] = None, None, None, None, None
testrank_res['BLEU_unfilter'], testrank_res['ROUGE2_f_unfilter'], testrank_res['ROUGE1_f_unfilter'], testrank_res['ROUGE1_p_unfilter'], testrank_res['ROUGE2_p_unfilter'] = None, None, None, None, None
testrank_res['Summary'] = None
testrank_res['ROUGEl_p_unfilter'], testrank_res['ROUGEl_p'], testrank_res['ROUGEl_f'], testrank_res['ROUGEl_f_unfilter'] = None, None, None, None

for index, paper in dataset.iterrows():
    try:
        content = paper['text']
        if len(content) < len(paper['abstract']) * 3:
            print("Too small text for paper", paper['id'], index)
            raise ValueError
        # ratio = round(len(paper['abstract'])/len(content), 3)
        num_words = len(paper['abstract'].split(' '))
        sum_text, filtered_words = summarize_doc(content, num_words)

        abstract = paper['abstract'].split()
        tokenizer = RegexpTokenizer(r'\w+')
        filtered_abstract = [word for word in abstract if word not in stopwords.words('english')]
        filtered_abstract = tokenizer.tokenize(' '.join(filtered_abstract))

        bleu_score = sentence_bleu(filtered_abstract, filtered_words)
        rouge = Rouge()
        rouge_score = rouge.get_scores(' '.join(filtered_words), ' '.join(filtered_abstract))

        # print(len(sum_text), len(paper['abstract']), len(' '.join(filtered_abstract)), len(' '.join(filtered_words)))
        testrank_res['Summary'].iloc[index] = sum_text
        testrank_res['BLEU'].iloc[index] = bleu_score
        testrank_res['ROUGE2_f'].iloc[index] = rouge_score[0]['rouge-2']['f']
        testrank_res['ROUGE1_f'].iloc[index] = rouge_score[0]['rouge-1']['f']
        testrank_res['ROUGE2_p'].iloc[index] = rouge_score[0]['rouge-2']['p']
        testrank_res['ROUGE1_p'].iloc[index] = rouge_score[0]['rouge-1']['p']

        # Score on not cleaned text
        bleu_score_unfilter = sentence_bleu(sum_text.split(' '), abstract)
        rouge_unfilter = Rouge()
        rouge_score_unfilter = rouge_unfilter.get_scores(sum_text, paper['abstract'])

        # print(len(sum_text), len(paper['abstract']), len(' '.join(filtered_abstract)), len(' '.join(filtered_words)))
        testrank_res['BLEU_unfilter'].iloc[index] = bleu_score_unfilter
        testrank_res['ROUGE2_f_unfilter'].iloc[index] = rouge_score_unfilter[0]['rouge-2']['f']
        testrank_res['ROUGE1_f_unfilter'].iloc[index] = rouge_score_unfilter[0]['rouge-1']['f']
        testrank_res['ROUGE2_p_unfilter'].iloc[index] = rouge_score_unfilter[0]['rouge-2']['p']
        testrank_res['ROUGE1_p_unfilter'].iloc[index] = rouge_score_unfilter[0]['rouge-1']['p']

        testrank_res['ROUGEl_p_unfilter'].iloc[index] = rouge_score_unfilter[0]['rouge-l']['p']
        testrank_res['ROUGEl_p'].iloc[index] = rouge_score[0]['rouge-l']['p']
        testrank_res['ROUGEl_f'].iloc[index] = rouge_score[0]['rouge-l']['f']
        testrank_res['ROUGEl_f_unfilter'].iloc[index] = rouge_score_unfilter[0]['rouge-l']['f']

        print("Iteration: ", index)
    except:
        pass

print(testrank_res.head(5))
testrank_res.to_csv('textrank_scores.csv', index=False)
