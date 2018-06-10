from rake_nltk import Rake
from rouge import Rouge
import pandas as pd


def summarize_doc(content, length):
    """
    Summarization of one document using RAKE model
    :param content: text in string format
    :param len_words: length of sequence to 'generate', output summary size
    :return: summary and list of cleaned words from summary
    """
    r = Rake()
    r.extract_keywords_from_text(content)
    # summarized = r.get_ranked_phrases_with_scores()
    summarized = ' '.join(r.get_ranked_phrases()).split(' ')[:length]
    return summarized

dataset = pd.read_csv('../../../data/papers-2K.csv')
rake_res = pd.DataFrame(dataset['id'])
rake_res['ROUGE2_f_unfilter'], rake_res['ROUGE1_f_unfilter'], rake_res['ROUGE1_p_unfilter'], rake_res['ROUGE2_p_unfilter'] = None, None, None, None
rake_res['Summary'], rake_res['ROUGEl_p_unfilter'], rake_res['ROUGEl_f_unfilter'] = None, None, None

for index, paper in dataset.iterrows():
    try:
        content = paper['text']
        if len(content) < len(paper['abstract']) * 3:
            print("Too small text for paper", paper['id'], index)
            raise ValueError
        # ratio = round(len(paper['abstract'])/len(content), 3)
        num_words = len(paper['abstract'].split(' '))
        sum_text  = summarize_doc(content, num_words)

        abstract = paper['abstract'].split()

        rouge_unfilter = Rouge()
        rouge_score_unfilter = rouge_unfilter.get_scores(' '.join(sum_text), paper['abstract'])
        rake_res['Summary'].iloc[index] = ' '.join(sum_text)
        rake_res['ROUGE2_f_unfilter'].iloc[index] = rouge_score_unfilter[0]['rouge-2']['f']
        rake_res['ROUGE1_f_unfilter'].iloc[index] = rouge_score_unfilter[0]['rouge-1']['f']
        rake_res['ROUGEl_f_unfilter'].iloc[index] = rouge_score_unfilter[0]['rouge-l']['f']

        print("Iteration: ", index)
    except:
        pass

print(rake_res.head(5))
rake_res.to_csv('rake_scores.csv', index=False)
