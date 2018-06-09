import pandas as pd
from tika import parser
import re


# Major file for dataset generation: Parsing and Cleaning PDF

papers_dir = '/home/maria/Documents/Courses_UCU/ML/Course_work/stat.ML-2K/papers/'
abstracts_dir = '/home/maria/Documents/Courses_UCU/ML/Course_work/abstracts-2K.csv'

abstracts_df = pd.read_csv(abstracts_dir, delimiter='\t')
abstracts_df['text'] = 0  # initialize

with open('./stopwords.txt', "r") as f:
    stopwords = f.read().split('\n')

for i, paper in abstracts_df.iterrows():
    try:
        paper_name = paper['id']
        paper_abstract = paper['abstract']

        parsedPDF = parser.from_file(papers_dir+paper_name+'.pdf')
        pdf = ' '.join(parsedPDF['content'].split())
        words, word = [], []
        for st in pdf:
            word.append(st)
            if st == ' ':
                words.append(''.join(word))
                word = []
        text = ''.join(words)

        # Delete abstract
        paper_abstract = paper_abstract.lower()
        text = text.lower()
        abs_start = text.find(paper_abstract[:30])
        abs_end = text.find(paper_abstract[-30:])+ 28
        txt = text[:abs_start]+text[abs_end:]

        # Get rid of ending (acknowledgments, reference)
        ending = []
        try:
            wor = txt.split(' ')
            for st in range(len(wor)):
                # or 'references' in wor[st].lower() \
                if st > len(wor)/1.5 and 'acknowledgments' in wor[st].lower() \
                        or 'acknowledgements' in wor[st].lower() :
                    ending.append(st)
            txt = ' '.join(wor[:min(ending)])
        except ValueError:
            pass

        # Delete all non english signs
        text_words = re.sub("[^a-zA-Z .']", '', txt).split()
        text_words_clean = []
        for w_i, w in enumerate(text_words):
            if len(w) <= 2 and w.lower() not in stopwords:
                pass
            else:
                text_words_clean.append(w)

        abstracts_df['text'].iloc[i] = ' '.join(text_words_clean)
        print("Iteration:", i, "  paper: ", paper_name)
    except:
         abstracts_df.set_value(i, 'text', None)

print(abstracts_df.shape)
abstracts_df.to_csv('./dataset.csv', index=False)
