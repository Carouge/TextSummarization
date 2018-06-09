import pandas as pd
import sys, os

file_dir = '/home/maria/Documents/Courses_UCU/ML/Course_work/abstracts-2K.csv'
directory = './abstracts_txt/'
if not os.path.exists(directory):
    os.makedirs(directory)

all_papers = pd.read_csv(file_dir, delimiter='\t')

# This representation of dataset's abstracts is needed for SeqGAN
for i, paper in all_papers.iterrows():
    with open(directory + paper['id'] + '.txt', "w") as f:
        abstract = paper['abstract']

        abstract = [abstract[i:i+50] for i in range(0, len(abstract), 50)]
        f.write('\n'.join(abstract))

print("Done!")
