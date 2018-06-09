import pandas as pd
import sys, os

file_dir = '/home/maria/Documents/Courses_UCU/ML/Course_work/data.csv'
directory = './texts_noabstract_txt/'

if not os.path.exists(directory):
    os.makedirs(directory)

all_papers = pd.read_csv(file_dir)

# This representation of dataset is needed for SeqGAN
# Creating folder with name {directory} with txt files. Each file - parsed paper in '50 chars per row' format.
for i, paper in all_papers.iterrows():
    with open('./abstracts_txt/'+ paper['name'][:-4]+'.txt') as abs:
        abstract = ' '.join(abs.read().split('\n'))
        # print(abstract)
    with open(directory + paper['name'][:-4] + '.txt', "w") as f:
        text = paper['text']
        paper = text # + abstract    # if we need both abstract and text
        paper = [paper[i:i+50] for i in range(0, len(paper), 50)]
        f.write('\n'.join(paper))

print("Done!")
