import os
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

papers = os.listdir('papers/')
papers = [name for name in papers if name != '.DS_Store']

# Removing .pdf
ids = [name[:-4] for name in papers]
df = pd.DataFrame({"id": ids})

def get_url(pid):
    return 'https://arxiv.org/abs/{}'.format(pid)

df['title'] = ''

for i in tqdm(range(len(df))):
    # We make too many requests and eventually will get disconnected 
    # This condition will allow us to continue where we left
    if df.title.iloc[i] == '':
        pid = df.id.iloc[i]
        purl = get_url(pid)

        req = requests.get(purl)

        if req.status_code == 200:
            soup = BeautifulSoup(req.text, "lxml")

            # Extract abstract
            title = soup.find('h1', {'class': 'title'})

            # Remove Title:
            title = title.text[7:]

            df.title.iloc[i] = title

df.to_csv('abstracts.csv', header=True, index=False, sep="\t")

