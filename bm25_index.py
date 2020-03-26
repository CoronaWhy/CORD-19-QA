"""
Sets up a BM25 index over the abstracts and results of the data.
"""

# data loading and storage
import pandas as pd
import numpy as np
import os
from pathlib import Path
import json
from  copy import deepcopy
import pickle

# preprocessing
import string
import nltk

# model
from rank_bm25 import BM25Okapi

# other
from tqdm import tqdm
tqdm.pandas()

from argparse import ArgumentParser

PUNCTUATION_REMOVER = str.maketrans('', '', string.punctuation)
STOPWORDS = set(nltk.corpus.stopwords.words('english'))

def format_name(author):
    """
    Formats the author's name from a JSON file in the CORD-19 dataset.
    """
    middle_name = " ".join(author['middle'])

    if author['middle']:
        return " ".join([author['first'], middle_name, author['last']])
    else:
        return " ".join([author['first'], author['last']])


def format_affiliation(affiliation):
    """
    Formats the paper affiliations from a JSON file in the CORD-19 dataset.
    """
    text = []
    location = affiliation.get('location')
    if location:
        text.extend(list(affiliation['location'].values()))

    institution = affiliation.get('institution')
    if institution:
        text = [institution] + text
    return ", ".join(text)

def format_authors(authors, with_affiliation=False):
    """
    Formats the paper authors from a JSON file in the CORD-19 dataset.
    """
    name_ls = []

    for author in authors:
        name = format_name(author)
        if with_affiliation:
            affiliation = format_affiliation(author['affiliation'])
            if affiliation:
                name_ls.append(f"{name} ({affiliation})")
            else:
                name_ls.append(name)
        else:
            name_ls.append(name)

    return ", ".join(name_ls)

def format_body(body_text):
    """
    Formats the body of the paper from the JSON file.
    """
    texts = [(di['section'], di['text']) for di in body_text]
    texts_di = {di['section']: "" for di in body_text}

    for section, text in texts:
        texts_di[section] += text

    body = ""

    for section, text in texts_di.items():
        body += section
        body += "\n\n"
        body += text
        body += "\n\n"
    return body

def paragraphize_body(body_text):
    paragraphs = [di['text'] for di in body_text if len(di['text'].split()) > 1]
    return paragraphs

def format_bib(bibs):
    """
    Formats the bibliography from the JSON file.
    """
    if type(bibs) == dict:
        bibs = list(bibs.values())
    bibs = deepcopy(bibs)
    formatted = []

    for bib in bibs:
        bib['authors'] = format_authors(
            bib['authors'],
            with_affiliation=False
        )
        formatted_ls = [str(bib[k]) for k in ['title', 'authors', 'venue', 'year']]
        formatted.append(", ".join(formatted_ls))

    return "; ".join(formatted)

def load_files(dirname):
    """
    Loads all JSON files recurisvely in a directory. Returns a list of JSONs.
    """
    raw_files = []
    for path in tqdm(list(Path(dirname).rglob('*.json'))):
        file = json.load(open(path, 'rb'))
        raw_files.append(file)
    return raw_files

def generate_clean_df(all_files, paragraphs=True):
    """
    Generates a Pandas DataFrame from the raw file data created by load_files().
    """
    cleaned_files = []

    for file in tqdm(all_files):
        features = [
            file['paper_id'],
            file['metadata']['title'],
            format_authors(file['metadata']['authors']),
            format_authors(file['metadata']['authors'],
                           with_affiliation=True),
            format_body(file['abstract']),
            # format_bib(file['bib_entries']),
            # file['metadata']['authors'],
            # file['bib_entries']
        ]

        if paragraphs:
            all_paragraphs = paragraphize_body(file['body_text'])
            for p in all_paragraphs:
                new_features = features.copy()
                new_features.append(p)
                cleaned_files.append(new_features)
        else:
            features.append(format_body(file['body_text']))
            cleaned_files.append(features)

    col_names = ['paper_id', 'title', 'authors',
                 'affiliations', 'abstract', 'text']
                 # 'bibliography','raw_authors','raw_bibliography']

    clean_df = pd.DataFrame(cleaned_files, columns=col_names)
    clean_df['abstract'] = clean_df['abstract'].fillna(clean_df['text']) # fall back to full-text
    clean_df = clean_df.drop_duplicates(subset='title')
    return clean_df

class BM25Index:
    def __init__(self, df, ngram_length=1):
        self.data = df
        self.ngram_length = ngram_length
        self.clean_data = df.abstract.progress_apply(lambda x: clean_text(x, ngram_length)).tolist()
        self.index = BM25Okapi(self.clean_data)

    def search(self, query, k=10):
        processed = clean_text(query, self.ngram_length)
        doc_scores = self.index.get_scores(processed)
        ind = np.argsort(doc_scores)[::-1][:k]
        results = self.data.iloc[ind].copy()
        results['score'] = doc_scores[ind]
        return results

def format_ngram(ngram):
    return '((' + ','.join(ngram) + '))'

def clean_text(text, ngram_length):
    uncased = text.translate(PUNCTUATION_REMOVER).lower()
    tokens = [token for token in nltk.word_tokenize(uncased)
                if len(token) > 1
                and not token in STOPWORDS
                and not (token.isnumeric() and len(token) != 4)
                and (not token.isnumeric() or token.isalpha())]
    out = tokens.copy()
    num_tokens = len(tokens)
    ngram_length = min(ngram_length, num_tokens)
    for n in range(1, ngram_length):
        ngrams = [format_ngram(tokens[i:i+n]) for i in range(num_tokens - n)]
        out.extend(ngrams)
    return out

if __name__ == '__main__':
    psr = ArgumentParser()
    psr.add_argument("--data-dir", type=str, default="./data/")
    psr.add_argument("--rebuild-index", action='store_true')
    psr.add_argument("--index-path", type=str, default="bm25.pkl")
    psr.add_argument("--result-path-base", type=str, default="results/query")
    psr.add_argument("--query", type=str, default="cruise ship")
    psr.add_argument("--nresults", type=int, default=5)
    psr.add_argument("--ngram-length", type=int, default=1)
    psr.add_argument("--paragraphs", action='store_true')
    args = psr.parse_args()

    # if args.rebuild_index or not os.path.isfile(args.index_path):
    files = load_files(args.data_dir)
    print("Loaded {} files".format(len(files)))
    df = generate_clean_df(files, paragraphs=args.paragraphs)
    search_idx = BM25Index(df, args.ngram_length)
    print("Caching index...")
    pickle.dump(search_idx, open(args.index_path, "wb"))
    # else:
    #     print("Loading cached index...")
    #     search_idx = pickle.load(open(args.index_path, "rb"))
    # results = search_idx.search(args.query)
    # print(results[['title','score']])
    # results.to_csv("_".join([args.result_path_base, args.query.replace(" ","_"), "top{}".format(args.nresults)]) + ".csv", index=False)
