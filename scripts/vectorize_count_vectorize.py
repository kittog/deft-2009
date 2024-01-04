import xml.etree.ElementTree as ET
import stopwordsiso
import spacy
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm
import re

def process_csv(input_csv, output_tfidf_csv, lang, n_components=20):
    df = pd.read_csv(input_csv)

    tfidf_matrix_svd = TruncatedSVD(n_components=n_components).fit_transform(
        CountVectorizer().fit_transform(df['Clean']))

    pd.DataFrame(tfidf_matrix_svd).to_csv(output_tfidf_csv, index=False)

def process_all_languages():
    languages = ['en', 'fr', 'it']

    for lang in languages:
        input_csv = f'data/ausecours/extracted_data_train_lemma_{lang}.csv'
        # input_txt = f'deft09/Données de référence/deft09_parlement_ref_{lang}.txt'
        # df_parties = pd.read_csv(input_txt, sep='\t', names=['id', 'parties'])
        process_xml(input_csv, f'data/ausecours/vectorized_data_train_cvect_{lang}.csv', lang)


process_all_languages()
