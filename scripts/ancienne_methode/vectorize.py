import xml.etree.ElementTree as ET
import stopwordsiso
import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm
import re

def tokenize_and_filter(texts, lang):
    lm = {
        "en":"en_core_web_sm",
        "it":"it_core_news_sm",
        "fr":"fr_core_news_sm"
    }
    nlp = spacy.load(lm[lang]) # language model
    stop_words = nlp.Defaults.stop_words
    filtered_docs = []
    for text in tqdm(texts):
        doc = nlp(text)
        # tokenize
        tokens = [word.lemma_ for word in doc]
        filtered_tokens = []
        # filter out raw punctuation, numbers etc
        for token in tokens:
            if re.search('[a-zA-Z]', token):
                filtered_tokens.append(token.lower())
        # filter out stopwords
        # stop_words = stopwordsiso.stopwords(lang)
        filtered_tokens = [token for token in filtered_tokens if token not in stop_words]
        filtered_docs.append(" ".join(filtered_tokens)) # returns text, filtered
    return filtered_docs


def process_xml(input_xml, output_csv, output_tfidf_csv, lang, n_components=20, input_df):
    tree = ET.parse(input_xml)
    texts = []
    # texts, parties = [], []

    for doc in tree.findall('.//doc'):
        text = ' '.join(p.text for p in doc.findall('.//texte//p') if p.text)
        # party_value = doc.find('.//EVAL_PARTI/PARTI').get('valeur') if doc.find('.//EVAL_PARTI') else None

        if text:
            texts.append(text)
        #if text and party_value:
        #    texts.append(text)
        #    parties.append(party_value)

    # df = pd.DataFrame({'Text': texts, 'Party': parties})
    df = pd.DataFrame({'Text':texts})
    df['Parties'] = input_df # series
    df['Clean'] = tokenize_and_filter(texts, lang)
    df.to_csv(output_csv, index=False)

    tfidf_matrix_svd = TruncatedSVD(n_components=n_components).fit_transform(
        TfidfVectorizer(
            max_df=0.8,
            min_df=0.1,
        ).fit_transform(df['Clean']))

    pd.DataFrame(tfidf_matrix_svd).to_csv(output_tfidf_csv, index=False)

def process_all_languages():
    languages = ['en', 'fr', 'it']

    for lang in languages:
        input_xml = f'deft09/Corpus de test/deft09_parlement_test_{lang}.xml'
        input_txt = f'deft09/Données de référence/deft09_parlement_ref_{lang}.txt'
        df_parties = pd.read_csv(input_txt, sep='\t', names=['id', 'parties'])
        process_xml(input_xml, f'ausecours/extracted_data_test_lemma_{lang}.csv', f'vectorized_data_test_tfidf_{lang}.csv', lang, df_parties)


process_all_languages()
