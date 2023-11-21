import xml.etree.ElementTree as ET
import stopwordsiso
import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm

def process_txt(texts, lang):
    lm = {
        "en":"en_core_web_sm",
        "it":"it_core_news_sm",
        "fr":"fr_core_news_sm"
    }
    nlp = spacy.load(lm[lang])
    stopwords = stopwordsiso.stopwords(lang)
    clean_texts = []
    for x in tqdm(texts):
        clean_tokens = []
        doc = nlp(x)
        for token in doc:
            if token not in stopwords:
                clean_tokens.append(token.text)
        clean_texts.append(" ".join(clean_tokens))
    return clean_texts


def process_xml(input_xml, output_csv, output_tfidf_csv, lang, n_components=20):
    tree = ET.parse(input_xml)
    texts, parties = [], []

    for doc in tree.findall('.//doc'):
        text = ' '.join(p.text for p in doc.findall('.//texte//p') if p.text)
        party_value = doc.find('.//EVAL_PARTI/PARTI').get('valeur') if doc.find('.//EVAL_PARTI') else None

        if text and party_value:
            texts.append(text)
            parties.append(party_value)

    df = pd.DataFrame({'Text': texts, 'Party': parties})
    df["Clean"] = process_txt(texts, lang)
    df.to_csv(output_csv, index=False)

    tfidf_matrix_svd = TruncatedSVD(n_components=n_components).fit_transform(TfidfVectorizer().fit_transform(df["Clean"]))
    pd.DataFrame(tfidf_matrix_svd).to_csv(output_tfidf_csv, index=False)

def process_all_languages():
    languages = ['en', 'fr', 'it']

    for lang in languages:
        input_xml = f'deft09/Corpus d_apprentissage/deft09_parlement_appr_{lang}.xml'
        process_xml(input_xml, f'extracted_data_{lang}.csv', f'vectorized_data_{lang}.csv', lang)


process_all_languages()
