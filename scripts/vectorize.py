import xml.etree.ElementTree as ET
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def process_xml(input_xml, output_csv, output_tfidf_csv, n_components=20):
    tree = ET.parse(input_xml)
    texts, parties = [], []

    for doc in tree.findall('.//doc'):
        text = ' '.join(p.text for p in doc.findall('.//texte//p') if p.text)
        party_value = doc.find('.//EVAL_PARTI/PARTI').get('valeur') if doc.find('.//EVAL_PARTI') else None

        if text and party_value:
            texts.append(text)
            parties.append(party_value)

    df = pd.DataFrame({'Text': texts, 'Party': parties})
    df.to_csv(output_csv, index=False)

    tfidf_matrix_svd = TruncatedSVD(n_components=n_components).fit_transform(TfidfVectorizer().fit_transform(texts))
    pd.DataFrame(tfidf_matrix_svd).to_csv(output_tfidf_csv, index=False)

def process_all_languages():
    languages = ['en', 'fr', 'it']

    for lang in languages:
        input_xml = f'deft09/Corpus d_apprentissage/deft09_parlement_appr_{lang}.xml'
        process_xml(input_xml, f'extracted_data_{lang}.csv', f'vectorized_data_{lang}.csv')

process_all_languages()
