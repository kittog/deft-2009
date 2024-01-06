import numpy as np
import xml.etree.ElementTree as ET
import os
import re
from collections import defaultdict
import pandas as pd
def get_data(path, extensions=['.xml', '.txt']):
    '''Récupérer les listes des fichiers xml et txt'''
    file_list = []
    for dirpath, dirs, files in os.walk(path):
        for filename in files:
            name = os.path.join(dirpath, filename)
            if any(name.endswith(ext) for ext in extensions):
                file_list.append(name)
    return file_list
    
def tokenizer_and_normalizer(s):
    return [match.lower() for match in re.findall(r"\b\w+?\b(?:'|(?:-\w+?\b)*)?", s)]
    

def parse_xml_train(file, dico):
    '''Parser des fichiers .xml d'entraînement.'''
    tree = ET.parse(file)
    root = tree.getroot()
    langue = file.split('_')[-1][:2]

    for i, doc in enumerate(root.findall('.//doc')):
        doc_ide = (i+1,'corpus'+  langue)
        dico[doc_ide] = {partie: [] for partie in set(partie.get('valeur') for partie in doc.findall('.//PARTI'))}
        
        texte = doc.find('.//texte')

        if texte is not None:
            for content in texte:
                if content.text is not None:
                    for partie in dico[doc_ide]:
                        dico[doc_ide][partie] += tokenizer_and_normalizer(content.text)

    return dico

def parse_xml_test(file,dico):
    '''Parser des fichiers .xml et .txt de test.'''
    tree = ET.parse(file)
    root = tree.getroot()
    langue = file.split('_')[-1][:2]
    ref_file = f'deft09/Données de référence/deft09_parlement_ref_{langue}.txt'

    dico = {}

    for doc in root.findall('.//doc'):
        doc_id = doc.get('id')
        doc_ide = (doc_id, 'corpus ' + langue)

        with open(ref_file, 'r', encoding='UTF-8') as ref:
            matching_lines = [line.split() for line in ref if line.split()[0] == doc_id]
            party = matching_lines[0][1].strip() if matching_lines and len(matching_lines[0]) > 1 else None


        party_list = dico.setdefault(doc_ide, {}).setdefault(party, [])

        texte = doc.find('.//texte')
        if texte is not None:
            for content in texte:
                if content.text is not None:
                    party_list.extend(tokenizer_and_normalizer(content.text))
    return dico
def create_df(dico, output_file):
    '''Transformation du dictionnaire en dataframe.
    Sauvegarde en .csv.'''
    data = [{'document': doc_ide, 'parti politique': party, 'texte': text}
            for doc_ide, parties in dico.items()
            for party, text in parties.items()]

    df = pd.DataFrame(data)

    df.to_csv(output_file, index=False, encoding='UTF8')
    
def remove_empty_lines(file_path):
    with open(file_path, 'r', encoding='UTF8') as file:
        lines = file.readlines()

    with open(file_path, 'w', encoding='UTF8') as file:
        for line in lines:
            if line.strip():
                file.write(line)
def main():
    path_train = 'deft09/Corpus d_apprentissage'
    path_test = 'deft09/Corpus de test'

    train_files = get_data(path_train)
    test_files = get_data(path_test)

    dico_train = {}
    for file in train_files:
        dico_train.update(parse_xml_train(file, dico_train))
    create_df(dico_train, 'corpus/corpus_train.csv')
    remove_empty_lines('corpus/corpus_train.csv')

    dico_test = {}
    for file in test_files:
        dico_test.update(parse_xml_test(file, dico_test))
    create_df(dico_test, 'corpus/corpus_test.csv')
    remove_empty_lines('corpus/corpus_test.csv')

if __name__ == "__main__":
    main()
