import pandas as pd
import functools as ft


langs = ["en", "fr", "it"]
path_to_data = "data/train/extracted_data_"
path_to_vectors = "data/train/vectorized_data_"

# en
data_en = pd.read_csv(path_to_data + "en" + ".csv")
df_en = pd.read_csv(path_to_vectors + "en" + ".csv")
df_en['Party'] = data_en['Party']

# it
data_it = pd.read_csv(path_to_data + "it" + ".csv")
df_it = pd.read_csv(path_to_vectors + "it" + ".csv")
df_it['Party'] = data_it['Party']

# fr
data_fr = pd.read_csv(path_to_data + "fr" + ".csv")
df_fr = pd.read_csv(path_to_vectors + "fr" + ".csv")
df_fr['Party'] = data_fr['Party']

# merge
df_multi = [df_en, df_fr, df_it]
df_final = pd.concat(df_multi, axis=0)

df_final.to_csv("data/train/multilingual_corpora.csv")
