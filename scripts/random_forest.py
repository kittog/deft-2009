#!/usr/bin/python

# importing modules
import click
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV

def clf_report(y_test, y_pred, lang):
    report = classification_report(y_test, y_pred,output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(f"classification_report_{lang}.csv")

def conf_matrix(y_test, y_pred, lang):
    labels = ["ELDR", "GUE-NGL", "PPE-DE","PSE","Verts-ALE"]
    plt.figure()
    ax = plt.subplot()
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='g', ax=ax, xticklabels=labels, yticklabels=labels).set_title(f"Confusion matrix for {lang}")
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    # ax.set_title(f"Confusion Matrix for {lang}")
    plt.savefig(f"conf_matrix_{lang}.png")

def random_forest(x_train, y_train, x_test):
    # returns y_pred
    rf = RandomForestClassifier(random_state=42,
                        n_estimators=20,
                        max_depth=15,
                        criterion="gini",
                        min_samples_leaf=2,
                        bootstrap=True,
                        max_samples=500)
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_test)
    return y_pred

# no need for grid search
def grid_search(x_train, y_train, x_test):
    # TODO : rerun in notebook to check which
    # model would be the most interesting
    # grid search
    n_estimators = [20, 25]
    max_depth = [15, 25, 30]
    criterion = ["gini"]
    min_samples_leaf = [2, 3]
    bootstrap = [True]
    max_samples = [500, 1000]

    param_grid = {
        "n_estimators" : n_estimators,
        "max_depth" : max_depth,
        "min_samples_leaf" : min_samples_leaf,
        "bootstrap" : bootstrap,
        "max_samples" : max_samples,
        "criterion" : criterion
    }
    # run gridsearch algorithm
    rf = RandomForestClassifier(random_state=42)
    rf_model = GridSearchCV(estimator=rf, param_grid=param_grid,
                            verbose=10, n_jobs=1)
    rf_model.fit(x_train, y_train)
    # predictions
    y_pred = rf_model.predict(x_test)
    return y_pred

def encoding_labels(encoder, labels):
    encoder.fit(labels)
    labels = encoder.transform(labels)
    return labels

@click.command()
@click.option('--lang', type=str, prompt='Language?',
              help='it, fr, en.')
def language(lang):
    # load data
    click.echo(f"language is {lang}")
    return lang

def main():
    dic_l = {"en":"english", "it":"italian", "fr":"french"}
    lang = language.main(standalone_mode=False)
    # load data
    df = pd.read_csv(f"data/extracted_data_{lang}.csv")
    data = pd.read_csv(f"data/vectorized_data_{lang}.csv")
    data["cat_id"] = df['Party']
    # train test split
    train, test = train_test_split(data, test_size=0.2,
                                   train_size=0.8, shuffle=True)
    # x, y
    x_train, y_train_label = train.loc[:, "0":"19"], train["cat_id"].values.astype(object)
    x_test, y_test_label = test.loc[:, "0":"19"], test["cat_id"].values.astype(object)
    # encoding
    encoder = preprocessing.LabelEncoder()
    y_train_label = encoding_labels(encoder, y_train_label)
    y_test_label = encoding_labels(encoder, y_test_label)

    # scaling
    # scaler = StandardScaler()
    # standardization of the data
    # x_train_scaled = scaler.fit_transform(x_train)
    # x_test_scaled = scaler.fit_transform(x_test)

    # normalize data
    x_train_norm = normalize(x_train)
    x_test_norm = normalize(x_test)

    # gridsearch
    # y_pred = grid_search(x_train_scaled, y_train_label, x_test_scaled) # categorical

    y_pred = random_forest(x_train, y_train_label, x_test)
    y_pred_label = list(encoder.inverse_transform(y_pred)) # string
    # confusion matrix
    conf_matrix(y_test_label, y_pred, dic_l[lang])
    # report
    clf_report(y_test_label, y_pred, dic_l[lang])

if __name__ == '__main__':
    main()
