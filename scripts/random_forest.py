# importing modules
import click
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV

def conf_matrix(y_test, y_pred, lang):
    labels = ["ELDR", "GUE-NGL", "PPE-DE","PSE","Verts-ALE"]
    ax = plt.subplot()
    cm = confusion_matrix(y_test_label, y_pred_label)
    sns.heatmap(cm, annot=True, fmt='g', ax=ax, xticklabels=labels, yticklabels=labels)
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    ax.set_title(f"Confusion Matrix for {lang}")
    plt.savefig(f"conf_matrix_{lang}.png")


def grid_search(x_train, y_train, x_test):
    # grid search
    n_estimators = [20, 25]
    max_depth = [15, 25, 30]
    criterion = ["gini"]
    min_samples_leaf = [2, 3]
    bootstrap = [True]
    max_samples = [500, 1000, 5000, 7000]

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

def encoding_labels(labels):
    encoder = preprocessing.LabelEncoder()
    encoder.fit(labels)
    labels = encoder.transform(labels)
    return labels

@click.command()
@click.option('--lang', prompt='Language?',
              help='it, fr, en.')
def main():
    # load data
    df = pd.read_csv(f"../data/extracted_data_{lang}.csv")
    data = pd.read_csv(f"../data/vectorized_data_{lang}.csv")
    # adding labels to vectorized texts
    data['cat_id'] = df['Party']
    # train test split
    train, test = train_test_split(data, test_size=0.2,
                                   train_size=0.8, shuffle=True)
    # x, y
    x_train, y_train_label = train.loc[:, "0":"19"], train["cat_id"].values.astype(object)
    x_test, y_test_label = test.loc[:, "0":"19"], test["cat_id"].values.astype(object)
    # encoding
    y_train_label = encoding_labels(y_train_label)
    y_test_label = encoding_labels(y_test_label)
    # scaling
    scaler = StandardScaler()
    # standardization of the data
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.fit_transform(x_test)
    # gridsearch
    y_pred = grid_search(x_train_scaled, y_train_label, x_test_scaled)
    y_pred_label = list(encoder.inverse_transform(y_pred))
    # confusion matrix
    conf_matrix(y_test_label, y_pred_label, lang)

main()
