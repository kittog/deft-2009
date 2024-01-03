import click
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_predict, StratifiedKFold

def clf_report(y_test, y_pred, lang):
    report = classification_report(y_test, y_pred, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(f"classification_report_{lang}.csv")

def conf_matrix(y_test, y_pred, lang):
    labels = ["ELDR", "GUE-NGL", "PPE-DE", "PSE", "Verts-ALE"]
    plt.figure()
    ax = plt.subplot()
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='g', ax=ax, xticklabels=labels, yticklabels=labels).set_title(f"Confusion matrix for {lang}")
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    plt.savefig(f"conf_matrix_{lang}.png")

def logistic_regression(x_train, y_train, x_test, C=3.0, solver='lbfgs', max_iter=200, penalty='l2', tol=1e-4):
    lr = LogisticRegression(C=C, solver=solver, max_iter=max_iter, random_state=42, penalty=penalty, tol=tol)
    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_test)
    return y_pred

def kfold_cross_validate_model(model, X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X, y, cv=skf)
    return y_pred

@click.command()
@click.option('--lang', type=str, prompt='Language?', help='it, fr, en.')
def language(lang):
    click.echo(f"language is {lang}")
    return lang

def main():
    dic_l = {"en": "english", "it": "italian", "fr": "french"}
    lang = language.main(standalone_mode=False)

    train_df = pd.read_csv(f"data/train/extracted_data_{lang}.csv")
    test_df = pd.read_csv(f"data/test/extracted_data_test_{lang}.csv")

    train_data = pd.read_csv(f"data/train/vectorized_data_{lang}.csv")
    test_data = pd.read_csv(f"data/test/vectorized_data_test_{lang}.csv")

    train_data["cat_id"] = train_df['Party']
    test_data["cat_id"] = test_df['Parties']

    x_train, y_train_label = train_data.loc[:, "0":"19"], train_data["cat_id"].values.astype(object)
    x_test, y_test_label = test_data.loc[:, "0":"19"], test_data["cat_id"].values.astype(object)
    print(set(y_test_label))
    encoder = preprocessing.LabelEncoder()
    y_train_label = encoding_labels(encoder, y_train_label)
    y_test_label = encoding_labels(encoder, y_test_label)

    y_pred = kfold_cross_validate_model(LogisticRegression(random_state=42, C=1.0, solver='liblinear', max_iter=100, penalty='l1', tol=1e-3), x_train, y_train_label, n_splits=5)

    conf_matrix(y_train_label, y_pred, dic_l[lang])
    clf_report(y_train_label, y_pred, dic_l[lang])

def encoding_labels(encoder, labels):
    encoder.fit(labels)
    labels = encoder.transform(labels)
    return labels

if __name__ == '__main__':
    main()
