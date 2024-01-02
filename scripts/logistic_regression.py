import click
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score

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

def logistic_regression(x_train, y_train, x_test):
    lr = LogisticRegression(random_state=42)
    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_test)
    return y_pred

def cross_validate_model(model, X, y):
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"Cross-validated Accuracy: {np.mean(scores)}")

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

    encoder = preprocessing.LabelEncoder()
    y_train_label = encoding_labels(encoder, y_train_label)
    y_test_label = encoding_labels(encoder, y_test_label)

    cross_validate_model(LogisticRegression(random_state=42), x_train, y_train_label)

    y_pred = logistic_regression(x_train, y_train_label, x_test)
    y_pred_label = list(encoder.inverse_transform(y_pred))

    conf_matrix(y_test_label, y_pred, dic_l[lang])
    clf_report(y_test_label, y_pred, dic_l[lang])


def encoding_labels(encoder, labels):
    encoder.fit(labels)
    labels = encoder.transform(labels)
    return labels

if __name__ == '__main__':
    main()
