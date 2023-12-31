import click
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
import xgboost as xgb

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

def xgboost_classification(x_train, y_train, x_test):
    xgb_model = xgb.XGBClassifier(random_state=42)
    xgb_model.fit(x_train, y_train)
    y_pred = xgb_model.predict(x_test)
    return y_pred

def cross_validate_model(model, X, y):
    encoder = preprocessing.LabelEncoder()
    y_encoded = encoding_labels(encoder, y)
    
    scores = cross_val_score(model, X, y_encoded, cv=5, scoring='accuracy')
    print(f"Cross-validated Accuracy: {np.mean(scores)}")

@click.command()
@click.option('--lang', type=str, prompt='Language?', help='it, fr, en.')
def language(lang):
    click.echo(f"language is {lang}")
    return lang

def main():
    dic_l = {"en": "english", "it": "italian", "fr": "french"}
    lang = language.main(standalone_mode=False)
    df = pd.read_csv(f"data/extracted_data_{lang}.csv")
    data = pd.read_csv(f"data/vectorized_data_{lang}.csv")
    data["cat_id"] = df['Party']
    
    train, test = train_test_split(data, test_size=0.2, train_size=0.8, shuffle=True)
    x_train, y_train_label = train.loc[:, "0":"19"], train["cat_id"].values.astype(object)
    x_test, y_test_label = test.loc[:, "0":"19"], test["cat_id"].values.astype(object)
    
    encoder = preprocessing.LabelEncoder()
    y_train_label = encoding_labels(encoder, y_train_label)
    y_test_label = encoding_labels(encoder, y_test_label)

    cross_validate_model(xgb.XGBClassifier(random_state=42), data.loc[:, "0":"19"], data["cat_id"].values.astype(object))
    
    y_pred = xgboost_classification(x_train, y_train_label, x_test)
    y_pred_label = list(encoder.inverse_transform(y_pred))
    
    conf_matrix(y_test_label, y_pred, dic_l[lang])
    clf_report(y_test_label, y_pred, dic_l[lang])

def encoding_labels(encoder, labels):
    encoder.fit(labels)
    labels = encoder.transform(labels)
    return labels

if __name__ == '__main__':
    main()
