import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, StratifiedKFold
import seaborn as sns
import matplotlib.pyplot as plt

# ouverture des fichiers csv
train_data = pd.read_csv('corpus/corpus_train.csv')
test_data = pd.read_csv('corpus/corpus_test.csv')

train_data = train_data[train_data['parti politique'].notna()]
test_data = test_data[test_data['parti politique'].notna()]

X_train = train_data['texte']  
y_train = train_data['parti politique'] 
X_test = test_data['texte']    
y_test = test_data['parti politique'] 
y_test = y_test.astype(str)

# vectorisation avec tfidf
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# validation croisée
model = LogisticRegression(max_iter=5000, multi_class='multinomial', solver='saga', penalty="l2",C=10,tol=1e-3)
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cross_val_scores = cross_val_score(model, X_train_tfidf, y_train, cv=kf, scoring='accuracy')

print("Scores de validation croisée :")
print(cross_val_scores)
print(f"Précision moyenne : {cross_val_scores.mean()}")

# entraînement du modèle sur données d'entraînement
model.fit(X_train_tfidf, y_train)
# prédictions
predictions = model.predict(X_test_tfidf)
print(y_test.dtype, predictions.dtype)

accuracy = accuracy_score(y_test, predictions)
print(f"Précision du modèle sur l'ensemble de test : {accuracy}")

# graphique : matrice de confusion
conf_matrix = confusion_matrix(y_test, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.title("Matrice de confusion")
plt.show()

# rapport de classification
class_report = classification_report(y_test, predictions)
print("Rapport de classification :")
print(class_report)
