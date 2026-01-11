import pandas as pd
import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np

print("=" * 60)
print("ENTRAÎNEMENT SVM")
print("=" * 60)

# 1. Charger données
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

# 2. Charger vectorizer
with open('models/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

# 3. Préparer données
X_train = tfidf_vectorizer.transform(train_data['text'])
X_test = tfidf_vectorizer.transform(test_data['text'])
y_train = train_data['label'].values.astype(int)
y_test = test_data['label'].values.astype(int)

print(f"📊 Classes uniques: {np.unique(y_train)}")

# 4. Entraîner
print("\n🎯 Entraînement du modèle...")
model = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
model.fit(X_train, y_train)

# 5. Prédictions
y_pred = model.predict(X_test)

# 6. CALCULER TOUTES LES MÉTRIQUES
print("\n📈 ÉVALUATION DU MODÈLE:")

# Métriques globales
accuracy = accuracy_score(y_test, y_pred)
precision_macro = precision_score(y_test, y_pred, average='macro')
recall_macro = recall_score(y_test, y_pred, average='macro')
f1_macro = f1_score(y_test, y_pred, average='macro')

print(f"✅ Précision (Accuracy): {accuracy:.2%}")
print(f"✅ Précision (macro): {precision_macro:.2%}")
print(f"✅ Rappel (macro): {recall_macro:.2%}")
print(f"✅ F1-Score (macro): {f1_macro:.2%}")

# Métriques par classe
precision_per_class = precision_score(y_test, y_pred, average=None)
recall_per_class = recall_score(y_test, y_pred, average=None)
f1_per_class = f1_score(y_test, y_pred, average=None)

print("\n📊 Métriques par classe (0-4):")
for i in range(5):
    print(f"  Classe {i}:")
    print(f"    Précision: {precision_per_class[i]:.2%}")
    print(f"    Rappel: {recall_per_class[i]:.2%}")
    print(f"    F1-Score: {f1_per_class[i]:.2%}")

# Rapport détaillé
print("\n📋 Rapport de classification détaillé:")
print(classification_report(y_test, y_pred, digits=3))

# 7. Sauvegarder
with open('models/svm_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# 8. Sauvegarder métriques
metrics_data = {
    'Model': ['SVM'],
    'Accuracy': [accuracy],
    'Precision_macro': [precision_macro],
    'Recall_macro': [recall_macro],
    'F1_macro': [f1_macro]
}

for i in range(5):
    metrics_data[f'Precision_class_{i}'] = [precision_per_class[i]]
    metrics_data[f'Recall_class_{i}'] = [recall_per_class[i]]
    metrics_data[f'F1_class_{i}'] = [f1_per_class[i]]

metrics_df = pd.DataFrame(metrics_data)
metrics_df.to_csv('metrics/svm_metrics.csv', index=False)

print("\n" + "=" * 60)
print("✅ SVM ENTRÂÎNÉ ET ÉVALUÉ!")
print(f"📁 Modèle: models/svm_model.pkl")
print(f"📊 Métriques: metrics/svm_metrics.csv")
print("=" * 60)