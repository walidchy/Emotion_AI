import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Dropout, Embedding
from keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

print("=" * 60)
print("ENTRAÎNEMENT RNN (Deep Learning)")
print("=" * 60)

# 1. Charger données
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

# 2. Charger Word2Vec
word2vec_model = Word2Vec.load('models/word2vec_model.bin')
embedding_dim = 100
max_length = 30

print(f"📊 Word2Vec vocab size: {len(word2vec_model.wv)}")

# 3. Préparer séquences
def text_to_sequence(text):
    words = str(text).lower().split()
    sequence = []
    for word in words:
        if word in word2vec_model.wv:
            sequence.append(word2vec_model.wv.key_to_index[word])
    return sequence

X_train_seq = [text_to_sequence(text) for text in train_data['text']]
X_test_seq = [text_to_sequence(text) for text in test_data['text']]

X_train_padded = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
X_test_padded = pad_sequences(X_test_seq, maxlen=max_length, padding='post')

y_train = train_data['label'].values.astype(int)
y_test = test_data['label'].values.astype(int)

print(f"📊 Classes uniques: {np.unique(y_train)}")

# 4. Créer matrice d'embedding
vocab_size = len(word2vec_model.wv)
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, idx in word2vec_model.wv.key_to_index.items():
    embedding_matrix[idx] = word2vec_model.wv[word]

# 5. Construire modèle RNN
model = Sequential([
    Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        input_length=max_length,
        trainable=False
    ),
    SimpleRNN(128, return_sequences=True),
    Dropout(0.3),
    SimpleRNN(64),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(6, activation='softmax')  # 6 classes
])

# 6. Compiler
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 7. Entraîner
print("\n🎯 Début entraînement...")
history = model.fit(
    X_train_padded, y_train,
    epochs=15,
    batch_size=32,
    validation_data=(X_test_padded, y_test),
    verbose=1
)

# 8. Prédictions
y_pred_proba = model.predict(X_test_padded, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)

# 9. CALCULER TOUTES LES MÉTRIQUES
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

# 10. Sauvegarder
model.save('models/rnn_model.h5')

# 11. Sauvegarder métriques
metrics_data = {
    'Model': ['RNN'],
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
metrics_df.to_csv('metrics/rnn_metrics.csv', index=False)



print("\n" + "=" * 60)
print("✅ RNN ENTRÂÎNÉ ET ÉVALUÉ!")
print(f"📁 Modèle: models/rnn_model.h5")
print(f"📊 Métriques: metrics/rnn_metrics.csv")
print("=" * 60)