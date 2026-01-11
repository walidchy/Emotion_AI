import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import numpy as np
from utils import load_data, clean_text

def create_tfidf_vectorizer():
    print("🔄 Création du vectorizer TF-IDF...")
    
    train_data, test_data = load_data()
    
    # Nettoyer texte
    train_data['clean_text'] = train_data['text'].apply(clean_text)
    test_data['clean_text'] = test_data['text'].apply(clean_text)
    
    # TF-IDF
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = tfidf_vectorizer.fit_transform(train_data['clean_text'])
    X_test_tfidf = tfidf_vectorizer.transform(test_data['clean_text'])
    
    # Sauvegarder
    with open('models/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    
    print(f"✅ TF-IDF créé avec {len(tfidf_vectorizer.get_feature_names_out())} features")
    return X_train_tfidf, train_data['label'], X_test_tfidf, test_data['label']

def create_word2vec_model():
    print("🔄 Création du modèle Word2Vec...")
    train_data, _ = load_data()
    train_data['clean_text'] = train_data['text'].apply(clean_text)
    
    sentences = [text.split() for text in train_data['clean_text']]
    word2vec_model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, workers=4)
    word2vec_model.save("models/word2vec_model.bin")
    
    print(f"✅ Word2Vec créé avec {len(word2vec_model.wv)} mots")
    return word2vec_model


if __name__ == "__main__":
    print("=" * 50)
    print("PRÉTRAITEMENT DES DONNÉES")
    print("=" * 50)
    
    X_train_tfidf, y_train_tfidf, X_test_tfidf, y_test_tfidf = create_tfidf_vectorizer()
    word2vec_model = create_word2vec_model()
    
    print("\n" + "=" * 50)
    print("✅ PRÉTRAITEMENT TERMINÉ!")
    print("=" * 50)