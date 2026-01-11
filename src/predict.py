import pickle
import numpy as np
import re
from tensorflow import keras
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences

class EmotionPredictor:
    def __init__(self, model_name='lstm'):
        self.model_name = model_name
        self.load_models()
        print(f"Model '{self.model_name.upper()}' loaded successfully!")
    
    def load_models(self):
        """Charge tous les modèles nécessaires"""
        try:
            # Charger vectorizer
            with open('models/tfidf_vectorizer.pkl', 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)
            
            # Charger modèle
            if self.model_name in ['naive_bayes', 'svm', 'knn', 'random_forest']:
                with open(f'models/{self.model_name}_model.pkl', 'rb') as f:
                    self.model = pickle.load(f)
            elif self.model_name in ['rnn', 'lstm']:
                self.word2vec_model = Word2Vec.load('models/word2vec_model.bin')
                self.model = keras.models.load_model(f'models/{self.model_name}_model.h5')
            
            print(f"Successfully loaded {self.model_name}")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            raise
    
    def clean_text(self, text):
        """Nettoie le texte"""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def preprocess_for_dl(self, text, max_length=30):
        """Prépare le texte pour les modèles Deep Learning"""
        words = self.clean_text(text).split()
        sequence = []
        for word in words:
            if word in self.word2vec_model.wv:
                sequence.append(self.word2vec_model.wv.key_to_index[word])
        
        if sequence:
            return pad_sequences([sequence], maxlen=max_length, padding='post')
        return np.zeros((1, max_length))
    
    def predict(self, text):
        """Prédit l'émotion (0-5)"""
        clean_text = self.clean_text(text)
        
        try:
            if self.model_name in ['naive_bayes', 'svm', 'knn', 'random_forest']:
                # Modèles classiques
                vector = self.tfidf_vectorizer.transform([clean_text])
                
                if hasattr(self.model, 'predict_proba'):
                    probas = self.model.predict_proba(vector)[0]
                    prediction_idx = np.argmax(probas)
                    confidence = probas[prediction_idx]
                else:
                    prediction = self.model.predict(vector)[0]
                    prediction_idx = int(prediction)  # Directement 0-5
                    confidence = 1.0
            
            else:  # RNN/LSTM
                # Modèles Deep Learning
                sequence = self.preprocess_for_dl(text)
                probas = self.model.predict(sequence, verbose=0)[0]
                prediction_idx = np.argmax(probas)
                confidence = probas[prediction_idx]
            
            # S'assurer que c'est entre 0 et 5
            final_emotion = int(prediction_idx)
            if final_emotion < 0 or final_emotion > 5:
                final_emotion = final_emotion % 6
            
            # Noms des émotions (6 classes)
            emotion_names = {
                0: "TRISTESSE",
                1: "JOIE", 
                2: "AMOUR",
                3: "COLÈRE",
                4: "PEUR",
                5: "SURPRISE"
            }
            
            emojis = {
                0: "😢",
                1: "😊",
                2: "❤️",
                3: "😠",
                4: "😨",
                5: "😲"
            }
            
            return {
                'emotion': final_emotion,  # 0-5
                'emotion_name': emotion_names.get(final_emotion, "INCONNU"),
                'emoji': emojis.get(final_emotion, '🎭'),
                'confidence': float(confidence),
                'text': text
            }
            
        except Exception as e:
            print(f"❌ Error in prediction: {e}")
            return {
                'error': str(e),
                'emotion': 0,
                'confidence': 0.0,
                'text': text
            }

if __name__ == "__main__":
    # Test
    predictor = EmotionPredictor('lstm')
    test_texts = [
        "I feel sad",
        "I am very happy",
        "I love you",
        "I am angry",
        "I am afraid",
        "What a surprise!"
    ]
    
    print("Test des prédictions (6 classes):")
    for text in test_texts:
        result = predictor.predict(text)
        print(f"'{text}' → {result['emoji']} {result['emotion_name']} (Classe: {result['emotion']})")