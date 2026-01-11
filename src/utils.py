import pandas as pd
import numpy as np
import re
import pickle
import os
from nltk.corpus import stopwords

def load_data():
    train_data = pd.read_csv('data/train.csv')
    test_data = pd.read_csv('data/test.csv')
    return train_data, test_data

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # Nettoyage de base
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Enlever stopwords anglais
    try:
        words = text.split()
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if w not in stop_words]
        text = ' '.join(words)
    except:
        pass
    
    return text

def save_model(model, filename):
    with open(f'models/{filename}', 'wb') as f:
        pickle.dump(model, f)

def load_model(filename):
    with open(f'models/{filename}', 'rb') as f:
        return pickle.load(f)
