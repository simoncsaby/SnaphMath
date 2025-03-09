import os
import numpy as np
import pickle
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from data_processing import preprocess_image
from config import MODEL_SAVE_PATH, TOKENIZER_PATH, MAX_SEQUENCE_LENGTH


def load_resources():
    """Betölti a modellt és a tokenizert"""
    if not os.path.exists(MODEL_SAVE_PATH):
        raise FileNotFoundError(f"A modell nem található: {MODEL_SAVE_PATH}")
    
    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(f"A tokenizer nem található: {TOKENIZER_PATH}")
    
    # Modell betöltése
    model = load_model(MODEL_SAVE_PATH)
    
    # Tokenizer betöltése
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
    
    return model, tokenizer


def predict_from_image(image_path, model=None, tokenizer=None):
    """
    Kép alapján LaTeX egyenletet jósol
    """
    if model is None or tokenizer is None:
        model, tokenizer = load_resources()
    
    # Kép előfeldolgozása
    img = preprocess_image(image_path)
    if img is None:
        return "Nem sikerült a kép beolvasása vagy feldolgozása."
    
    # Kép kiterjesztése batch dimenzióval
    img = np.expand_dims(img, axis=0)
    
    # Szókincs méret
    vocab_size = min(len(tokenizer.word_index) + 1, tokenizer.num_words)
    
    # Kezdeti input a decoder számára
    decoder_input = np.zeros((1, MAX_SEQUENCE_LENGTH))
    decoder_input[0, 0] = vocab_size - 1  # <START> token
    
    # Jóslat generálása
    predicted_sequence = []
    for i in range(1, MAX_SEQUENCE_LENGTH):
        predictions = model.predict([img, decoder_input], verbose=0)
        sampled_token_index = np.argmax(predictions[0, i-1, :])
        
        predicted_sequence.append(sampled_token_index)
        
        if sampled_token_index == 0:  # <PAD> token, a szekvencia vége
            break
        
        if i < MAX_SEQUENCE_LENGTH - 1:
            decoder_input[0, i] = sampled_token_index
    
    # Konvertálás szövegre
    predicted_indices = [idx for idx in predicted_sequence if idx > 0]  # Eltávolítjuk a PAD tokent
    predicted_tokens = []
    
    # Fordított szótár létrehozása: index -> token
    index_to_word = {v: k for k, v in tokenizer.word_index.items()}
    
    for idx in predicted_indices:
        if idx in index_to_word:
            predicted_tokens.append(index_to_word[idx])
    
    # LaTeX egyenlet összefűzése
    latex_equation = " ".join(predicted_tokens)
    
    return latex_equation


def evaluate_on_test_image(test_image_path):
    """
    Teszteli a modellt egy kiválasztott képen és megjeleníti az eredményt
    """
    model, tokenizer = load_resources()
    
    # Kép beolvasása és előfeldolgozása vizualizációhoz
    img = cv2.imread(test_image_path)
    if img is None:
        print(f"Nem sikerült a kép beolvasása: {test_image_path}")
        return
    
    # RGB-re konvertálás a megjelenítéshez
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Jóslat
    latex_prediction = predict_from_image(test_image_path, model, tokenizer)
    
    # Megjelenítés
    plt.figure(figsize=(10, 6))
    plt.imshow(img_rgb)
    plt.title("Egyenlet")
    plt.axis('off')
    
    plt.figtext(0.5, 0.01, f"Felismert LaTeX: {latex_prediction}", 
                ha="center", fontsize=12, bbox={"facecolor":"white", "alpha":0.8, "pad":5})
    
    plt.tight_layout()
    plt.savefig('prediction_result.png')
    plt.show()
    
    print(f"Felismert LaTeX egyenlet: {latex_prediction}")
    
    return latex_prediction
