import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2
import os
import pickle
from tensorflow.keras.models import load_model
from prediction import predict_from_image, load_resources
from config import MODEL_SAVE_PATH, TOKENIZER_PATH, TEST_SIZE


def visualize_training_history(history_file='logs/training_history.npz'):
    """Megjeleníti a tanítási történetet"""
    if not os.path.exists(history_file):
        print(f"A tanítási történetfájl nem található: {history_file}")
        return
    
    # Betöltjük a tanítási történetet
    history = np.load(history_file, allow_pickle=True)
    history_dict = history['history'].item()
    
    # Ábrázoljuk a veszteséget és a pontosságot
    plt.figure(figsize=(15, 5))
    
    # Veszteség
    plt.subplot(1, 2, 1)
    plt.plot(history_dict['loss'], label='Tanítási veszteség')
    plt.plot(history_dict['val_loss'], label='Validációs veszteség')
    plt.title('Modell veszteség tanítás során')
    plt.xlabel('Epoch')
    plt.ylabel('Veszteség')
    plt.legend()
    
    # Pontosság
    plt.subplot(1, 2, 2)
    plt.plot(history_dict['accuracy'], label='Tanítási pontosság')
    plt.plot(history_dict['val_accuracy'], label='Validációs pontosság')
    plt.title('Modell pontosság tanítás során')
    plt.xlabel('Epoch')
    plt.ylabel('Pontosság')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/training_history_plot.png')
    plt.show()


def visualize_confusion_matrix(cm, class_names, title='Tévesztési mátrix'):
    """
    Megjeleníti a tévesztési mátrixot
    
    Args:
        cm: A tévesztési mátrix
        class_names: Az osztálynevek listája
        title: A diagram címe
    """
    # Normalizáljuk a mátrixot
    cm = np.asarray(cm, dtype=np.float32)
    row_sums = cm.sum(axis=1)
    cm_normalized = cm / row_sums[:, np.newaxis]
    
    # Ábra készítése
    plt.figure(figsize=(10, 8))
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    
    # Tengelyek beállítása
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Értékek megjelenítése
    fmt = '.2f'
    thresh = cm_normalized.max() / 2.
    for i, j in np.ndindex(cm_normalized.shape):
        plt.text(j, i, format(cm_normalized[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm_normalized[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('Valós címke')
    plt.xlabel('Jósolt címke')
    plt.savefig('results/confusion_matrix.png')
    plt.show()


def visualize_attention(image_path, attention_weights, predicted_latex):
    """
    Figyelem súlyok megjelenítése
    
    Args:
        image_path: A kép elérési útja
        attention_weights: Figyelem súlyok (batch_size, output_length, input_length)
        predicted_latex: Jósolt LaTeX kifejezés
    """
    # Kép betöltése
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Figyelem súlyok megjelenítése
    plt.figure(figsize=(16, 12))
    
    # Eredeti kép
    plt.subplot(2, 1, 1)
    plt.imshow(img_rgb)
    plt.title("Eredeti kép")
    plt.axis('off')
    
    # Figyelem hőtérkép
    plt.subplot(2, 1, 2)
    attention = np.mean(attention_weights[0], axis=0)  # Átlagoljuk a figyelem súlyokat
    attention = cv2.resize(attention, (img.shape[1], img.shape[0]))  # Átméretezés a kép méretére
    
    plt.imshow(img_rgb)
    plt.imshow(attention, cmap='jet', alpha=0.5)
    plt.title(f"Figyelem hőtérkép\nJósolt LaTeX: {predicted_latex}")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/attention_visualization.png')
    plt.show()


def visualize_test_samples(num_samples=5):
    """
    Megjeleníti a tesztmintákat és a modell jóslatát
    
    Args:
        num_samples: A megjelenítendő tesztminták száma
    """
    from data_processing import load_or_process_data
    
    # Adatok betöltése
    print("Adatok betöltése...")
    _, _, X_test, _, _, y_test, tokenizer = load_or_process_data()
    
    # Ellenőrizzük, hogy van-e elég tesztminta
    if len(X_test) < num_samples:
        num_samples = len(X_test)
        print(f"Csak {num_samples} tesztminta érhető el.")
    
    # Modell betöltése
    print("Modell betöltése...")
    model, tokenizer = load_resources()
    
    # Minták véletlenszerű kiválasztása
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    
    plt.figure(figsize=(15, 5 * num_samples))
    
    for i, idx in enumerate(indices):
        # Kép konvertálása
        img = X_test[idx]
        img_rgb = np.squeeze(img)  # Eltávolítjuk a csatorna dimenziót
        
        # LaTeX jóslat készítése
        img_batch = np.expand_dims(img, axis=0)
        
        # Szókincs méret
        vocab_size = min(len(tokenizer.word_index) + 1, tokenizer.num_words)
        
        # Kezdeti input a decoder számára
        batch_size = 1
        decoder_input = np.zeros((batch_size, y_test.shape[1]))
        decoder_input[:, 0] = vocab_size - 1  # <START> token
        
        # Jóslat
        predictions = model.predict([img_batch, decoder_input], verbose=0)
        
        # Jósolt indexek
        predicted_indices = np.argmax(predictions[0], axis=1)
        
        # Konvertálás szövegre
        index_to_word = {v: k for k, v in tokenizer.word_index.items()}
        
        predicted_tokens = [index_to_word.get(idx, "") for idx in predicted_indices if idx > 0]
        predicted_latex = " ".join(predicted_tokens)
        
        # Valós LaTeX
        true_indices = y_test[idx]
        true_tokens = [index_to_word.get(idx, "") for idx in true_indices if idx > 0]
        true_latex = " ".join(true_tokens)
        
        # Megjelenítés
        plt.subplot(num_samples, 1, i+1)
        plt.imshow(img_rgb, cmap='gray')
        plt.title(f"Példa {i+1}")
        plt.xlabel(f"Valós: {true_latex}\nJósolt: {predicted_latex}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/test_samples.png')
    plt.show()


def visualize_predictions(image_paths):
    """
    Megjeleníti a modell jóslatát a megadott képekre
    
    Args:
        image_paths: A képek elérési útjai
    """
    # Modell betöltése
    model, tokenizer = load_resources()
    
    # Képek feldolgozása
    num_images = len(image_paths)
    plt.figure(figsize=(15, 5 * num_images))
    
    for i, image_path in enumerate(image_paths):
        # LaTeX jóslat készítése
        predicted_latex = predict_from_image(image_path, model, tokenizer)
        
        # Kép betöltése
        img = cv2.imread(image_path)
        if img is None:
            print(f"Nem sikerült betölteni a képet: {image_path}")
            continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Megjelenítés
        plt.subplot(num_images, 1, i+1)
        plt.imshow(img_rgb)
        plt.title(f"Egyenlet {i+1}")
        plt.xlabel(f"Jósolt LaTeX: {predicted_latex}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/custom_predictions.png')
    plt.show()


if __name__ == "__main__":
    # Példa használat
    import argparse
    
    parser = argparse.ArgumentParser(description="Matematikai egyenletfelismerő eredmények vizualizációja")
    parser.add_argument('--history', action='store_true', help="Tanítási történet megjelenítése")
    parser.add_argument('--test', action='store_true', help="Tesztminták megjelenítése")
    parser.add_argument('--images', nargs='+', help="Saját képek megadása jósláshoz")
    
    args = parser.parse_args()
    
    if args.history:
        visualize_training_history()
    
    if args.test:
        visualize_test_samples(5)
    
    if args.images:
        visualize_predictions(args.images)
        
    # Ha nincs megadva paraméter, megjelenítjük az összes vizualizációt
    if not (args.history or args.test or args.images):
        print("Használat: python visualize_results.py --history --test --images kep1.png kep2.png")

        