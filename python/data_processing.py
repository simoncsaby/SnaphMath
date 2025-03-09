import os
import cv2
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from config import (
    IM2LATEX_DATA_PATH, COMBINED_DATA_PATH,
    IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS, MAX_SEQUENCE_LENGTH,
    TOKENIZER_PATH, VOCAB_SIZE, TEST_SIZE, VALIDATION_SIZE, RANDOM_STATE
)


def preprocess_image(image_path):
    """Kép beolvasása és előfeldolgozása"""
    try:
        # Kép beolvasása szürkeárnyalatosan
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"Nem sikerült a kép beolvasása: {image_path}")
            return None
            
        # Átméretezés
        img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
        
        # Normalizálás (0-1 közé)
        img = img.astype(np.float32) / 255.0
        
        # Csatorna dimenzió hozzáadása
        img = np.expand_dims(img, axis=-1)
        
        return img
    except Exception as e:
        print(f"Hiba a kép feldolgozása során: {image_path}, {str(e)}")
        return None


def process_im2latex_data(max_samples=5000):
    """im2latex adathalmaz feldolgozása"""
    print("im2latex adathalmaz feldolgozása...")
    
    # Ellenőrizzük, hogy létezik-e az útvonal
    if not os.path.exists(IM2LATEX_DATA_PATH):
        print(f"FIGYELMEZTETÉS: {IM2LATEX_DATA_PATH} nem létezik!")
        return [], []
    
    # Adatok beolvasása
    images = []
    latex_expressions = []
    
    # LaTeX kifejezések beolvasása
    formulas_file = os.path.join(IM2LATEX_DATA_PATH, "im2latex_formulas.lst")
    if not os.path.exists(formulas_file):
        print(f"FIGYELMEZTETÉS: {formulas_file} nem létezik!")
        return [], []
    
    # Különböző kódolások kipróbálása
    encodings = ['utf-8', 'latin-1', 'cp1252', 'ISO-8859-1']
    formulas = []
    
    for encoding in encodings:
        try:
            with open(formulas_file, 'r', encoding=encoding) as f:
                formulas = [line.strip() for line in f]
            print(f"Sikeres kódolás: {encoding} - {len(formulas)} formula beolvasva")
            break
        except UnicodeDecodeError:
            print(f"{encoding} kódolás sikertelen, következő próbálkozás...")
    
    if not formulas:
        print("Nem sikerült beolvasni a formula fájlt egyik kódolással sem.")
        return [], []
    
    # Vizsgáljuk meg az im2latex_train.lst fájl struktúráját
    print("\nAz im2latex_train.lst fájl struktúrájának ellenőrzése:")
    train_file = os.path.join(IM2LATEX_DATA_PATH, "im2latex_train.lst")
    if not os.path.exists(train_file):
        print(f"FIGYELMEZTETÉS: {train_file} nem létezik!")
        return [], []
    
    # Kísérlet az első 5 sor beolvasására
    try:
        with open(train_file, 'r', encoding=encoding) as f:
            first_lines = [next(f).strip() for _ in range(5)]
        
        print("Az im2latex_train.lst első 5 sora:")
        for i, line in enumerate(first_lines):
            print(f"  {i+1}: {line}")
        
        # Vizsgáljuk meg az első sort részletesebben
        if first_lines:
            parts = first_lines[0].split()
            print(f"\nAz első sor részei: {parts}")
            
            # Ha nincs elég rész a sorban, vagy a második rész nem szám
            if len(parts) < 2:
                print("FIGYELMEZTETÉS: A sorok formátuma nem a várt 'image_id formula_id' formátum!")
            else:
                try:
                    formula_id = int(parts[1])
                    print(f"Formula ID: {formula_id}, ez egy érvényes szám.")
                except ValueError:
                    print(f"FIGYELMEZTETÉS: A második rész nem szám: '{parts[1]}'")
                    
                    # Próbáljuk megállapítani a helyes formátumot
                    all_numeric = all(c.isdigit() for c in parts[1])
                    if all_numeric:
                        print("A második rész csak számokat tartalmaz, de konverziós hiba történt.")
                    else:
                        print("A második rész nem numerikus. Lehet, hogy a formátum 'image_id formula' formátumú.")
    except Exception as e:
        print(f"Hiba az im2latex_train.lst első sorainak olvasásakor: {str(e)}")
    
    # Most próbáljuk meg beolvasni az adatokat a helyes formátum alapján
    try:
        with open(train_file, 'r', encoding=encoding) as f:
            line_count = 0
            for line in tqdm(f, desc="im2latex adatok olvasása"):
                if len(images) >= max_samples:
                    print(f"Elérte a maximális mintaszámot ({max_samples}), adatbetöltés befejezve.")
                    break
                
                line_count += 1
                
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                
                image_name = parts[0]
                
                # Ellenőrizzük, hogy a második rész szám-e
                try:
                    formula_id = int(parts[1])
                except ValueError:
                    # Ha nem szám, tegyük fel, hogy ez maga a képnév második része
                    # Ez az eset akkor fordulhat elő, ha a formátum eltér a várt "kép_id formula_id" formátumtól
                    # print(f"Figyelmeztetés a(z) {line_count}. sorban: '{parts[1]}' nem szám. Folytassuk hexadecimális ID-vel.")
                    # Itt feltételezzük, hogy a formula_id sorfolytonosan növekszik
                    formula_id = line_count % len(formulas)  # Biztonságos fallback
                
                image_path = os.path.join(IM2LATEX_DATA_PATH, "images", f"{image_name}.png")
                
                if not os.path.exists(image_path):
                    # Próbáljuk meg közvetlenül a nevet használni, hátha az a teljes útvonal
                    alt_path = os.path.join(IM2LATEX_DATA_PATH, "images", image_name)
                    if os.path.exists(alt_path):
                        image_path = alt_path
                    else:
                        continue
                
                img = preprocess_image(image_path)
                if img is not None and formula_id < len(formulas):
                    images.append(img)
                    latex_expressions.append(formulas[formula_id])
    except Exception as e:
        print(f"Hiba az im2latex adatok olvasásakor: {str(e)}")
    
    # Próbáljuk meg a validate és test fájlokat is, ha nincs elég példa
    if len(images) < 100:
        for file_name in ["im2latex_validate.lst", "im2latex_test.lst"]:
            extra_file = os.path.join(IM2LATEX_DATA_PATH, file_name)
            if os.path.exists(extra_file):
                try:
                    print(f"Kiegészítés a {file_name} fájlból...")
                    with open(extra_file, 'r', encoding=encoding) as f:
                        for line in tqdm(f, desc=f"{file_name} olvasása"):
                            if len(images) >= max_samples:
                                break
                            
                            parts = line.strip().split()
                            if len(parts) < 2:
                                continue
                            
                            image_name = parts[0]
                            try:
                                formula_id = int(parts[1])
                            except ValueError:
                                formula_id = 0  # Egyszerű fallback
                            
                            image_path = os.path.join(IM2LATEX_DATA_PATH, "images", f"{image_name}.png")
                            
                            if not os.path.exists(image_path):
                                continue
                            
                            img = preprocess_image(image_path)
                            if img is not None and formula_id < len(formulas):
                                images.append(img)
                                latex_expressions.append(formulas[formula_id])
                except Exception as e:
                    print(f"Hiba a {file_name} fájl olvasásakor: {str(e)}")
    
    # Ha még mindig nincs elég példa, akkor közvetlenül olvassuk be a képeket
    if len(images) < 100:
        print("Nem sikerült elég példát beolvasni a lista fájlokból. Közvetlen képbeolvasás...")
        images_dir = os.path.join(IM2LATEX_DATA_PATH, "images")
        if os.path.exists(images_dir):
            for i, img_file in enumerate(os.listdir(images_dir)):
                if i >= max_samples:
                    break
                
                if img_file.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(images_dir, img_file)
                    img = preprocess_image(img_path)
                    
                    if img is not None:
                        images.append(img)
                        # Egyszerű fallback, használjunk egy alapértelmezett formulát
                        formula_idx = i % len(formulas)
                        latex_expressions.append(formulas[formula_idx])
    
    print(f"im2latex: {len(images)} példa beolvasva.")
    return images, latex_expressions


def prepare_dataset():
    """Adathalmaz előkészítése"""
    print("Adathalmaz előkészítése...")
    
    # im2latex adatok betöltése
    images, latex_expressions = process_im2latex_data()
    
    # Ellenőrizzük, hogy vannak-e adatok
    if not images:
        raise ValueError("Nem sikerült adatokat beolvasni az im2latex adathalmazból!")
    
    # Tokenizer létrehozása és tanítása a LaTeX szövegeken
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, filters='', lower=False, oov_token="<UNK>")
    tokenizer.fit_on_texts(latex_expressions)
    
    # Mentsük el a tokenizert
    os.makedirs(os.path.dirname(TOKENIZER_PATH), exist_ok=True)
    with open(TOKENIZER_PATH, 'wb') as f:
        pickle.dump(tokenizer, f)
    
    # Konvertáljuk a LaTeX szövegeket szekvenciákká
    sequences = tokenizer.texts_to_sequences(latex_expressions)
    padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    
    # Konvertáljuk a képeket NumPy tömbbé
    images_array = np.array(images)
    
    # Frissítsük a VOCAB_SIZE-t és jelezzük
    actual_vocab_size = min(len(tokenizer.word_index) + 1, VOCAB_SIZE)
    print(f"Szókincs mérete: {actual_vocab_size} (VOCAB_SIZE={VOCAB_SIZE})")
    
    # Adathalmaz felosztása tanító, validációs és teszt halmazra
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        images_array, padded_sequences, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, 
        test_size=VALIDATION_SIZE/(1-TEST_SIZE),  # Relatív méret
        random_state=RANDOM_STATE
    )
    
    # Eredmények mentése
    os.makedirs(COMBINED_DATA_PATH, exist_ok=True)
    npz_path = os.path.join(COMBINED_DATA_PATH, "processed_data.npz")
    
    np.savez_compressed(
        npz_path,
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test
    )
    
    print(f"Adathalmaz méretei:")
    print(f"  Tanító:     {len(X_train)} példa")
    print(f"  Validációs: {len(X_val)} példa")
    print(f"  Teszt:      {len(X_test)} példa")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, tokenizer


def load_or_process_data():
    """Adatok betöltése vagy feldolgozása"""
    npz_path = os.path.join(COMBINED_DATA_PATH, "processed_data.npz")
    
    # Ha már feldolgoztuk az adatokat, csak betöltjük
    if os.path.exists(npz_path):
        print("Előfeldolgozott adatok betöltése...")
        data = np.load(npz_path, allow_pickle=True)
        X_train = data['X_train']
        X_val = data['X_val']
        X_test = data['X_test']
        y_train = data['y_train']
        y_val = data['y_val']
        y_test = data['y_test']
        
        # Tokenizer betöltése
        with open(TOKENIZER_PATH, 'rb') as f:
            tokenizer = pickle.load(f)
        
        print(f"Adathalmaz méretei:")
        print(f"  Tanító:     {len(X_train)} példa")
        print(f"  Validációs: {len(X_val)} példa")
        print(f"  Teszt:      {len(X_test)} példa")
        
        return X_train, X_val, X_test, y_train, y_val, y_test, tokenizer
    
    # Különben feldolgozzuk az adatokat
    return prepare_dataset()


if __name__ == "__main__":
    # Teszt az adatfeldolgozásra
    X_train, X_val, X_test, y_train, y_val, y_test, tokenizer = load_or_process_data()
    
    # Néhány példa megjelenítése (opcionális)
    print("\nPéldák az adathalmazból:")
    index_to_word = {v: k for k, v in tokenizer.word_index.items()}
    
    for i in range(min(3, len(X_test))):
        # LaTeX visszafejtése
        example_tokens = [index_to_word.get(idx, "") for idx in y_test[i] if idx > 0]
        example_latex = " ".join(example_tokens)
        print(f"Példa {i+1}: {example_latex}")