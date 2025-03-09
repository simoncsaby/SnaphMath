import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from config import CROHME_DATA_PATH, IM2LATEX_DATA_PATH


def check_dataset_availability():
    """
    Ellenőrzi, hogy az adathalmazok elérhetőek-e, és segítséget ad a letöltésükhöz
    """
    crohme_available = os.path.exists(CROHME_DATA_PATH) and len(os.listdir(CROHME_DATA_PATH)) > 0
    im2latex_available = os.path.exists(IM2LATEX_DATA_PATH) and len(os.listdir(IM2LATEX_DATA_PATH)) > 0
    
    if not crohme_available:
        print("CROHME adathalmaz nem található vagy üres.")
        print("Letöltési útmutató:")
        print("1. Látogass el a https://www.isical.ac.in/~crohme/ oldalra")
        print("2. Töltsd le a CROHME adathalmazt")
        print(f"3. Csomagold ki a letöltött fájlokat a '{CROHME_DATA_PATH}' mappába")
        print("4. Ellenőrizd, hogy az adatok megfelelő struktúrában vannak-e")
    
    if not im2latex_available:
        print("im2latex-100k adathalmaz nem található vagy üres.")
        print("Letöltési útmutató:")
        print("1. Látogass el a https://github.com/harvardnlp/im2markup oldalra")
        print("2. Kövesd a README.md útmutatását az adathalmaz letöltéséhez")
        print(f"3. Csomagold ki a letöltött fájlokat a '{IM2LATEX_DATA_PATH}' mappába")
        print("4. Ellenőrizd, hogy az adatok megfelelő struktúrában vannak-e")
    
    return crohme_available or im2latex_available


def visualize_batch(images, latex_equations, tokenizer=None, num_samples=4):
    """
    Megjelenít néhány példát az adathalmazból
    
    Args:
        images: numpy tömb a képekkel
        latex_equations: numpy tömb a címkékkel (tokenized)
        tokenizer: tokenizer a dekódoláshoz
        num_samples: megjelenítendő példák száma
    """
    # Ellenőrizzük, hogy van-e elég példa
    batch_size = min(num_samples, len(images))
    
    plt.figure(figsize=(15, 5*batch_size))
    
    for i in range(batch_size):
        plt.subplot(batch_size, 1, i+1)
        
        # Kép megjelenítése
        img = images[i]
        if img.shape[-1] == 1:  # Ha szürkeárnyalatos
            plt.imshow(np.squeeze(img), cmap='gray')
        else:
            plt.imshow(img)
        
        # LaTeX egyenlet megjelenítése
        if tokenizer is not None:
            # Tokenekből szöveggé
            index_to_word = {v: k for k, v in tokenizer.word_index.items()}
            tokens = [index_to_word.get(idx, '') for idx in latex_equations[i] if idx > 0]
            latex_text = ' '.join(tokens)
        else:
            latex_text = f"Szekvencia: {latex_equations[i]}"
            
        plt.title(f"Példa {i+1}: {latex_text}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_data.png')
    plt.show()


def calculate_metrics(y_true, y_pred, tokenizer):
    """
    Számoljunk metrikákat a modell teljesítményének értékeléséhez
    
    Args:
        y_true: valós címkék
        y_pred: jósolt címkék
        tokenizer: tokenizer a dekódoláshoz
    
    Returns:
        dict: metrikák szótára
    """
    # Tokenek dekódolása
    index_to_word = {v: k for k, v in tokenizer.word_index.items()}
    
    exact_match = 0
    char_match = 0
    total_chars = 0
    
    for i in range(len(y_true)):
        true_tokens = [index_to_word.get(idx, '') for idx in y_true[i] if idx > 0]
        pred_tokens = [index_to_word.get(idx, '') for idx in y_pred[i] if idx > 0]
        
        true_text = ' '.join(true_tokens)
        pred_text = ' '.join(pred_tokens)
        
        # Pontos egyezés
        if true_text == pred_text:
            exact_match += 1
        
        # Karakter szintű egyezés
        true_chars = list(true_text)
        pred_chars = list(pred_text)
        
        for j in range(min(len(true_chars), len(pred_chars))):
            if true_chars[j] == pred_chars[j]:
                char_match += 1
        
        total_chars += len(true_chars)
    
    # Metrikák számítása
    metrics = {
        'exact_match_accuracy': exact_match / len(y_true),
        'character_accuracy': char_match / total_chars if total_chars > 0 else 0
    }
    
    return metrics


def check_gpu_availability():
    """
    Ellenőrzi a GPU elérhetőségét és információkat szolgáltat róla
    """
    gpus = tf.config.list_physical_devices('GPU')
    
    if not gpus:
        print("Nem található GPU. A modell CPU-n fog futni, ami jelentősen lassabb lehet.")
        return False
    
    print(f"Talált GPU eszközök: {len(gpus)}")
    
    for i, gpu in enumerate(gpus):
        print(f"GPU {i+1}: {gpu.name}")
        
        # GPU információk lekérése
        try:
            gpu_details = tf.config.experimental.get_device_details(gpu)
            if 'compute_capability' in gpu_details:
                cc = gpu_details['compute_capability']
                print(f"  Compute Capability: {cc[0]}.{cc[1]}")
            
            # Memória információ
            memory_info = tf.config.experimental.get_memory_info(f'GPU:{i}')
            if memory_info:
                free_memory = memory_info['free'] / (1024**3)  # GB-ra konvertálás
                total_memory = memory_info['total'] / (1024**3)
                print(f"  Memória: {free_memory:.2f} GB szabad / {total_memory:.2f} GB összes")
        except:
            print("  Részletes GPU információk nem elérhetőek")
    
    return True


def create_directory_structure():
    """
    Létrehozza a szükséges könyvtárszerkezetet
    """
    directories = [
        "data/crohme",
        "data/im2latex",
        "data/combined",
        "models",
        "logs",
        "results"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Könyvtár létrehozva: {directory}")
    
    # README fájl létrehozása minden mappában
    for directory in directories:
        readme_path = os.path.join(directory, "README.md")
        if not os.path.exists(readme_path):
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(f"# {directory.split('/')[-1].title()} könyvtár\n\n")
                
                if "crohme" in directory:
                    f.write("Ide kerülnek a CROHME adathalmaz fájljai.\n")
                    f.write("Letöltés: https://www.isical.ac.in/~crohme/\n")
                elif "im2latex" in directory:
                    f.write("Ide kerülnek az im2latex adathalmaz fájljai.\n")
                    f.write("Letöltés: https://github.com/harvardnlp/im2markup\n")
                elif "combined" in directory:
                    f.write("Ide kerülnek a feldolgozott és egyesített adathalmazok.\n")
                elif "models" in directory:
                    f.write("Ide kerülnek a betanított modellek.\n")
                elif "logs" in directory:
                    f.write("Ide kerülnek a tanítási logok és tensorboard adatok.\n")
                elif "results" in directory:
                    f.write("Ide kerülnek a kiértékelési és tesztelési eredmények.\n")
