import os
import argparse
import tensorflow as tf
from model import train_model, load_trained_model
from prediction import evaluate_on_test_image
from config import *


def setup_environment():
    """Környezet beállítása, GPU konfiguráció"""
    if USE_GPU:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                print(f"Talált GPU eszközök: {len(gpus)}")
                # Csak a dinamikus memória-növekedést használjuk
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("GPU memóriafoglalás dinamikus növekedésre állítva")
            except RuntimeError as e:
                print(f"GPU konfigurációs hiba: {e}")
        else:
            print("Nem található GPU. CPU használat.")
    else:
        print("GPU kikapcsolva a konfigurációban. CPU használat.")
        
    # Könyvtárak létrehozása, ha még nem léteznek
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(IM2LATEX_DATA_PATH, exist_ok=True)
    os.makedirs(COMBINED_DATA_PATH, exist_ok=True)


def main(mode="train", test_image=None):
    """Fő program"""
    setup_environment()
    
    if mode == "train":
        print("Matematikai egyenletfelismerő modell tanítása...")
        model, tokenizer = train_model()
        print(f"Model mentve: {MODEL_SAVE_PATH}")
        
    elif mode == "test":
        if test_image is None or not os.path.exists(test_image):
            print("Kérlek adj meg egy létező képfájlt a teszteléshez!")
            return
            
        print(f"Egyenlet felismerése a következő képről: {test_image}")
        latex_prediction = evaluate_on_test_image(test_image)
        print(f"Eredmény: {latex_prediction}")
        
    else:
        print(f"Ismeretlen mód: {mode}. Használd a 'train' vagy 'test' módot.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Matematikai egyenletfelismerő tanítása és tesztelése.")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help="Válassz 'train' (tanítás) vagy 'test' (tesztelés) módot.")
    parser.add_argument('--image', type=str, default=None, 
                        help="Tesztelés esetén az egyenletet tartalmazó kép elérési útja.")
    
    args = parser.parse_args()
    main(mode=args.mode, test_image=args.image)