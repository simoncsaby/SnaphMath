import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import pickle
import glob

class MathSymbolTester:
    def __init__(self, model_path, crohme_root=None):
        """
        Inicializálja a felismerő rendszert
        
        Args:
            model_path: A betanított modell elérési útja
            crohme_root: A CROHME adatkészlet főkönyvtára (opcionális)
        """
        print("Modell betöltése...")
        self.model = load_model(model_path)
        self.img_size = 64  # Alapértelmezett képméret
        
        # Osztálynevek létrehozása
        self.class_names = self.create_class_mapping(crohme_root)
        print(f"Osztályok száma: {len(self.class_names)}")
    
    def create_class_mapping(self, crohme_root=None):
        """
        Létrehozza az osztályok leképezését index -> név
        
        Args:
            crohme_root: A CROHME adatkészlet főkönyvtára
            
        Returns:
            Osztályok leképezése (dict)
        """
        # Ha nincs megadva CROHME elérési út, manuálisan definiáljuk a gyakori karaktereket
        if crohme_root is None or not os.path.isdir(crohme_root):
            print("CROHME elérési út nincs megadva vagy nem létezik. Alapértelmezett osztálynevek használata.")
            
            # Alapértelmezett osztálynév lista - ezeket általában tartalmazza a CROHME
            # Az indexek 0-tól indulnak, így a 0. elem a '0' osztály, stb.
            default_classes = [
                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                'a', 'alpha', 'b', 'beta', 'cos', 'd', 'Delta', 'div', 'e', 'equals',
                'f', 'forall', 'g', 'gamma', 'geq', 'gt', 'h', 'i', 'in', 'infty',
                'int', 'k', 'l', 'lambda', 'leq', 'lim', 'log', 'lt', 'm', 'mu',
                'n', 'neq', 'o', 'p', 'phi', 'pi', 'pm', 'q', 'r', 's',
                'sigma', 'sin', 'sqrt', 'sum', 't', 'tan', 'theta', 'times', 'u', 'v',
                'w', 'x', 'y', 'z', '+', '-', '(', ')', '[', ']'
            ]
            
            # Ha a modellünk kevesebb osztályt tartalmaz, vágjuk le a listát
            output_shape = self.model.output_shape[1]
            if output_shape < len(default_classes):
                default_classes = default_classes[:output_shape]
            
            # Ha a modellünk több osztályt tartalmaz, töltsük fel generikus nevekkel
            while len(default_classes) < output_shape:
                default_classes.append(f"class_{len(default_classes)}")
            
            return {i: name for i, name in enumerate(default_classes)}
        
        # CROHME könyvtárból osztálynevek kiolvasása
        class_mapping = {}
        symbol_dirs = sorted([d for d in os.listdir(crohme_root) if os.path.isdir(os.path.join(crohme_root, d))])
        
        for i, symbol_dir in enumerate(symbol_dirs):
            # DIR utótag eltávolítása, ha van
            clean_name = symbol_dir.replace("DIR", "")
            class_mapping[i] = clean_name
        
        return class_mapping
    
    def preprocess_image(self, image_path):
        """
        Előfeldolgozza a képet a modell számára
        
        Args:
            image_path: A kép elérési útja
            
        Returns:
            A feldolgozott kép
        """
        # Kép betöltése
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Nem sikerült betölteni a képet: {image_path}")
        
        # Binarizálás Otsu módszerrel
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Vonalak vastagítása (opcionális)
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)
        
        # Átméretezés
        resized = cv2.resize(dilated, (self.img_size, self.img_size))
        
        # Normalizálás
        normalized = resized / 255.0
        
        return normalized
    
    def predict(self, image_path, show_top_n=3):
        """
        Felismer egy karaktert egy képről
        
        Args:
            image_path: A kép elérési útja
            show_top_n: Hány legjobb predikciót mutasson
            
        Returns:
            A legjobb predikció és konfidencia
        """
        # Kép előfeldolgozása
        processed_img = self.preprocess_image(image_path)
        
        # Predikció
        input_data = processed_img.reshape(1, self.img_size, self.img_size, 1)
        predictions = self.model.predict(input_data, verbose=0)[0]
        
        # Legjobb találatok
        top_indices = np.argsort(predictions)[-show_top_n:][::-1]
        top_predictions = [(self.class_names[idx], predictions[idx]) for idx in top_indices]
        
        # Eredeti kép betöltése a vizualizációhoz
        original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Eredmények vizualizálása
        plt.figure(figsize=(12, 6))
        
        # Eredeti kép
        plt.subplot(1, 2, 1)
        plt.imshow(original_img, cmap='gray')
        plt.title("Eredeti kép")
        plt.axis('off')
        
        # Feldolgozott kép
        plt.subplot(1, 2, 2)
        plt.imshow(processed_img, cmap='gray')
        plt.title("Feldolgozott kép")
        plt.axis('off')
        
        # Predikciók kiíratása
        print("\nFelismerési eredmények:")
        print("-" * 40)
        for i, (class_name, confidence) in enumerate(top_predictions):
            print(f"{i+1}. {class_name}: {confidence:.4f} ({confidence*100:.1f}%)")
            # Adjunk hozzá szöveget a képhez
            plt.figtext(0.5, 0.3 - i*0.05, f"{i+1}. {class_name}: {confidence*100:.1f}%", 
                       ha='center', fontsize=12, bbox={"facecolor":"white", "alpha":0.8, "pad":5})
        
        plt.tight_layout()
        plt.show()
        
        return top_predictions[0]
    
    def batch_test(self, directory_path, pattern='*.png'):
        """
        Tesztel egy mappában található összes képet
        
        Args:
            directory_path: A képeket tartalmazó mappa elérési útja
            pattern: Fájlminták (pl: '*.png', '*.jpg')
        """
        # Képek keresése a megadott mintával
        image_paths = glob.glob(os.path.join(directory_path, pattern))
        
        if not image_paths:
            print(f"Nem találhatók képek a következő helyen: {directory_path} (minta: {pattern})")
            return
        
        print(f"{len(image_paths)} kép található, teszt indítása...\n")
        
        for i, image_path in enumerate(image_paths):
            print(f"Teszt {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
            try:
                self.predict(image_path)
            except Exception as e:
                print(f"Hiba a kép feldolgozása során: {str(e)}")
            print("\n" + "-" * 50 + "\n")

# Fő program
if __name__ == "__main__":
    # Modell betöltése
    model_path = 'models/simple_line_model.h5'
    
    # CROHME adatkészlet elérési útja (ha van)
    crohme_root = None  # Cserélje le a valós elérési útra, ha elérhető
    
    # Tesztelő inicializálása
    tester = MathSymbolTester(model_path, crohme_root)
    
    # Felhasználói interfész
    while True:
        print("\nVálasszon a következő opciók közül:")
        print("1: Egyetlen karakter felismerése")
        print("2: Batch teszt egy mappában")
        print("0: Kilépés")
        
        choice = input("Választás: ")
        
        if choice == "1":
            image_path = input("Adja meg a kép elérési útját: ")
            if os.path.isfile(image_path):
                tester.predict(image_path, show_top_n=5)
            else:
                print(f"A fájl nem létezik: {image_path}")
        
        elif choice == "2":
            directory = input("Adja meg a mappa elérési útját: ")
            if os.path.isdir(directory):
                pattern = input("Adja meg a fájlmintát (alapértelmezett: *.png): ") or "*.png"
                tester.batch_test(directory, pattern)
            else:
                print(f"A mappa nem létezik: {directory}")
        
        elif choice == "0":
            print("Kilépés...")
            break
        
        else:
            print("Érvénytelen választás!")