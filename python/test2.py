import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import os

# Konfiguráció
MODEL_PATH = "models/batchnorm_math_model.h5"  # Betanított modell útvonala
IMAGE_SIZE = (64, 64)  # A kép mérete

# Osztálynevek betöltése
def load_class_names():
    class_dirs = []
    DATA_ROOT = "./archive/"
    
    # DIR-re végződő mappák keresése
    for dir_name in os.listdir(DATA_ROOT):
        if dir_name.endswith('DIR'):
            class_dirs.append(dir_name)
    
    class_dirs.sort()
    return class_dirs

# Kép betöltése és előfeldolgozása
def load_and_preprocess_image(image_path):
    # Ellenőrizzük, hogy létezik-e a fájl
    if not os.path.exists(image_path):
        print(f"A fájl nem található: {image_path}")
        return None
    
    # Kép betöltése szürkeárnyalatosan
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Nem sikerült betölteni a képet: {image_path}")
        return None
    
    # Átméretezés
    img = cv2.resize(img, IMAGE_SIZE)
    
    # Normalizálás
    img = img / 255.0
    
    # Batch és csatorna dimenzió hozzáadása
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    
    return img

# Predikció elvégzése
def predict_symbol(model, image, class_names):
    # Előrejelzés
    pred = model.predict(image)[0]
    
    # Legjobb találat indexe és konfidenciája
    predicted_class_idx = np.argmax(pred)
    confidence = pred[predicted_class_idx]
    
    # A legjobb osztálynév
    predicted_class = class_names[predicted_class_idx]
    
    # Top 5 predikció
    top5_indices = np.argsort(pred)[-5:][::-1]
    top5_classes = [class_names[i] for i in top5_indices]
    top5_confidences = [pred[i] * 100 for i in top5_indices]
    
    return predicted_class, confidence, top5_classes, top5_confidences

# Eredmények megjelenítése
def display_results(image_path, image, predicted_class, confidence, top5_classes, top5_confidences):
    plt.figure(figsize=(12, 6))
    
    # Eredeti kép megjelenítése
    plt.subplot(1, 2, 1)
    orig_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    plt.imshow(orig_img, cmap='gray')
    plt.title(f"Predikció: {predicted_class}\nBizonyosság: {confidence*100:.2f}%")
    plt.axis('off')
    
    # Top 5 predikció ábrázolása
    plt.subplot(1, 2, 2)
    y_pos = np.arange(len(top5_classes))
    plt.barh(y_pos, top5_confidences, align='center')
    plt.yticks(y_pos, top5_classes)
    plt.xlabel('Konfidencia (%)')
    plt.title('Top 5 predikció')
    
    plt.tight_layout()
    plt.savefig('prediction_result.png')
    plt.show()

def main():
    # Parancssori argumentum ellenőrzése
    image_path = input("Kérlek, add meg a kép elérési útját: ")
    
    if not image_path:
        print("Nem adtál meg képet!")
        return
    
    # Osztálynevek betöltése
    print("Osztálynevek betöltése...")
    class_names = load_class_names()
    print(f"{len(class_names)} osztály betöltve.")
    
    # Modell betöltése
    print(f"Modell betöltése innen: {MODEL_PATH}...")
    try:
        model = load_model(MODEL_PATH)
        print("Modell sikeresen betöltve.")
    except Exception as e:
        print(f"Hiba a modell betöltésekor: {str(e)}")
        return
    
    # Kép betöltése
    print(f"Kép betöltése: {image_path}")
    image = load_and_preprocess_image(image_path)
    
    if image is None:
        return
    
    # Predikció
    print("Predikció folyamatban...")
    predicted_class, confidence, top5_classes, top5_confidences = predict_symbol(model, image, class_names)
    
    # Eredmény kiírása
    print(f"\nEredmény:")
    print(f"Predikált szimbólum: {predicted_class}")
    print(f"Bizonyosság: {confidence*100:.2f}%")
    print("\nTop 5 predikció:")
    for i in range(len(top5_classes)):
        print(f"{i+1}. {top5_classes[i]}: {top5_confidences[i]:.2f}%")
    
    # Vizualizáció
    display_results(image_path, image, predicted_class, confidence, top5_classes, top5_confidences)

if __name__ == "__main__":
    # TensorFlow figyelmeztetések kikapcsolása
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # GPU memória növekedés engedélyezése
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU eszköz elérhető: {gpus[0]}")
        except RuntimeError as e:
            print(f"GPU hiba: {e}")
    
    main()