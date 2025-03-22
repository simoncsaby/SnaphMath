import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import cv2
import pickle

# Egyszerűbb konfiguráció a vonalrajzokhoz
class Config:
    BATCH_SIZE = 32  # Kisebb batch méret
    EPOCHS = 100  # Több epoch a lassabb tanuláshoz
    LEARNING_RATE = 0.0002  # Nagyon alacsony tanulási ráta
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.15
    RANDOM_STATE = 42
    IMG_SIZE = 64  # Nagyobb képméret a vonalak jobb megőrzéséhez
    NUM_CHANNELS = 1
    MODEL_SAVE_PATH = 'models/simple_line_model.h5'
    LABEL_ENCODER_PATH = 'models/simple_line_encoder.pkl'
    
    # Mappa, ahol a CROHME adatkészlet található
    CROHME_ROOT = "archive"  # Cserélje ki a valós elérési úttal
    
    # GPU konfiguráció
    GPU_ID = 0
    MEMORY_GROWTH = True

# GPU konfiguráció
def configure_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Csak a kiválasztott GPU használata
            if len(gpus) > 1 and Config.GPU_ID < len(gpus):
                tf.config.set_visible_devices(gpus[Config.GPU_ID], 'GPU')
            
            # Memória növekedés engedélyezése
            if Config.MEMORY_GROWTH:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"{len(gpus)} Fizikai GPU és {len(logical_gpus)} Logikai GPU")
        except RuntimeError as e:
            print(e)
    else:
        print("Nem található GPU. CPU-n fog futni a tanítás.")

# Speciális előfeldolgozás a vonalrajzokhoz
def preprocess_image(image_path):
    # Kép betöltése
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    # Binarizálás Otsu módszerrel a jobb élkiemeléshez
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morfológiai műveletek a vonalak megerősítéséhez
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    
    # Zaj eltávolítása
    denoised = cv2.medianBlur(dilated, 3)
    
    # Átméretezés
    resized = cv2.resize(denoised, (Config.IMG_SIZE, Config.IMG_SIZE))
    
    # Normalizálás
    normalized = resized / 255.0
    
    return normalized

# CROHME adatkészlet betöltése a vonalrajzokhoz optimalizálva
def load_crohme_dataset():
    images = []
    labels = []
    label_mapping = {}
    class_counts = {}
    
    print("CROHME adatkészlet betöltése vonalrajz optimalizációval...")
    
    # Szimbólumdirektóriumok
    symbol_dirs = [d for d in os.listdir(Config.CROHME_ROOT) if os.path.isdir(os.path.join(Config.CROHME_ROOT, d))]
    
    # Címke leképezés
    for i, symbol_dir in enumerate(sorted(symbol_dirs)):
        label_mapping[symbol_dir] = i
        clean_name = symbol_dir.replace("DIR", "")
        print(f"Címke hozzárendelése: {clean_name} -> {i}")
        class_counts[clean_name] = 0
    
    # Képek betöltése
    for symbol_dir in symbol_dirs:
        symbol_path = os.path.join(Config.CROHME_ROOT, symbol_dir)
        label_idx = label_mapping[symbol_dir]
        clean_name = symbol_dir.replace("DIR", "")
        
        # Rekurzív bejárás
        for root, _, files in os.walk(symbol_path):
            for file in files:
                if file.endswith(('.png', '.jpg')):
                    file_path = os.path.join(root, file)
                    
                    # Speciális előfeldolgozás
                    processed_img = preprocess_image(file_path)
                    
                    if processed_img is not None:
                        images.append(processed_img)
                        labels.append(label_idx)
                        class_counts[clean_name] += 1
                        
                        # Adatkiterjesztés már itt
                        # Tükrözés vízszintesen - vonalaknál segíthet
                        flipped_h = cv2.flip(processed_img, 1)
                        images.append(flipped_h)
                        labels.append(label_idx)
                        class_counts[clean_name] += 1
    
    # Statisztikák
    print("\nAdatkészlet statisztikák:")
    print(f"Összes kép: {len(images)}")
    print(f"Osztályok száma: {len(label_mapping)}")
    
    for class_name, count in class_counts.items():
        print(f"  - {class_name}: {count} kép")
    
    return np.array(images), np.array(labels), label_mapping

# Adatkészlet előkészítése
def prepare_dataset(images, labels):
    # Adatok formázása
    x = np.expand_dims(images, axis=-1)
    
    # LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(labels)
    
    # Címke encoder mentése
    with open(Config.LABEL_ENCODER_PATH, 'wb') as f:
        pickle.dump(le, f)
    
    # Adatkészlet felosztása
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE, stratify=y
    )
    
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, 
        test_size=Config.VALIDATION_SIZE, 
        random_state=Config.RANDOM_STATE,
        stratify=y_train
    )
    
    print(f"Tanító adatok: {x_train.shape[0]} minta")
    print(f"Validációs adatok: {x_val.shape[0]} minta")
    print(f"Teszt adatok: {x_test.shape[0]} minta")
    
    return x_train, x_val, x_test, y_train, y_val, y_test, le

# Nagyon enyhe adatkiterjesztés a vonalrajzokhoz
def configure_data_augmentation():
    train_datagen = ImageDataGenerator(
        rotation_range=5,  # Minimális forgatás
        width_shift_range=0.05,  # Minimális eltolás
        height_shift_range=0.05,
        zoom_range=0.05,  # Minimális nagyítás
        fill_mode='constant',  # Konstans kitöltés fehérrel
        cval=0  # Fehér háttér
    )
    
    val_datagen = ImageDataGenerator()
    
    return train_datagen, val_datagen

# Egyszerűsített modell vonalrajzokhoz
def create_simple_model(num_classes):
    print("Egyszerűsített modell létrehozása vonalrajzokhoz...")
    
    model = Sequential([
        # Első konvolúciós blokk - kisebb szűrők a vonalak jobb detektálásához
        Conv2D(16, (3, 3), padding='same', activation='relu', 
               input_shape=(Config.IMG_SIZE, Config.IMG_SIZE, Config.NUM_CHANNELS)),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Második konvolúciós blokk
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Harmadik konvolúciós blokk - kevesebb réteg
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Teljesen kapcsolt rétegek - egyszerűbb struktúra
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    # Modell fordítása
    optimizer = Adam(learning_rate=Config.LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    return model

# Callback-ek konfigurálása
def configure_callbacks():
    # Korai leállítás
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,  # Hosszabb türelem
        restore_best_weights=True,
        verbose=1
    )
    
    # Modell mentése
    checkpoint = ModelCheckpoint(
        Config.MODEL_SAVE_PATH,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    # Tanulási ráta csökkentése
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,  # Erősebb csökkentés
        patience=10,
        min_lr=1e-6,
        verbose=1,
        cooldown=3
    )
    
    return [early_stopping, checkpoint, reduce_lr]

# Modell tanítása
def train_model(model, x_train, y_train, x_val, y_val, train_datagen):
    print("Modell tanítása...")
    
    # Adatkiterjesztés a tanító adatokra
    train_generator = train_datagen.flow(
        x_train, y_train,
        batch_size=Config.BATCH_SIZE
    )
    
    # Callback-ek
    callbacks = configure_callbacks()
    
    # Tanítás
    history = model.fit(
        train_generator,
        steps_per_epoch=len(x_train) // Config.BATCH_SIZE,
        epochs=Config.EPOCHS,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    return history

# Eredmények vizualizálása
def visualize_training(history, x_test, y_test, label_encoder, model):
    # Tanulási görbék
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Tanító pontosság')
    plt.plot(history.history['val_accuracy'], label='Validációs pontosság')
    plt.title('Modell pontosság')
    plt.xlabel('Epoch')
    plt.ylabel('Pontosság')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Tanító veszteség')
    plt.plot(history.history['val_loss'], label='Validációs veszteség')
    plt.title('Modell veszteség')
    plt.xlabel('Epoch')
    plt.ylabel('Veszteség')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('simple_training_curves.png')
    plt.show()
    
    # Tesztelés
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f"Teszt pontosság: {test_accuracy:.4f}")
    print(f"Teszt veszteség: {test_loss:.4f}")
    
    # Példa predikciók
    predictions = model.predict(x_test)
    pred_classes = np.argmax(predictions, axis=1)
    
    # Néhány mintakép megjelenítése
    plt.figure(figsize=(15, 10))
    n_samples = min(25, len(x_test))
    for i in range(n_samples):
        plt.subplot(5, 5, i+1)
        plt.imshow(x_test[i].reshape(Config.IMG_SIZE, Config.IMG_SIZE), cmap='gray')
        true_class = str(label_encoder.inverse_transform([y_test[i]])[0]).replace("DIR", "")
        pred_class = label_encoder.inverse_transform([pred_classes[i]])[0].replace("DIR", "")
        color = 'green' if y_test[i] == pred_classes[i] else 'red'
        plt.title(f"T: {true_class}\nP: {pred_class}", color=color, fontsize=8)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('simple_predictions.png')
    plt.show()

# Eredmények elemzése és mentése
def analyze_results(model, x_test, y_test, label_encoder):
    # Predikciók
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Osztályok
    class_names = [name.replace("DIR", "") for name in label_encoder.classes_]
    
    # Osztályozási jelentés
    from sklearn.metrics import classification_report, confusion_matrix
    
    print("\nOsztályozási jelentés:")
    report = classification_report(y_test, y_pred_classes, target_names=class_names)
    print(report)
    
    # Tévesztési mátrix
    conf_matrix = confusion_matrix(y_test, y_pred_classes)
    
    # Vizualizáljuk a tévesztési mátrixot
    plt.figure(figsize=(12, 10))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Tévesztési mátrix')
    plt.colorbar()
    
    # Címkék
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)
    
    # Értékek kiírása
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, conf_matrix[i, j],
                    horizontalalignment="center",
                    color="white" if conf_matrix[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('Valós címke')
    plt.xlabel('Predikált címke')
    plt.savefig('confusion_matrix.png')
    plt.show()

# Főprogram
def main():
    # GPU konfiguráció
    configure_gpu()
    
    # Mappák létrehozása
    os.makedirs(os.path.dirname(Config.MODEL_SAVE_PATH), exist_ok=True)
    
    # Adatok betöltése
    images, labels, label_mapping = load_crohme_dataset()
    
    # Adatkészlet előkészítése
    x_train, x_val, x_test, y_train, y_val, y_test, label_encoder = prepare_dataset(images, labels)
    
    # Data augmentation
    train_datagen, _ = configure_data_augmentation()
    
    # Modell létrehozása
    num_classes = len(label_mapping)
    model = create_simple_model(num_classes)
    
    # Modell tanítása
    history = train_model(model, x_train, y_train, x_val, y_val, train_datagen)
    
    # Eredmények vizualizálása
    visualize_training(history, x_test, y_test, label_encoder, model)
    
    # Eredmények elemzése
    analyze_results(model, x_test, y_test, label_encoder)

if __name__ == "__main__":
    main()