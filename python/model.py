import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Reshape, Dense, LSTM, Dropout,
    BatchNormalization, Bidirectional, Attention, Embedding, 
    TimeDistributed, Flatten, concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
import matplotlib.pyplot as plt
from config import (
    IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS, BATCH_SIZE, EPOCHS, 
    LEARNING_RATE, EMBEDDING_DIM, HIDDEN_DIM, MAX_SEQUENCE_LENGTH,
    MODEL_SAVE_PATH, GPU_MEMORY_LIMIT, USE_GPU
)
from data_processing import load_or_process_data

# GPU memória konfiguráció
if USE_GPU:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU memória dinamikus növekedés engedélyezve")
        except RuntimeError as e:
            print(f"GPU konfigurációs hiba: {e}")


def build_model(vocab_size):
    """
    Képfelismerő és szekvencia generáló modell felépítése
    - CNN alapú encoder
    - LSTM alapú decoder
    """
    # Kép bemenet
    image_input = Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), name="image_input")
    
    # CNN encoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(image_input)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Kép jellemzők átalakítása szekvenciává
    # A CNN kimenete (batch_size, height, width, channels)
    new_shape = ((IMAGE_HEIGHT // 16) * (IMAGE_WIDTH // 16), 256)
    x = Reshape(target_shape=new_shape)(x)
    
    # LSTM encoder
    encoder = Bidirectional(LSTM(HIDDEN_DIM, return_sequences=True))(x)
    encoder_last = Bidirectional(LSTM(HIDDEN_DIM))(x)
    
    # Decoder - Ez fogja generálni a LaTeX kimenetet
    decoder_input = Input(shape=(MAX_SEQUENCE_LENGTH,), name="decoder_input")
    
    # Embedding réteg a tokenek számára
    embedding = Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM)(decoder_input)
    
    # LSTM decoder
    decoder = LSTM(HIDDEN_DIM * 2, return_sequences=True)(embedding, initial_state=[encoder_last, encoder_last])
    
    # Kimenet előrejelzés
    output = TimeDistributed(Dense(vocab_size, activation='softmax'))(decoder)
    
    # Modell összeállítása
    model = Model(inputs=[image_input, decoder_input], outputs=output)
    
    # Optimalizáló és veszteségfüggvény
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Modell struktúra kiírása
    model.summary()
    
    return model


def train_model():
    """Modell tanítása"""
    print("Adatok betöltése...")
    X_train, X_val, X_test, y_train, y_val, y_test, tokenizer = load_or_process_data()
    
    # Valós szókincs méret
    vocab_size = min(len(tokenizer.word_index) + 1, tokenizer.num_words)
    print(f"Szókincs mérete: {vocab_size}")
    
    # Decoder input előkészítése (eltolt szekvenciák)
    # Itt egy <START> token-t adunk a szekvencia elejére, hogy a decoder tudja, honnan kell kezdenie
    decoder_input_train = np.zeros_like(y_train)
    decoder_input_train[:, 1:] = y_train[:, :-1]
    decoder_input_train[:, 0] = vocab_size - 1  # Speciális <START> token
    
    decoder_input_val = np.zeros_like(y_val)
    decoder_input_val[:, 1:] = y_val[:, :-1]
    decoder_input_val[:, 0] = vocab_size - 1
    
    # Modell építése
    print("Modell építése...")
    model = build_model(vocab_size)
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        TensorBoard(
            log_dir='./logs',
            histogram_freq=1,
            write_graph=True
        )
    ]
    
    # Könyvtár létrehozása a modell számára, ha még nem létezik
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    # Modell tanítása
    print("Modell tanítása...")
    history = model.fit(
        [X_train, decoder_input_train],
        np.expand_dims(y_train, -1),  # sparse_categorical_crossentropy miatt kibővítjük
        validation_data=([X_val, decoder_input_val], np.expand_dims(y_val, -1)),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Tanulási görbék megjelenítése
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Modell veszteség')
    plt.ylabel('Veszteség')
    plt.xlabel('Epoch')
    plt.legend(['Tanító', 'Validációs'], loc='upper right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Modell pontosság')
    plt.ylabel('Pontosság')
    plt.xlabel('Epoch')
    plt.legend(['Tanító', 'Validációs'], loc='lower right')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    
    # Modell kiértékelése a teszt halmazon
    print("Modell kiértékelése a teszt halmazon...")
    decoder_input_test = np.zeros_like(y_test)
    decoder_input_test[:, 1:] = y_test[:, :-1]
    decoder_input_test[:, 0] = vocab_size - 1
    
    test_loss, test_acc = model.evaluate(
        [X_test, decoder_input_test],
        np.expand_dims(y_test, -1),
        batch_size=BATCH_SIZE,
        verbose=1
    )
    
    print(f"Teszt veszteség: {test_loss:.4f}")
    print(f"Teszt pontosság: {test_acc:.4f}")
    
    return model, tokenizer


def load_trained_model():
    """Betölti a betanított modellt"""
    try:
        model = load_model(MODEL_SAVE_PATH)
        print(f"Modell betöltve: {MODEL_SAVE_PATH}")
        return model
    except:
        print(f"Nem sikerült betölteni a modellt: {MODEL_SAVE_PATH}")
        return None
