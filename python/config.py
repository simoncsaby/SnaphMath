# Adathalmazok elérési útjai
IM2LATEX_DATA_PATH = 'data/im2latex/'
COMBINED_DATA_PATH = 'data/combined/'

# Modell mentési útja
MODEL_SAVE_PATH = 'models/math_equation_model.h5'
TOKENIZER_PATH = 'models/tokenizer.pkl'

# Tanítási paraméterek
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.0001
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15
RANDOM_STATE = 42

# Képparaméterek
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 64
IMAGE_CHANNELS = 1

# Modell paraméterek
MAX_SEQUENCE_LENGTH = 150
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
ATTENTION_DIM = 512
VOCAB_SIZE = 5000  # Ezt a tokenizer alapján frissítjük

# GPU beállítások
USE_GPU = True
GPU_MEMORY_LIMIT = 0.8  # GPU memória használatának korlátozása (0-1 között)