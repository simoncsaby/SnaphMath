import multiprocessing

# Worker timeout növelése
timeout = 120  # 2 perc (EasyOCR inicializáláshoz)
keepalive = 2

# Worker számának csökkentése (memória optimalizálás)
workers = 1  # Csak 1 worker (kevés RAM miatt)
worker_class = "sync"
worker_connections = 1000

# Memória limit
max_requests = 100  # Worker restart 100 kérés után
max_requests_jitter = 10

# Timeout-ok
graceful_timeout = 30
preload_app = True  # App előre betöltése

print("Gunicorn konfiguráció betöltve - EasyOCR optimalizált")
