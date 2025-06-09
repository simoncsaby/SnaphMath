# ===== PROCFILE =====
web: gunicorn --config gunicorn.conf.py app:app

# ===== REQUIREMENTS.TXT (OPTIMALIZÁLT) =====
flask==3.0.0
flask-cors==4.0.0
opencv-python-headless==4.8.1.78
numpy==1.24.3
Pillow==10.1.0
sympy==1.12
easyocr==1.7.0
torch==2.1.0
torchvision==0.16.0
gunicorn==21.2.0

# ===== RUNTIME.TXT =====
python-3.11.6

# ===== .slugignore (Heroku optimalizálás) =====
*.pyc
__pycache__/
.git/
.gitignore
README.md
tests/
.env
*.log

# ===== RAILWAY.JSON (Railway konfigurációhoz) =====
{
  "build": {
    "buildCommand": "pip install -r requirements.txt"
  },
  "deploy": {
    "startCommand": "gunicorn --config gunicorn.conf.py app:app",
    "healthcheckPath": "/health",
    "healthcheckTimeout": 300
  },
  "variables": {
    "PYTHONUNBUFFERED": "1",
    "WEB_CONCURRENCY": "1"
  }
}

# ===== RENDER.YAML (Render konfigurációhoz) =====
services:
  - type: web
    name: math-ocr-solver
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn --config gunicorn.conf.py app:app
    plan: starter
    envVars:
      - key: PYTHONUNBUFFERED
        value: "1"
      - key: WEB_CONCURRENCY  
        value: "1"
    healthCheckPath: /health

# ===== DOCKERFILE (Docker optimalizáláshoz) =====
FROM python:3.11-slim

# Rendszer csomagok telepítése
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Munkaterület beállítása
WORKDIR /app

# Python függőségek
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App fájlok másolása
COPY . .

# Port deklarálása
EXPOSE 5000

# Gunicorn indítása
CMD ["gunicorn", "--config", "gunicorn.conf.py", "app:app"]

# ===== DOCKER-COMPOSE.YML (fejlesztéshez) =====
version: '3.8'
services:
  math-ocr-api:
    build: .
    ports:
      - "5000:5000"
    environment:
      - PYTHONUNBUFFERED=1
      - WEB_CONCURRENCY=1
    volumes:
      - .:/app
    command: python app.py

# ===== .GITIGNORE =====
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# celery beat schedule file
celerybeat-schedule

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# EasyOCR models cache
.EasyOCR/

# Hosting specific
.slugignore
railway.json
render.yaml