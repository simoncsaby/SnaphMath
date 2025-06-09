# Math OCR API

Matematikai egyenletek felismerése képekről EasyOCR segítségével.

## 🚀 Gyors telepítés

### Lokális futtatás
```bash
git clone <repo-url>
cd math-ocr-api
pip install -r requirements.txt
python app.py
```

### Hosting (Railway/Render/Heroku)
1. Fork-old ezt a repót
2. Csatlakoztasd a hosting szolgáltatáshoz
3. Automatikus deployment

## 📝 API használat

### Képfeltöltés (JSON)
```bash
curl -X POST http://localhost:5000/recognize \
  -H "Content-Type: application/json" \
  -d '{
    "image": "base64_encoded_image_here",
    "handwritten": false
  }'
```

### Képfeltöltés (Form data)
```bash
curl -X POST http://localhost:5000/recognize \
  -F "image=@equation.jpg" \
  -F "handwritten=false"
```

### Válasz formátum
```json
{
  "success": true,
  "equation": "2x + 3 = 7",
  "wolfram_format": "2*x + 3 = 7",
  "wolfram_url": "https://www.wolframalpha.com/input/?i=2*x%2B3%3D7",
  "confidence": "medium",
  "handwritten_mode": false,
  "timestamp": "2024-01-01T12:00:00"
}
```

## 🔧 Támogatott formátumok

- **Képek**: PNG, JPG, JPEG, GIF, BMP, WEBP
- **Max méret**: 16MB
- **Típusok**: Nyomtatott és kézírásos egyenletek

## 📋 Endpoints

- `GET /` - API információk
- `GET /health` - Állapot ellenőrzés  
- `POST /recognize` - Egyenlet felismerés

### Railway deploy
1. Menj a [railway.app](https://railway.app) oldalra
2. GitHub repo csatlakoztatása
3. Auto-deploy engedélyezése

### Render deploy  
1. Menj a [render.com](https://render.com) oldalra
2. "New Web Service" → GitHub repo
3. Python környezet automatikus felismerés

## 🛠️ Fejlesztés

```bash
# Virtuális környezet
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Függőségek telepítése
pip install -r requirements.txt

# Futtatás debug módban
export FLASK_ENV=development
python app.py
```

## 📦 Projekt struktúra

```
math-ocr-api/
├── app.py              # Flask alkalmazás
├── math_recognizer.py  # OCR logika
├── requirements.txt    # Python csomagok
├── Procfile           # Hosting konfig
└── README.md          # Ez a fájl
```

## ⚡ Teljesítmény tippek

- Használj `opencv-python-headless` csomagot szerver környezetben
- EasyOCR első futtatáskor letölti a modelleket (~100MB)
- Hosting szolgáltatásoknál figyelj a memória limitekre

## 🐛 Hibaelhárítás

**"Module not found" hiba**: 
```bash
pip install --upgrade pip
pip install -r requirements.txt