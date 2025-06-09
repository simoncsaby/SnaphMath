# Math OCR + Solver API

Matematikai egyenletek felismerése képekről OCR-rel és azonnali lépésenkénti megoldás SymPy segítségével.

## Funkciók

- **OCR felismerés** - EasyOCR alapú egyenlet felismerés képekről
- **Automatikus megoldás** - Lépésenkénti matematikai megoldás
- **Kézírásos támogatás** - Nyomtatott és kézírásos egyenletek
- **LaTeX kimenet** - Gyönyörű matematikai formázás
- **Wolfram Alpha integráció** - Azonnali ellenőrzés
- **REST API** - Könnyű integráció bármilyen alkalmazásba

## Gyors telepítés

### Lokális futtatás
```bash
git clone <repo-url>
cd math-ocr-solver-api
pip install -r requirements.txt
python app.py
```

### Hosting (Railway/Render/Heroku)
1. Fork-old ezt a repót
2. Csatlakoztasd a hosting szolgáltatáshoz
3. Automatikus deployment
4. **Memória**: Minimum 1GB RAM ajánlott (EasyOCR modellek miatt)

## API használat

### 1. Csak OCR felismerés

**Képfeltöltés (JSON)**
```bash
curl -X POST http://localhost:5000/recognize \
  -H "Content-Type: application/json" \
  -d '{
    "image": "base64_encoded_image_here",
    "handwritten": false
  }'
```

**Képfeltöltés (Form data)**
```bash
curl -X POST http://localhost:5000/recognize \
  -F "image=@equation.jpg" \
  -F "handwritten=false"
```

### 2. Csak egyenlet megoldás (szövegből)

```bash
curl -X POST http://localhost:5000/solve \
  -H "Content-Type: application/json" \
  -d '{
    "equation": "2*x + 5 = 11",
    "include_latex": true
  }'
```

### 3. Teljes folyamat: OCR + Megoldás

```bash
curl -X POST http://localhost:5000/full_solve \
  -H "Content-Type: application/json" \
  -d '{
    "image": "base64_encoded_image_here",
    "handwritten": false,
    "include_latex": true
  }'
```

## Válasz formátumok

### OCR felismerés válasz
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

### Egyenlet megoldás válasz
```json
{
  "success": true,
  "original_equation": "2*x + 5 = 11",
  "equation_type": "linear",
  "solutions": ["3"],
  "steps": [
    {
      "step": 1,
      "description": "Eredeti egyenlet",
      "expression": "Eq(2*x + 5, 11)",
      "latex": "2 x + 5 = 11"
    },
    {
      "step": 2,
      "description": "Minden tag átmozgatása bal oldalra",
      "expression": "2*x - 6 = 0",
      "latex": "2 x - 6 = 0"
    },
    {
      "step": 3,
      "description": "Megoldás x változóra",
      "expression": "x = 3",
      "latex": "x = 3"
    }
  ],
  "wolfram_url": "https://www.wolframalpha.com/input/?i=2*x%2B5%3D11",
  "timestamp": "2024-01-01T12:00:00"
}
```

### Teljes folyamat válasz
```json
{
  "success": true,
  "ocr_result": {
    "equation": "x^2 - 5x + 6 = 0",
    "wolfram_format": "x**2 - 5*x + 6 = 0",
    "confidence": "high"
  },
  "solution": {
    "equation_type": "quadratic",
    "solutions": ["2", "3"],
    "steps": [...]
  },
  "wolfram_url": "https://www.wolframalpha.com/input/?i=x**2-5*x%2B6%3D0",
  "handwritten_mode": false,
  "timestamp": "2024-01-01T12:00:00"
}
```

## Támogatott egyenlet típusok

### Alapvető típusok
- **Lineáris egyenletek**: `2x + 3 = 7`
- **Másodfokú egyenletek**: `x^2 - 5x + 6 = 0`
- **Polinomiális egyenletek**: `x^3 - 8 = 0`
- **Kifejezések**: `x^2 + 4x + 4` (faktorizálás)

### Speciális típusok
- **Trigonometrikus**: `sin(x) = 0.5`
- **Exponenciális**: `e^x = 10`
- **Logaritmikus**: `log(x) = 2`
- **Abszolút érték**: `|x - 3| = 2`
- **Egyenlőtlenségek**: `x^2 - 5x + 6 > 0`

## Támogatott formátumok

- **Képek**: PNG, JPG, JPEG, GIF, BMP, WEBP
- **Max méret**: 16MB
- **Típusok**: Nyomtatott és kézírásos egyenletek

## Endpoints

| Method | Endpoint | Leírás |
|--------|----------|---------|
| GET | `/` | API információk |
| GET | `/health` | Állapot ellenőrzés |
| GET | `/equation_types` | Támogatott egyenlet típusok |
| POST | `/recognize` | Csak OCR felismerés |
| POST | `/solve` | Csak egyenlet megoldás |
| POST | `/full_solve` | OCR + megoldás |
