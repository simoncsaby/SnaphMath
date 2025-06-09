from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import io
import base64
import os
from datetime import datetime
import urllib.parse
from math_solver import MathSolver

# Flask app inicializálás
app = Flask(__name__)
CORS(app)

# Konfiguráció
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max 16MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Math solver inicializálása (ez gyors)
math_solver = MathSolver()

# LAZY LOADING - EasyOCR csak akkor töltődik be, amikor szükség van rá
_math_recognizer = None

def get_math_recognizer():
    """Lazy loading EasyOCR - csak első használatkor"""
    global _math_recognizer
    
    if _math_recognizer is None:
        print("EasyOCR inicializálása... (ez ~30-60 másodpercet vehet igénybe)")
        try:
            from math_recognizer import MathEquationRecognizer
            _math_recognizer = MathEquationRecognizer(use_gpu=False, handwritten=False)
            print("EasyOCR sikeresen inicializálva!")
        except Exception as e:
            print(f"EasyOCR inicializálási hiba: {e}")
            raise
    
    return _math_recognizer

def allowed_file(filename):
    """Ellenőrzi, hogy a fájl kiterjesztése engedélyezett-e"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def decode_base64_image(base64_string):
    """Base64 kódolt képet dekódol numpy array-be"""
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))
    img_array = np.array(img)
    
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    return img_array

@app.route('/', methods=['GET'])
def home():
    """Főoldal - API információk"""
    return jsonify({
        'name': 'Math OCR + Solver API',
        'version': '2.1',
        'status': 'online',
        'ocr_status': 'lazy_loaded',  # Jelzi hogy lazy loading
        'endpoints': {
            '/recognize': 'POST - Matematikai egyenlet felismerése képről',
            '/solve': 'POST - Matematikai egyenlet megoldása szövegből',
            '/full_solve': 'POST - Kép feldolgozása és egyenlet megoldása',
            '/health': 'GET - API állapot'
        },
        'memory_optimization': 'enabled'
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Állapot ellenőrzés"""
    ocr_status = 'not_loaded' if _math_recognizer is None else 'loaded'
    
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'components': {
            'ocr': ocr_status,
            'solver': 'available',
            'wolfram_integration': 'available'
        },
        'memory_info': {
            'lazy_loading': True,
            'workers': 1
        }
    })

# WARMUP endpoint - OCR előzetes betöltéséhez
@app.route('/warmup', methods=['POST'])
def warmup_ocr():
    """EasyOCR előzetes betöltése (opcionális)"""
    try:
        print("Warmup: EasyOCR betöltése...")
        recognizer = get_math_recognizer()
        return jsonify({
            'success': True,
            'message': 'EasyOCR sikeresen betöltve',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Warmup hiba: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/solve', methods=['POST'])
def solve_equation():
    """Matematikai egyenlet megoldása szövegből - MEMÓRIA OPTIMALIZÁLT"""
    try:
        if not request.is_json:
            return jsonify({'error': 'JSON formátum szükséges'}), 400
        
        data = request.get_json()
        if 'equation' not in data:
            return jsonify({'error': 'Hiányzó egyenlet'}), 400
        
        equation = data['equation']
        include_latex = data.get('include_latex', True)
        
        # Egyenlet megoldása - ez nem használ sok memóriát
        result = math_solver.solve_step_by_step(equation)
        
        if result['success']:
            wolfram_query = urllib.parse.quote(equation)
            wolfram_url = f"https://www.wolframalpha.com/input/?i={wolfram_query}"
            
            response = {
                'success': True,
                'original_equation': equation,
                'equation_type': result['type'],
                'solutions': result.get('solutions', []),
                'steps': result['steps'],
                'wolfram_url': wolfram_url,
                'timestamp': datetime.now().isoformat()
            }
            
            # LaTeX támogatás elhagyása ha nem kérik
            if not include_latex:
                for step in response['steps']:
                    if 'latex' in step:
                        del step['latex']
            
            return jsonify(response)
        else:
            return jsonify({
                'success': False,
                'error': result['error'],
                'original_equation': equation,
                'timestamp': datetime.now().isoformat()
            }), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/recognize', methods=['POST'])
def recognize_equation():
    """Matematikai egyenlet felismerése - LAZY LOADING"""
    try:
        # CSAK AKKOR tölti be az EasyOCR-t, amikor szükség van rá
        recognizer = get_math_recognizer()
        
        image = None
        handwritten = False
        
        # JSON formátum (base64)
        if request.is_json:
            data = request.get_json()
            if 'image' not in data:
                return jsonify({'error': 'Hiányzó kép adat'}), 400
            
            image = decode_base64_image(data['image'])
            handwritten = data.get('handwritten', False)
        
        # Multipart form data
        elif 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'Nincs kiválasztott fájl'}), 400
            
            if file and allowed_file(file.filename):
                img_stream = io.BytesIO(file.read())
                img = Image.open(img_stream)
                image = np.array(img)
                
                if len(image.shape) == 3 and image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                
                handwritten = request.form.get('handwritten', 'false').lower() == 'true'
            else:
                return jsonify({'error': 'Nem támogatott fájlformátum'}), 400
        else:
            return jsonify({'error': 'Hiányzó kép adat'}), 400
        
        # OCR feldolgozás
        result = recognizer.recognize(image)
        
        if result['success']:
            wolfram_query = urllib.parse.quote(result['wolfram_format'])
            wolfram_url = f"https://www.wolframalpha.com/input/?i={wolfram_query}"
            
            return jsonify({
                'success': True,
                'equation': result['equation'],
                'wolfram_format': result['wolfram_format'],
                'wolfram_url': wolfram_url,
                'confidence': result['confidence'],
                'handwritten_mode': handwritten,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'success': False,
                'error': result['error'],
                'timestamp': datetime.now().isoformat()
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/full_solve', methods=['POST'])
def full_solve():
    """Teljes folyamat: OCR + megoldás - MEMÓRIA OPTIMALIZÁLT"""
    try:
        # EasyOCR lazy loading
        recognizer = get_math_recognizer()
        
        image = None
        handwritten = False
        include_latex = True
        
        # Kép feldolgozás (ugyanaz mint eddig)
        if request.is_json:
            data = request.get_json()
            if 'image' not in data:
                return jsonify({'error': 'Hiányzó kép adat'}), 400
            
            image = decode_base64_image(data['image'])
            handwritten = data.get('handwritten', False)
            include_latex = data.get('include_latex', True)
        
        elif 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'Nincs kiválasztott fájl'}), 400
            
            if file and allowed_file(file.filename):
                img_stream = io.BytesIO(file.read())
                img = Image.open(img_stream)
                image = np.array(img)
                
                if len(image.shape) == 3 and image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                
                handwritten = request.form.get('handwritten', 'false').lower() == 'true'
                include_latex = request.form.get('include_latex', 'true').lower() == 'true'
            else:
                return jsonify({'error': 'Nem támogatott fájlformátum'}), 400
        else:
            return jsonify({'error': 'Hiányzó kép adat'}), 400
        
        # 1. lépés: OCR felismerés
        ocr_result = recognizer.recognize(image)
        
        if not ocr_result['success']:
            return jsonify({
                'success': False,
                'error': f"OCR hiba: {ocr_result['error']}",
                'stage': 'ocr',
                'timestamp': datetime.now().isoformat()
            }), 500
        
        equation = ocr_result['equation']
        
        # 2. lépés: Egyenlet megoldása
        solve_result = math_solver.solve_step_by_step(equation)
        
        if not solve_result['success']:
            wolfram_query = urllib.parse.quote(ocr_result['wolfram_format'])
            wolfram_url = f"https://www.wolframalpha.com/input/?i={wolfram_query}"
            
            return jsonify({
                'success': False,
                'error': f"Megoldási hiba: {solve_result['error']}",
                'stage': 'solving',
                'ocr_result': {
                    'equation': equation,
                    'wolfram_format': ocr_result['wolfram_format'],
                    'wolfram_url': wolfram_url,
                    'confidence': ocr_result['confidence']
                },
                'timestamp': datetime.now().isoformat()
            }), 400
        
        # Sikeres teljes feldolgozás
        wolfram_query = urllib.parse.quote(ocr_result['wolfram_format'])
        wolfram_url = f"https://www.wolframalpha.com/input/?i={wolfram_query}"
        
        response = {
            'success': True,
            'ocr_result': {
                'equation': equation,
                'wolfram_format': ocr_result['wolfram_format'],
                'confidence': ocr_result['confidence']
            },
            'solution': {
                'equation_type': solve_result['type'],
                'solutions': solve_result.get('solutions', []),
                'steps': solve_result['steps']
            },
            'wolfram_url': wolfram_url,
            'handwritten_mode': handwritten,
            'timestamp': datetime.now().isoformat()
        }
        
        # LaTeX támogatás elhagyása ha nem kérik
        if not include_latex:
            for step in response['solution']['steps']:
                if 'latex' in step:
                    del step['latex']
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'stage': 'general',
            'timestamp': datetime.now().isoformat()
        }), 500

# Hiba handlerek
@app.errorhandler(413)
def request_entity_too_large(e):
    return jsonify({'error': 'A fájl túl nagy (max 16MB)'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Végpont nem található'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Szerver hiba'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)