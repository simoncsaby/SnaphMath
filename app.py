from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import io
import base64
import os
import time
import hashlib
from datetime import datetime
import urllib.parse
from math_solver import MathSolver

app = Flask(__name__)
CORS(app)

# Konfiguráció
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max 16MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

math_solver = MathSolver()

_math_recognizer = None

# Egyszerű cache
recognition_cache = {}
MAX_CACHE_SIZE = 100

# Teljesítmény metrikák
performance_metrics = {
    'total_requests': 0,
    'success_count': 0,
    'error_count': 0,
    'cache_hits': 0
}

def get_math_recognizer():
    """Lazy loading EasyOCR"""
    global _math_recognizer
    
    if _math_recognizer is None:
        print("EasyOCR inicializálása...")
        try:
            from math_recognizer import MathEquationRecognizer
            _math_recognizer = MathEquationRecognizer(use_gpu=False, handwritten=True)
            print("EasyOCR sikeresen inicializálva!")
        except Exception as e:
            print(f"EasyOCR inicializálási hiba: {e}")
            raise
    
    return _math_recognizer

def decode_base64_image(base64_string):
    """Base64 kódolt képet dekódol numpy array-be"""
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        img_data = base64.b64decode(base64_string)
        img = Image.open(io.BytesIO(img_data))
        img_array = np.array(img)
        
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
        return img_array
    except Exception as e:
        print(f"Dekódolási hiba: {e}")
        raise

def generate_image_hash(base64_string):
    """Kép hash generálása cache-hez"""
    sample = base64_string[:1000] if len(base64_string) > 1000 else base64_string
    return hashlib.sha256(sample.encode()).hexdigest()[:16]

def cache_recognition(image_hash, result):
    """Eredmény cache-elése"""
    global recognition_cache
    
    if len(recognition_cache) >= MAX_CACHE_SIZE:
        oldest_key = next(iter(recognition_cache))
        del recognition_cache[oldest_key]
    
    recognition_cache[image_hash] = result

def get_cached_recognition(image_hash):
    """Cache-ből való lekérés"""
    global performance_metrics
    
    if image_hash in recognition_cache:
        performance_metrics['cache_hits'] += 1
        return recognition_cache[image_hash]
    
    return None

def optimize_image_for_ocr(image_array):
    """Alapvető kép optimalizálás OCR-hez"""
    try:
        # Méret optimalizálás
        height, width = image_array.shape[:2]
        max_dimension = 1200
        
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image_array = cv2.resize(image_array, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Kontraszt javítása
        lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        return enhanced
    except Exception as e:
        print(f"Optimalizálási hiba: {e}")
        return image_array

@app.route('/', methods=['GET'])
def home():
    """Főoldal"""
    return jsonify({
        'name': 'Math OCR + Solver API',
        'version': '3.0-simplified',
        'status': 'online',
        'endpoints': {
            '/recognize': 'POST - Egyenlet felismerés',
            '/smart_recognize': 'POST - Intelligens felismerés',
            '/solve': 'POST - Egyenlet megoldás',
            '/health': 'GET - API állapot',
            '/metrics': 'GET - Teljesítmény metrikák'
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Állapot ellenőrzés"""
    global performance_metrics
    
    ocr_status = 'not_loaded' if _math_recognizer is None else 'loaded'
    
    success_rate = (
        performance_metrics['success_count'] / performance_metrics['total_requests']
        if performance_metrics['total_requests'] > 0 else 0
    )
    
    cache_hit_rate = (
        performance_metrics['cache_hits'] / performance_metrics['total_requests']
        if performance_metrics['total_requests'] > 0 else 0
    )
    
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'components': {
            'ocr': ocr_status,
            'solver': 'available',
            'cache': f"{len(recognition_cache)}/{MAX_CACHE_SIZE}"
        },
        'performance': {
            'total_requests': performance_metrics['total_requests'],
            'success_rate': round(success_rate, 3),
            'cache_hit_rate': round(cache_hit_rate, 3)
        }
    })

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Teljesítmény metrikák"""
    return jsonify({
        'success': True,
        'metrics': performance_metrics,
        'cache_status': {
            'size': len(recognition_cache),
            'max_size': MAX_CACHE_SIZE
        }
    })

@app.route('/recognize', methods=['POST'])
def recognize_equation():
    """Egyenlet felismerés"""
    global performance_metrics
    
    start_time = time.time()
    performance_metrics['total_requests'] += 1
    
    try:
        if not request.is_json:
            return jsonify({'error': 'JSON formátum szükséges'}), 400
        
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'Hiányzó kép adat'}), 400
        
        base64_image = data['image']
        preprocessed = data.get('preprocessed', False)
        
        # Cache ellenőrzés
        image_hash = generate_image_hash(base64_image)
        cached_result = get_cached_recognition(image_hash)
        
        if cached_result:
            processing_time = time.time() - start_time
            performance_metrics['success_count'] += 1
            
            return jsonify({
                **cached_result,
                'processing_time': round(processing_time, 3),
                'cached': True,
                'timestamp': datetime.now().isoformat()
            })
        
        # Kép dekódolása
        image = decode_base64_image(base64_image)
        
        # Optimalizálás ha szükséges
        if not preprocessed:
            image = optimize_image_for_ocr(image)
        
        # OCR feldolgozás
        recognizer = get_math_recognizer()
        result = recognizer.recognize(image)
        
        processing_time = time.time() - start_time
        
        if result['success']:
            performance_metrics['success_count'] += 1
            
            wolfram_query = urllib.parse.quote(result['wolfram_format'])
            wolfram_url = f"https://www.wolframalpha.com/input/?i={wolfram_query}"
            
            response_data = {
                'success': True,
                'equation': result['equation'],
                'wolfram_format': result['wolfram_format'],
                'wolfram_url': wolfram_url,
                'confidence': result['confidence'],
                'processing_time': round(processing_time, 3),
                'cached': False,
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache-elés
            cache_recognition(image_hash, response_data)
            
            return jsonify(response_data)
        else:
            performance_metrics['error_count'] += 1
            return jsonify({
                'success': False,
                'error': result['error'],
                'processing_time': round(processing_time, 3),
                'timestamp': datetime.now().isoformat()
            }), 500
            
    except Exception as e:
        performance_metrics['error_count'] += 1
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/smart_recognize', methods=['POST'])
def smart_recognition():
    """Intelligens felismerés több próbálkozással"""
    global performance_metrics
    
    start_time = time.time()
    performance_metrics['total_requests'] += 1
    
    try:
        data = request.get_json()
        base64_image = data['image']
        confidence_threshold = data.get('confidence_threshold', 0.7)
        multiple_attempts = data.get('multiple_attempts', True)
        
        # Cache ellenőrzés
        image_hash = generate_image_hash(base64_image)
        cached_result = get_cached_recognition(image_hash)
        
        if cached_result and cached_result.get('confidence', 0) >= confidence_threshold:
            processing_time = time.time() - start_time
            performance_metrics['success_count'] += 1
            
            return jsonify({
                **cached_result,
                'processing_time': round(processing_time, 3),
                'cached': True,
                'processing_method': 'cache'
            })
        
        image = decode_base64_image(base64_image)
        recognizer = get_math_recognizer()
        
        attempts = []
        best_result = None
        best_confidence = 0
        
        # 1. próba: eredeti kép
        result1 = recognizer.recognize(image)
        attempts.append('original')
        
        if result1['success']:
            confidence1 = result1.get('confidence', 0)
            if confidence1 > best_confidence:
                best_result = result1
                best_confidence = confidence1
        
        # 2. próba: optimalizált kép
        if multiple_attempts and best_confidence < confidence_threshold:
            optimized_image = optimize_image_for_ocr(image)
            result2 = recognizer.recognize(optimized_image)
            attempts.append('optimized')
            
            if result2['success']:
                confidence2 = result2.get('confidence', 0)
                if confidence2 > best_confidence:
                    best_result = result2
                    best_confidence = confidence2
        
        # 3. próba: binary threshold
        if multiple_attempts and best_confidence < confidence_threshold:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
            
            result3 = recognizer.recognize(binary_rgb)
            attempts.append('binary')
            
            if result3['success']:
                confidence3 = result3.get('confidence', 0)
                if confidence3 > best_confidence:
                    best_result = result3
                    best_confidence = confidence3
        
        processing_time = time.time() - start_time
        
        if best_result and best_result['success']:
            performance_metrics['success_count'] += 1
            
            wolfram_query = urllib.parse.quote(best_result['wolfram_format'])
            wolfram_url = f"https://www.wolframalpha.com/input/?i={wolfram_query}"
            
            response_data = {
                'success': True,
                'equation': best_result['equation'],
                'confidence': best_confidence,
                'wolfram_format': best_result['wolfram_format'],
                'wolfram_url': wolfram_url,
                'processing_time': round(processing_time, 3),
                'processing_method': 'smart_multi_attempt',
                'attempts_made': len(attempts),
                'attempts': attempts,
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache-elés ha jó a confidence
            if best_confidence >= confidence_threshold:
                cache_recognition(image_hash, response_data)
            
            return jsonify(response_data)
        else:
            performance_metrics['error_count'] += 1
            return jsonify({
                'success': False,
                'error': 'Nem sikerült felismerni megfelelő bizalommal',
                'best_confidence': best_confidence,
                'threshold': confidence_threshold,
                'attempts_made': len(attempts),
                'processing_time': round(processing_time, 3)
            }), 500
            
    except Exception as e:
        performance_metrics['error_count'] += 1
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/solve', methods=['POST'])
def solve_equation():
    """Egyenlet megoldása szövegből"""
    try:
        if not request.is_json:
            return jsonify({'error': 'JSON formátum szükséges'}), 400
        
        data = request.get_json()
        if 'equation' not in data:
            return jsonify({'error': 'Hiányzó egyenlet'}), 400
        
        equation = data['equation']
        
        # Egyenlet megoldása
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

@app.route('/check_quality', methods=['POST'])
def check_image_quality():
    """Képminőség ellenőrzés"""
    try:
        data = request.get_json()
        base64_image = data['image']
        
        image = decode_base64_image(base64_image)
        
        # Alapvető minőség metrikák
        height, width = image.shape[:2]
        
        # Élesség mérése
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Kontraszt mérése
        contrast = gray.std()
        
        # Méret ellenőrzés
        size_score = 1.0 if min(height, width) >= 200 else min(height, width) / 200
        
        # Összesített minőség pontszám
        quality_score = min(1.0, (sharpness / 100) * 0.4 + (contrast / 50) * 0.3 + size_score * 0.3)
        
        suitable_for_ocr = quality_score > 0.5
        
        recommendations = []
        if sharpness < 50:
            recommendations.append("Készítsd újra a képet élesebben")
        if contrast < 30:
            recommendations.append("Javítsd a megvilágítást")
        if min(height, width) < 200:
            recommendations.append("Készíts nagyobb felbontású képet")
        
        return jsonify({
            'success': True,
            'quality_score': round(quality_score, 3),
            'suitable_for_ocr': suitable_for_ocr,
            'metrics': {
                'sharpness': round(sharpness, 2),
                'contrast': round(contrast, 2),
                'resolution': f"{width}x{height}",
                'size_score': round(size_score, 3)
            },
            'recommendations': recommendations
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/optimize_image', methods=['POST'])
def optimize_image_endpoint():
    """Kép optimalizálás"""
    try:
        data = request.get_json()
        base64_image = data['image']
        
        image = decode_base64_image(base64_image)
        original_shape = image.shape
        
        # Optimalizálás
        optimized_image = optimize_image_for_ocr(image)
        
        # Visszakódolás base64-be
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(optimized_image, cv2.COLOR_RGB2BGR))
        optimized_base64 = base64.b64encode(buffer).decode('utf-8')
        
        size_reduction = max(0, (len(base64_image) - len(optimized_base64)) / len(base64_image))
        
        return jsonify({
            'success': True,
            'optimized_image': optimized_base64,
            'original_size': original_shape,
            'optimized_size': optimized_image.shape,
            'size_reduction': round(size_reduction, 3),
            'improvements': ['contrast_enhancement', 'resize_optimization']
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/cache/<image_hash>', methods=['GET'])
def get_cache(image_hash):
    """Cache-elt eredmény lekérése"""
    cached = get_cached_recognition(image_hash)
    
    if cached:
        return jsonify(cached)
    else:
        return jsonify({
            'success': False,
            'error': 'Cache miss'
        }), 404

@app.route('/formats', methods=['GET'])
def supported_formats():
    """Támogatott formátumok"""
    return jsonify({
        'success': True,
        'image_formats': list(ALLOWED_EXTENSIONS),
        'max_file_size': '16MB',
        'recommended_resolution': '800x600 - 1200x800',
        'optimization_available': True,
        'supported_features': {
            'cache': True,
            'smart_recognition': True,
            'quality_check': True
        }
    })

@app.errorhandler(413)
def request_entity_too_large(e):
    return jsonify({'error': 'A fájl túl nagy (max 16MB)'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Végpont nem található'}), 404

@app.errorhandler(500)
def internal_error(e):
    global performance_metrics
    performance_metrics['error_count'] += 1
    return jsonify({'error': 'Szerver hiba'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Szerver: http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)