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
from math_recognizer import MathEquationRecognizer

# Flask app inicializálás
app = Flask(__name__)
CORS(app)

# Konfiguráció
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max 16MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

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
        'name': 'Math OCR API',
        'version': '1.0',
        'status': 'online',
        'endpoints': {
            '/recognize': 'POST - Matematikai egyenlet felismerése képről',
            '/health': 'GET - API állapot'
        },
        'usage': {
            'method': 'POST /recognize',
            'body': {'image': 'base64_encoded_image', 'handwritten': True/False}
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Állapot ellenőrzés"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/recognize', methods=['POST'])
def recognize_equation():
    """Matematikai egyenlet felismerése"""
    try:
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
        
        recognizer = MathEquationRecognizer(use_gpu=False, handwritten=handwritten)
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