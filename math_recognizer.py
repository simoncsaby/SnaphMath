import cv2
import numpy as np
import easyocr
import re
from typing import List, Tuple, Optional

# Globális OCR reader
_ocr_reader = None

def get_ocr_reader():
    """Lazy loading az OCR reader-hez"""
    global _ocr_reader
    if _ocr_reader is None:
        print("EasyOCR inicializálása...")
        _ocr_reader = easyocr.Reader(['en'], gpu=False)
    return _ocr_reader

class MathEquationRecognizer:
    """Matematikai egyenletek felismerése képekről"""
    
    def __init__(self, handwritten=False):
        self.reader = get_ocr_reader()
        self.handwritten = handwritten
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Kép előfeldolgozása"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        if self.handwritten:
            # Kézírás előfeldolgozás
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
            denoised = cv2.medianBlur(binary, 3)
            kernel = np.ones((2,2), np.uint8)
            processed = cv2.erode(denoised, kernel, iterations=1)
        else:
            # Nyomtatott szöveg előfeldolgozás
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            denoised = cv2.fastNlMeansDenoising(enhanced)
            _, processed = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return processed
    
    def detect_fractions(self, image: np.ndarray, results: list) -> list:
        """Törtek detektálása"""
        if not results:
            return results
        
        # Vízszintes vonalak keresése (törtvonalak)
        edges = cv2.Canny(image, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=30, maxLineGap=10)
        
        fraction_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(y2 - y1) < 5 and abs(x2 - x1) > 20:
                    fraction_lines.append({
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'y_avg': (y1 + y2) / 2
                    })
        
        # Törtek összeállítása
        processed_indices = set()
        new_results = []
        
        for i, result in enumerate(results):
            if i in processed_indices:
                continue
                
            bbox = result[0]
            text = result[1]
            confidence = result[2]
            
            elem_center_x = (bbox[0][0] + bbox[2][0]) / 2
            elem_center_y = (bbox[0][1] + bbox[2][1]) / 2
            
            fraction_found = False
            for frac_line in fraction_lines:
                if frac_line['x1'] <= elem_center_x <= frac_line['x2']:
                    numerator_elems = []
                    denominator_elems = []
                    
                    for j, other_result in enumerate(results):
                        other_bbox = other_result[0]
                        other_center_x = (other_bbox[0][0] + other_bbox[2][0]) / 2
                        other_center_y = (other_bbox[0][1] + other_bbox[2][1]) / 2
                        
                        if frac_line['x1'] - 10 <= other_center_x <= frac_line['x2'] + 10:
                            if other_center_y < frac_line['y_avg'] - 5:
                                numerator_elems.append((j, other_result))
                            elif other_center_y > frac_line['y_avg'] + 5:
                                denominator_elems.append((j, other_result))
                    
                    if numerator_elems and denominator_elems:
                        num_text = ' '.join([r[1][1] for r in sorted(numerator_elems, key=lambda x: x[1][0][0][0])])
                        den_text = ' '.join([r[1][1] for r in sorted(denominator_elems, key=lambda x: x[1][0][0][0])])
                        
                        all_points = []
                        for _, r in numerator_elems + denominator_elems:
                            all_points.extend([r[0][0], r[0][1], r[0][2], r[0][3]])
                        
                        min_x = min(p[0] for p in all_points)
                        min_y = min(p[1] for p in all_points)
                        max_x = max(p[0] for p in all_points)
                        max_y = max(p[1] for p in all_points)
                        
                        new_bbox = [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]
                        fraction_text = f"({num_text})/({den_text})"
                        
                        new_results.append((new_bbox, fraction_text, confidence))
                        
                        for idx, _ in numerator_elems + denominator_elems:
                            processed_indices.add(idx)
                        
                        fraction_found = True
                        break
            
            if not fraction_found and i not in processed_indices:
                new_results.append(result)
        
        return new_results
    
    def detect_math_structures(self, results: list) -> list:
        """Kitevők és indexek detektálása"""
        if not results:
            return results
        
        # Sorokba rendezés
        sorted_by_y = sorted(results, key=lambda x: x[0][0][1])
        
        lines = []
        current_line = [sorted_by_y[0]]
        current_y = sorted_by_y[0][0][0][1]
        
        for result in sorted_by_y[1:]:
            y = result[0][0][1]
            if abs(y - current_y) > 30:
                lines.append(sorted(current_line, key=lambda x: x[0][0][0]))
                current_line = [result]
                current_y = y
            else:
                current_line.append(result)
        
        if current_line:
            lines.append(sorted(current_line, key=lambda x: x[0][0][0]))
        
        # Kitevők és indexek detektálása
        structured_results = []
        for line in lines:
            i = 0
            while i < len(line):
                current = line[i]
                bbox = current[0]
                text = current[1]
                
                if i + 1 < len(line):
                    next_elem = line[i + 1]
                    next_bbox = next_elem[0]
                    next_text = next_elem[1]
                    
                    current_bottom = bbox[2][1]
                    current_top = bbox[0][1]
                    current_height = current_bottom - current_top
                    current_right = bbox[1][0]
                    
                    next_bottom = next_bbox[2][1]
                    next_top = next_bbox[0][1]
                    next_center_y = (next_top + next_bottom) / 2
                    next_left = next_bbox[0][0]
                    
                    h_distance = next_left - current_right
                    
                    # Kitevő detektálása
                    if (next_center_y < current_top + current_height * 0.3 and 
                        h_distance < 20 and h_distance > -10):
                        combined_text = f"{text}^{next_text}"
                        combined_bbox = [[bbox[0][0], next_bbox[0][1]], 
                                       [next_bbox[1][0], next_bbox[1][1]], 
                                       [next_bbox[2][0], bbox[2][1]], 
                                       [bbox[3][0], bbox[3][1]]]
                        structured_results.append((combined_bbox, combined_text, current[2]))
                        i += 2
                        continue
                    
                    # Index detektálása
                    elif (next_center_y > current_bottom - current_height * 0.3 and 
                          h_distance < 20 and h_distance > -10):
                        combined_text = f"{text}_{next_text}"
                        combined_bbox = [[bbox[0][0], bbox[0][1]], 
                                       [next_bbox[1][0], bbox[1][1]], 
                                       [next_bbox[2][0], next_bbox[2][1]], 
                                       [bbox[3][0], next_bbox[3][1]]]
                        structured_results.append((combined_bbox, combined_text, current[2]))
                        i += 2
                        continue
                
                structured_results.append(current)
                i += 1
        
        return structured_results
    
    def clean_equation(self, text: str) -> str:
        """Szöveg tisztítása"""
        text = ' '.join(text.split())
        
        replacements = {
            '×': '*', '÷': '/', '–': '-', '—': '-',
            'ˆ': '^', '²': '^2', '³': '^3', '⁴': '^4',
            '√': 'sqrt', 'π': 'pi'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # OCR hibák javítása
        text = re.sub(r'(\d)O(\d)', r'\g<1>0\g<2>', text)
        text = re.sub(r'(\d)l(?!x)', r'\g<1>1', text)
        text = re.sub(r'(\d)x', r'\g<1>*x', text)
        text = re.sub(r'(\d)X', r'\g<1>*x', text)
        text = re.sub(r'x(\d+)', r'x^\g<1>', text)
        
        return text.strip()
    
    def convert_to_wolfram(self, equation: str) -> str:
        """Wolfram Alpha formátumra konvertálás"""
        equation = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', equation)
        equation = re.sub(r'([a-zA-Z])(\d)', r'\1*\2', equation)
        equation = re.sub(r'\)(\w)', r')*\1', equation)
        equation = re.sub(r'(\w)\(', r'\1*(', equation)
        
        wolfram_functions = {
            'sqrt': 'Sqrt', 'sin': 'Sin', 'cos': 'Cos', 'tan': 'Tan',
            'log': 'Log', 'ln': 'Log', 'exp': 'Exp'
        }
        
        for old, new in wolfram_functions.items():
            equation = re.sub(rf'\b{old}\b', new, equation, flags=re.IGNORECASE)
        
        return equation
    
    def recognize(self, image: np.ndarray) -> dict:
        """Fő felismerő függvény"""
        try:
            processed_image = self.preprocess_image(image)
            results = self.reader.readtext(image)
            
            results = self.detect_fractions(processed_image, results)
            results = self.detect_math_structures(results)
            
            # Szövegek összefűzése pozíció szerint
            sorted_results = sorted(results, key=lambda x: (x[0][0][1], x[0][0][0]))
            
            texts = []
            last_y = -1
            
            for result in sorted_results:
                bbox, text, confidence = result
                current_y = bbox[0][1]
                
                if last_y != -1 and abs(current_y - last_y) > 20:
                    texts.append(' ')
                
                texts.append(text)
                last_y = current_y
            
            equation = ' '.join(texts)
            equation = self.clean_equation(equation)
            wolfram_format = self.convert_to_wolfram(equation)
            
            return {
                'success': True,
                'equation': equation,
                'wolfram_format': wolfram_format,
                'confidence': 'medium'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }