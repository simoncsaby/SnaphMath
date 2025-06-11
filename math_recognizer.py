import cv2
import numpy as np
import easyocr
import re
from typing import List, Tuple, Optional

# Globális OCR reader (ugyanúgy mint az eredeti kódban)
ocr_reader = None

def get_ocr_reader():
    """Lazy loading az OCR reader-hez"""
    global ocr_reader
    if ocr_reader is None:
        print("EasyOCR inicializálása...")
        ocr_reader = easyocr.Reader(['en'], gpu=False)
    return ocr_reader

class MathEquationRecognizer:
    """Matematikai egyenletek felismerése képekről"""
    
    def __init__(self, use_gpu=False, handwritten=True):
        """
        Inicializálás EasyOCR-rel - ugyanúgy mint az eredeti
        
        Args:
            use_gpu: GPU használata (ha elérhető)
            handwritten: Kézírás mód (True esetén speciális beállítások)
        """
        # EasyOCR inicializálás (angol nyelv)
        self.reader = get_ocr_reader()
        self.handwritten = handwritten
        
        # Mathpix API credentials (opcionális)
        self.mathpix_app_id = None
        self.mathpix_app_key = None
    
    def setup_mathpix(self, app_id: str, app_key: str):
        """
        Mathpix API beállítása (opcionális)
        
        Args:
            app_id: Mathpix app ID
            app_key: Mathpix app key
        """
        self.mathpix_app_id = app_id
        self.mathpix_app_key = app_key
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Kép előfeldolgozása a jobb felismeréshez
        
        Args:
            image: Kép numpy array formában
            
        Returns:
            Előfeldolgozott kép numpy array formában
        """
        # Szürkeárnyalatosra konvertálás ha szükséges
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Kézírás esetén speciális előfeldolgozás
        if self.handwritten:
            # Adaptív küszöbölés kézíráshoz
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
            
            # Zajszűrés mediánszűrővel
            denoised = cv2.medianBlur(binary, 3)
            
            # Vékonyítás a jobb karakterfelismeréshez
            kernel = np.ones((2,2), np.uint8)
            eroded = cv2.erode(denoised, kernel, iterations=1)
            
            return eroded
        else:
            # Nyomtatott szöveg előfeldolgozása
            # Kontraszt növelése CLAHE-val
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Zajszűrés
            denoised = cv2.fastNlMeansDenoising(enhanced)
            
            # Binarizálás Otsu módszerrel
            _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            return binary
    
    def calculate_confidence(self, results: list, processed_results: list) -> float:
        """
        Confidence számítása az OCR eredmények alapján
        
        Args:
            results: Eredeti OCR eredmények
            processed_results: Feldolgozott eredmények
            
        Returns:
            Confidence érték 0-1 között
        """
        if not results:
            return 0.0
        
        # OCR confidence értékek átlaga
        ocr_confidences = [result[2] for result in results if len(result) > 2]
        avg_ocr_confidence = sum(ocr_confidences) / len(ocr_confidences) if ocr_confidences else 0.0
        
        # Feldolgozási minőség értékelése
        processing_quality = 1.0
        
        # Csökkentjük a confidence-t, ha sok elem lett összevonva
        original_count = len(results)
        processed_count = len(processed_results)
        if original_count > 0:
            processing_ratio = processed_count / original_count
            # Ha túl sok elemet vontunk össze, csökkentjük a confidence-t
            if processing_ratio < 0.5:
                processing_quality *= 0.8
        
        # Matematikai jellegű elemek detektálása
        math_elements = 0
        total_elements = len(processed_results)
        
        for result in processed_results:
            text = result[1] if len(result) > 1 else ""
            # Matematikai jelek keresése
            if any(char in text for char in "=+-*/^()[]{}√∫∑∏αβγπ"):
                math_elements += 1
            # Számok keresése
            if re.search(r'\d', text):
                math_elements += 1
        
        # Matematikai tartalom arány
        math_ratio = math_elements / max(total_elements, 1)
        
        # Végső confidence számítása (súlyozott átlag)
        final_confidence = (
            avg_ocr_confidence * 0.5 +           # OCR confidence 50%
            processing_quality * 0.3 +           # Feldolgozási minőség 30%
            math_ratio * 0.2                     # Matematikai tartalom 20%
        )
        
        return min(max(final_confidence, 0.0), 1.0)  # 0-1 közé korlátozás
    
    def detect_fraction_regions(self, image: np.ndarray, results: list) -> Tuple[list, float]:
        """
        Törtek detektálása a felismert elemek pozíciója alapján
        
        Args:
            image: Előfeldolgozott kép
            results: EasyOCR eredmények
            
        Returns:
            Tuple: (Frissített eredmények törtekkel, feldolgozási confidence)
        """
        if not results:
            return results, 1.0
        
        processing_confidence = 1.0
        
        # Vízszintes vonalak detektálása (törtvonalak)
        edges = cv2.Canny(image, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=30, maxLineGap=10)
        
        fraction_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Csak a közel vízszintes vonalak (törtvonalak)
                if abs(y2 - y1) < 5 and abs(x2 - x1) > 20:
                    fraction_lines.append({
                        'x1': x1, 'y1': y1, 
                        'x2': x2, 'y2': y2,
                        'y_avg': (y1 + y2) / 2
                    })
        
        # Elemek csoportosítása törtek szerint
        processed_indices = set()
        new_results = []
        fractions_found = 0
        
        for i, result in enumerate(results):
            if i in processed_indices:
                continue
                
            bbox = result[0]
            text = result[1]
            confidence = result[2]
            
            # Elem középpontja
            elem_center_x = (bbox[0][0] + bbox[2][0]) / 2
            elem_center_y = (bbox[0][1] + bbox[2][1]) / 2
            
            # Keressük meg, hogy van-e törtvonal alatta/felette
            fraction_found = False
            for frac_line in fraction_lines:
                # Ha az elem x koordinátája a vonal tartományában van
                if frac_line['x1'] <= elem_center_x <= frac_line['x2']:
                    # Keressük meg a számláló és nevező elemeket
                    numerator_elems = []
                    denominator_elems = []
                    
                    for j, other_result in enumerate(results):
                        other_bbox = other_result[0]
                        other_center_x = (other_bbox[0][0] + other_bbox[2][0]) / 2
                        other_center_y = (other_bbox[0][1] + other_bbox[2][1]) / 2
                        
                        # Ha az elem a törtvonal tartományában van
                        if frac_line['x1'] - 10 <= other_center_x <= frac_line['x2'] + 10:
                            # Számláló (felette van)
                            if other_center_y < frac_line['y_avg'] - 5:
                                numerator_elems.append((j, other_result))
                            # Nevező (alatta van)
                            elif other_center_y > frac_line['y_avg'] + 5:
                                denominator_elems.append((j, other_result))
                    
                    if numerator_elems and denominator_elems:
                        # Tört összeállítása
                        num_text = ' '.join([r[1][1] for r in sorted(numerator_elems, key=lambda x: x[1][0][0][0])])
                        den_text = ' '.join([r[1][1] for r in sorted(denominator_elems, key=lambda x: x[1][0][0][0])])
                        
                        # Új bounding box a teljes törtnek
                        all_points = []
                        for _, r in numerator_elems + denominator_elems:
                            all_points.extend([r[0][0], r[0][1], r[0][2], r[0][3]])
                        
                        min_x = min(p[0] for p in all_points)
                        min_y = min(p[1] for p in all_points)
                        max_x = max(p[0] for p in all_points)
                        max_y = max(p[1] for p in all_points)
                        
                        new_bbox = [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]
                        
                        # Tört szöveg formátum
                        fraction_text = f"({num_text})/({den_text})"
                        
                        # Átlagos confidence a tört elemeiből
                        avg_confidence = sum([r[1][2] for r in numerator_elems + denominator_elems]) / len(numerator_elems + denominator_elems)
                        
                        new_results.append((new_bbox, fraction_text, avg_confidence))
                        
                        # Jelöljük meg a feldolgozott elemeket
                        for idx, _ in numerator_elems + denominator_elems:
                            processed_indices.add(idx)
                        
                        fraction_found = True
                        fractions_found += 1
                        break
            
            if not fraction_found and i not in processed_indices:
                new_results.append(result)
        
        # Ha túl sok tört detektáltunk, csökkentjük a confidence-t
        if fractions_found > len(results) * 0.3:  # Ha több mint 30% tört
            processing_confidence *= 0.9
        
        return new_results, processing_confidence
    
    def detect_mathematical_structures(self, results: list) -> Tuple[list, float]:
        """
        Matematikai struktúrák felismerése pozíció alapján
        
        Args:
            results: OCR eredmények
            
        Returns:
            Tuple: (Strukturált eredmények, feldolgozási confidence)
        """
        if not results:
            return results, 1.0
        
        processing_confidence = 1.0
        
        # Rendezés y koordináta szerint (sorok)
        sorted_by_y = sorted(results, key=lambda x: x[0][0][1])
        
        # Sorokba csoportosítás
        lines = []
        current_line = [sorted_by_y[0]]
        current_y = sorted_by_y[0][0][0][1]
        
        for result in sorted_by_y[1:]:
            y = result[0][0][1]
            # Ha új sor (y koordináta jelentősen eltér)
            if abs(y - current_y) > 30:
                lines.append(sorted(current_line, key=lambda x: x[0][0][0]))  # x szerint rendezve
                current_line = [result]
                current_y = y
            else:
                current_line.append(result)
        
        if current_line:
            lines.append(sorted(current_line, key=lambda x: x[0][0][0]))
        
        # Kitevők és indexek detektálása
        structured_results = []
        structures_found = 0
        
        for line in lines:
            i = 0
            while i < len(line):
                current = line[i]
                bbox = current[0]
                text = current[1]
                
                # Következő elem vizsgálata (kitevő vagy index lehet)
                if i + 1 < len(line):
                    next_elem = line[i + 1]
                    next_bbox = next_elem[0]
                    next_text = next_elem[1]
                    
                    # Vertikális pozíció különbség
                    current_bottom = bbox[2][1]
                    current_top = bbox[0][1]
                    current_height = current_bottom - current_top
                    
                    next_bottom = next_bbox[2][1]
                    next_top = next_bbox[0][1]
                    next_center_y = (next_top + next_bottom) / 2
                    
                    # Horizontális távolság
                    current_right = bbox[1][0]
                    next_left = next_bbox[0][0]
                    h_distance = next_left - current_right
                    
                    # Kitevő detektálása (fent van és közel)
                    if (next_center_y < current_top + current_height * 0.3 and 
                        h_distance < 20 and h_distance > -10):
                        # Ez egy kitevő
                        combined_text = f"{text}^{next_text}"
                        combined_bbox = [[bbox[0][0], next_bbox[0][1]], 
                                       [next_bbox[1][0], next_bbox[1][1]], 
                                       [next_bbox[2][0], bbox[2][1]], 
                                       [bbox[3][0], bbox[3][1]]]
                        # Átlagos confidence
                        avg_confidence = (current[2] + next_elem[2]) / 2
                        structured_results.append((combined_bbox, combined_text, avg_confidence))
                        structures_found += 1
                        i += 2  # Mindkét elemet feldolgoztuk
                        continue
                    
                    # Index detektálása (lent van és közel)
                    elif (next_center_y > current_bottom - current_height * 0.3 and 
                          h_distance < 20 and h_distance > -10):
                        # Ez egy index
                        combined_text = f"{text}_{next_text}"
                        combined_bbox = [[bbox[0][0], bbox[0][1]], 
                                       [next_bbox[1][0], bbox[1][1]], 
                                       [next_bbox[2][0], next_bbox[2][1]], 
                                       [bbox[3][0], next_bbox[3][1]]]
                        # Átlagos confidence
                        avg_confidence = (current[2] + next_elem[2]) / 2
                        structured_results.append((combined_bbox, combined_text, avg_confidence))
                        structures_found += 1
                        i += 2  # Mindkét elemet feldolgoztuk
                        continue
                
                # Normál elem
                structured_results.append(current)
                i += 1
        
        # Ha túl sok struktúrát detektáltunk, lehet hogy tévedés
        if structures_found > len(results) * 0.4:  # Ha több mint 40% struktúra
            processing_confidence *= 0.85
        
        return structured_results, processing_confidence
    
    def recognize_from_image(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Szöveg felismerés EasyOCR-rel numpy array-ből
        
        Args:
            image: Kép numpy array formában
            
        Returns:
            Tuple: (Felismert egyenlet szöveg formában, confidence érték)
        """
        # Előfeldolgozott kép készítése a struktúra detektáláshoz
        processed_image = self.preprocess_image(image)
        
        # OCR futtatása
        results = self.reader.readtext(image)
        
        if not results:
            return "", 0.0
        
        # Törtek detektálása
        results, fraction_confidence = self.detect_fraction_regions(processed_image, results)
        
        # Matematikai struktúrák (kitevők, indexek) detektálása
        results, structure_confidence = self.detect_mathematical_structures(results)
        
        # Szövegek összefűzése pozíció alapján rendezve
        sorted_results = sorted(results, key=lambda x: (x[0][0][1], x[0][0][0]))
        
        # Szövegek összefűzése
        texts = []
        last_y = -1
        line_threshold = 20  # Sorok közötti küszöb
        
        for result in sorted_results:
            bbox, text, confidence = result
            current_y = bbox[0][1]
            
            # Ha új sor, akkor space hozzáadása
            if last_y != -1 and abs(current_y - last_y) > line_threshold:
                texts.append(' ')
            
            texts.append(text)
            last_y = current_y
        
        equation = ' '.join(texts)
        
        # Tisztítás és formázás
        equation = self.clean_equation(equation)
        
        # Confidence számítása
        original_ocr_results = self.reader.readtext(image)
        final_confidence = self.calculate_confidence(original_ocr_results, results)
        
        # Feldolgozási confidence-k figyelembevétele
        final_confidence *= fraction_confidence * structure_confidence
        
        return equation, final_confidence
    
    def post_process_equation(self, equation: str) -> str:
        """
        Egyenlet utófeldolgozása kontextus alapján
        
        Args:
            equation: Tisztított egyenlet
            
        Returns:
            Véglegesen formázott egyenlet
        """
        # Algebra egyenlet detektálása
        # Ha van = jel és a bal oldalon számok és műveletek vannak
        if '=' in equation:
            left_side, right_side = equation.split('=', 1)
            
            # Tipikus algebra minták felismerése
            # Minta: szám + operátor + szám = szám (pl. 21-4=12)
            # Ez valószínűleg 2x-4=12
            import re
            
            # Ha a bal oldalon van egy kétjegyű szám és egy művelet
            pattern = r'^(\d{2})\s*([+\-*/])\s*(\d+)$'
            match = re.match(pattern, left_side.strip())
            
            if match:
                two_digit = match.group(1)
                operator = match.group(2)
                second_num = match.group(3)
                
                # Ha a kétjegyű szám első számjegye kicsi (1-9) és nincs más változó
                # akkor valószínűleg ez egy szám*x minta
                first_digit = two_digit[0]
                second_digit = two_digit[1]
                
                # Tipikus algebra egyenlet minták
                if first_digit in '123456789' and second_digit in '1234567890':
                    # Valószínűleg szám*változó volt
                    # 21 -> 2x, 31 -> 3x, stb.
                    if second_digit == '1':
                        left_side = f"{first_digit}*x {operator} {second_num}"
                    else:
                        # Lehet hogy tényleg kétjegyű szám
                        pass
            
            equation = f"{left_side}={right_side}"
        
        return equation
    
    def clean_equation(self, text: str) -> str:
        """
        Felismert szöveg tisztítása és matematikai formázása
        
        Args:
            text: Nyers felismert szöveg
            
        Returns:
            Tisztított egyenlet
        """
        # Felesleges whitespace eltávolítása
        text = ' '.join(text.split())
        
        # Gyakori OCR hibák javítása
        replacements = {
            '×': '*',
            '÷': '/',
            '–': '-',
            '—': '-',
            # ''': "'",  # okos aposztróf
            '"': '"',  # okos idézőjel nyitó
            '"': '"',  # okos idézőjel záró
            'ˆ': '^',
            '²': '^2',
            '³': '^3',
            '⁴': '^4',
            '√': 'sqrt',
            '∫': 'integral',
            '∑': 'sum',
            '∏': 'product',
            'α': 'alpha',
            'β': 'beta',
            'γ': 'gamma',
            'π': 'pi',
            '∞': 'infinity',
            # További gyakori hibák
            'Il': '||',
            'Z': '2',  # ha számok között van
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Speciális esetek kezelése regex-szel
        # O -> 0 számok között
        text = re.sub(r'(\d)O(\d)', r'\g<1>0\g<2>', text)
        # l -> 1 számok között vagy számok után (de nem változó x előtt)
        text = re.sub(r'(\d)l(?!x)', r'\g<1>1', text)
        
        # X/x felismerés javítása
        # Ha egy szám után közvetlenül 'x' karakter jön, akkor ez valószínűleg szorzás
        # Kivéve ha már van szóköz vagy operátor
        text = re.sub(r'(\d)x', r'\g<1>*x', text)
        text = re.sub(r'(\d)X', r'\g<1>*x', text)
        
        # x2, x3, stb. hatványozássá alakítása
        text = re.sub(r'x(\d+)', r'x^\g<1>', text)
        text = re.sub(r'X(\d+)', r'x^\g<1>', text)
        
        # Zárójelek egyensúlyozása
        text = self.balance_parentheses(text)
        
        # Utófeldolgozás kontextus alapján
        text = self.post_process_equation(text)
        
        return text.strip()
    
    def balance_parentheses(self, equation: str) -> str:
        """
        Zárójelek egyensúlyozása az egyenletben
        
        Args:
            equation: Egyenlet szöveg
            
        Returns:
            Kiegyensúlyozott zárójelekkel rendelkező egyenlet
        """
        open_count = equation.count('(')
        close_count = equation.count(')')
        
        if open_count > close_count:
            equation += ')' * (open_count - close_count)
        elif close_count > open_count:
            equation = '(' * (close_count - open_count) + equation
        
        return equation
    
    def convert_to_wolfram_format(self, equation: str) -> str:
        """
        Egyenlet konvertálása Wolfram Alpha formátumra
        
        Args:
            equation: Tisztított egyenlet
            
        Returns:
            Wolfram Alpha kompatibilis formátum
        """
        # Hatványozás formázása
        equation = re.sub(r'(\w)\^(\d+)', r'\1^\2', equation)
        
        # Implicit szorzás explicit szorzássá alakítása
        equation = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', equation)
        equation = re.sub(r'([a-zA-Z])(\d)', r'\1*\2', equation)
        equation = re.sub(r'\)(\w)', r')*\1', equation)
        equation = re.sub(r'(\w)\(', r'\1*(', equation)
        
        # Speciális függvények
        wolfram_functions = {
            'sqrt': 'Sqrt',
            'sin': 'Sin',
            'cos': 'Cos',
            'tan': 'Tan',
            'log': 'Log',
            'ln': 'Log',
            'exp': 'Exp',
            'arcsin': 'ArcSin',
            'arccos': 'ArcCos',
            'arctan': 'ArcTan',
            'abs': 'Abs',
            'integral': 'Integrate',
            'sum': 'Sum',
            'product': 'Product',
            'limit': 'Limit',
            'derivative': 'D'
        }
        
        for old, new in wolfram_functions.items():
            equation = re.sub(rf'\b{old}\b', new, equation, flags=re.IGNORECASE)
        
        return equation
    
    def recognize(self, image: np.ndarray) -> dict:
        """
        Fő felismerő függvény
        
        Args:
            image: Kép numpy array formában
            
        Returns:
            Dictionary a felismert egyenlettel és Wolfram formátummal
        """
        try:
            equation, confidence = self.recognize_from_image(image)
            
            wolfram_format = self.convert_to_wolfram_format(equation)
            
            return {
                'success': True,
                'equation': equation,
                'wolfram_format': wolfram_format,
                'confidence': round(confidence, 3),
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'confidence': 0,
            }