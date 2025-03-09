import os
import sys
import requests
import zipfile
import tarfile
import shutil
from tqdm import tqdm
from config import CROHME_DATA_PATH, IM2LATEX_DATA_PATH


def download_file(url, destination):
    """
    Fájl letöltése progressbar-ral
    """
    if os.path.exists(destination):
        print(f"A fájl már létezik: {destination}")
        return True
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 KB
        
        with open(destination, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(destination)) as pbar:
                for data in response.iter_content(block_size):
                    f.write(data)
                    pbar.update(len(data))
                    
        print(f"Letöltés sikeres: {destination}")
    except Exception as e:
        if os.path.exists(destination):
            os.remove(destination)
        print(f"Hiba történt a letöltés során: {str(e)}")
        return False
    
    return True


def extract_zip(zip_path, extract_to):
    """
    Zip fájl kicsomagolása
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Összes fájl listázása
            file_list = zip_ref.namelist()
            total_files = len(file_list)
            
            # Kicsomagolás progressbar-ral
            for i, file in enumerate(file_list):
                zip_ref.extract(file, extract_to)
                sys.stdout.write(f"\rKicsomagolás: {i+1}/{total_files} fájl")
                sys.stdout.flush()
            
            print("\nKicsomagolás befejezve!")
    except Exception as e:
        print(f"Hiba történt a zip fájl kicsomagolása során: {str(e)}")
        return False
    
    return True


def extract_tar(tar_path, extract_to):
    """
    Tar fájl kicsomagolása
    """
    try:
        with tarfile.open(tar_path) as tar_ref:
            # Összes fájl listázása
            file_list = tar_ref.getnames()
            total_files = len(file_list)
            
            # Kicsomagolás progressbar-ral
            for i, file in enumerate(file_list):
                tar_ref.extract(file, extract_to)
                sys.stdout.write(f"\rKicsomagolás: {i+1}/{total_files} fájl")
                sys.stdout.flush()
            
            print("\nKicsomagolás befejezve!")
    except Exception as e:
        print(f"Hiba történt a tar fájl kicsomagolása során: {str(e)}")
        return False
    
    return True


def download_crohme_dataset():
    """
    CROHME adathalmaz letöltése az új URL-ről
    """
    print("=== CROHME adathalmaz letöltése ===")
    
    # Ellenőrizzük, hogy már letöltöttük-e
    if os.path.exists(os.path.join(CROHME_DATA_PATH, "CROHME_full_v2")):
        print("A CROHME adathalmaz már letöltve.")
        return True
    
    # CROHME új letöltési URL
    url = "http://www.iapr-tc11.org/dataset/CROHME/CROHME_full_v2.zip"
    download_path = os.path.join(CROHME_DATA_PATH, "CROHME_full_v2.zip")
    
    # Könyvtár létrehozása
    os.makedirs(CROHME_DATA_PATH, exist_ok=True)
    
    # Letöltés
    if not download_file(url, download_path):
        return False
    
    # Kicsomagolás
    print("CROHME adathalmaz kicsomagolása...")
    if not extract_zip(download_path, CROHME_DATA_PATH):
        return False
    
    # A zip fájlok kibontása a CROHME adathalmazon belül is
    crohme_dir = os.path.join(CROHME_DATA_PATH, "CROHME_full_v2")
    if os.path.exists(crohme_dir):
        for root, dirs, files in os.walk(crohme_dir):
            for file in files:
                if file.endswith('.zip'):
                    zip_file = os.path.join(root, file)
                    print(f"További ZIP fájl kicsomagolása: {zip_file}")
                    extract_zip(zip_file, os.path.dirname(zip_file))
    
    # Formula címkék létrehozása
    print("Formula címkék létrehozása...")
    create_formula_labels()
    
    # ZIP fájl törlése a hely felszabadításához (opcionális)
    # os.remove(download_path)
    
    print("CROHME adathalmaz sikeresen letöltve és kicsomagolva.")
    return True


def create_formula_labels():
    """
    Formula címkék létrehozása a CROHME adathalmazból
    """
    try:
        # Tanító adatok könyvtára
        train_dir = os.path.join(CROHME_DATA_PATH, "CROHME_full_v2", "CROHME2013_data", "TrainINKML")
        
        if not os.path.exists(train_dir):
            print(f"A tanító adatok könyvtára nem létezik: {train_dir}")
            return False
        
        # InkML fájlok keresése
        inkml_files = []
        for root, dirs, files in os.walk(train_dir):
            for file in files:
                if file.endswith('.inkml'):
                    inkml_files.append(os.path.join(root, file))
        
        print(f"{len(inkml_files)} InkML fájl található.")
        
        # Formula címkék fájl létrehozása
        formula_labels_file = os.path.join(CROHME_DATA_PATH, "formula_labels.txt")
        
        # Egyszerű formula labels fájl létrehozása (fájlnév,LaTeX formátum)
        with open(formula_labels_file, 'w', encoding='utf-8') as f:
            for inkml_file in inkml_files:
                # Az egyszerűség kedvéért csak a fájl nevét használjuk, és később 
                # egy dummy latex formátumot (ez csak példa)
                file_name = os.path.basename(inkml_file)
                # Később innen kinyerhetjük a valódi LaTeX kódot az inkml fájlból
                f.write(f"{inkml_file},x^2 + y^2 = r^2\n")
        
        print(f"Formula címkék létrehozva: {formula_labels_file}")
        return True
        
    except Exception as e:
        print(f"Hiba a formula címkék létrehozása során: {str(e)}")
        return False


def download_im2latex_dataset():
    """
    im2latex-100k adathalmaz letöltése
    """
    print("=== im2latex-100k adathalmaz letöltése ===")
    
    # Ellenőrizzük, hogy már letöltöttük-e
    if os.path.exists(os.path.join(IM2LATEX_DATA_PATH, "im2latex_formulas.lst")):
        print("Az im2latex adathalmaz már letöltve.")
        return True
    
    # Könyvtár létrehozása
    os.makedirs(IM2LATEX_DATA_PATH, exist_ok=True)
    os.makedirs(os.path.join(IM2LATEX_DATA_PATH, "images"), exist_ok=True)
    
    # Fájlok letöltése
    files_to_download = [
        ("https://zenodo.org/record/56198/files/im2latex_formulas.lst", "im2latex_formulas.lst"),
        ("https://zenodo.org/record/56198/files/im2latex_train.lst", "im2latex_train.lst"),
        ("https://zenodo.org/record/56198/files/im2latex_validate.lst", "im2latex_validate.lst"),
        ("https://zenodo.org/record/56198/files/im2latex_test.lst", "im2latex_test.lst"),
        ("https://zenodo.org/record/56198/files/formula_images.tar.gz", "formula_images.tar.gz")
    ]
    
    for url, filename in files_to_download:
        download_path = os.path.join(IM2LATEX_DATA_PATH, filename)
        if not download_file(url, download_path):
            return False
    
    # Képek kicsomagolása
    images_tar = os.path.join(IM2LATEX_DATA_PATH, "formula_images.tar.gz")
    print("im2latex képek kicsomagolása...")
    if not extract_tar(images_tar, IM2LATEX_DATA_PATH):
        return False
    
    # Fájlok áthelyezése a megfelelő könyvtárba
    src_dir = os.path.join(IM2LATEX_DATA_PATH, "formula_images")
    if os.path.exists(src_dir):
        dest_dir = os.path.join(IM2LATEX_DATA_PATH, "images")
        for item in os.listdir(src_dir):
            s = os.path.join(src_dir, item)
            d = os.path.join(dest_dir, item)
            shutil.move(s, d)
        os.rmdir(src_dir)
    
    # TAR fájl törlése a hely felszabadításához
    os.remove(images_tar)
    
    print("im2latex-100k adathalmaz sikeresen letöltve és kicsomagolva.")
    return True


def main():
    """Adathalmazok letöltése"""
    print("Matematikai egyenletfelismerő adathalmazok letöltése...")
    
    # CROHME adathalmaz
    download_crohme_dataset()
    
    # im2latex adathalmaz
    download_im2latex_dataset()
    
    print("Adathalmazok letöltése befejezve.")
    print(f"CROHME adathalmaz: {CROHME_DATA_PATH}")
    print(f"im2latex adathalmaz: {IM2LATEX_DATA_PATH}")


if __name__ == "__main__":
    main()