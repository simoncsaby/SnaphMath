#!/usr/bin/env python3
"""
Matematikai egyenletfelismerő projekt telepítőscript.
Ez a script létrehozza a könyvtárszerkezetet és előkészíti a környezetet.
"""

import os
import sys
import subprocess
import platform
from utils import create_directory_structure, check_gpu_availability


def check_python_version():
    """Python verzió ellenőrzése"""
    if sys.version_info < (3, 6):
        print("Hiba: Python 3.6 vagy újabb verzió szükséges!")
        sys.exit(1)
    print(f"Python verzió: {platform.python_version()} ✓")


def install_dependencies():
    """Függőségek telepítése a requirements.txt fájlból"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Függőségek telepítve ✓")
    except subprocess.CalledProcessError as e:
        print(f"Hiba a függőségek telepítése során: {str(e)}")
        sys.exit(1)


def setup_project():
    """Projekt beállítása"""
    print("=== Matematikai egyenletfelismerő projekt telepítése ===")
    
    # Python verzió ellenőrzése
    check_python_version()
    
    # Könyvtárstruktúra létrehozása
    create_directory_structure()
    
    # GPU elérhetőség ellenőrzése
    check_gpu_availability()
    
    # Függőségek telepítése
    install_dependencies()
    
    # Adathalmazok letöltése (opcionális, később is futtatható)
    download_datasets = input("Szeretné most letölteni az adathalmazokat? (y/n): ").lower()
    if download_datasets == 'y':
        try:
            subprocess.check_call([sys.executable, "download_datasets.py"])
            print("Adathalmazok letöltve ✓")
        except subprocess.CalledProcessError as e:
            print(f"Hiba az adathalmazok letöltése során: {str(e)}")
    
    print("\n=== Telepítés befejezve! ===")
    print("\nA projekt használatához futtassa a következő parancsokat:")
    print("1. Modell tanítása:")
    print("   python main.py --mode train")
    print("2. Egyenlet felismerése:")
    print("   python main.py --mode test --image test_images/egyenlet.png")


if __name__ == "__main__":
    setup_project()
