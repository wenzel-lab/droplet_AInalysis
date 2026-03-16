# Run this file to install all necessary libraries

import sys
from subprocess import check_call, CalledProcessError
from sys import executable
import os
import pkg_resources
from typing import List

# ========================
# Droplet AInalysis Installer
# ========================
# This script installs all required Python packages for both webcam-v1 and static-image-v1 modules.
# Usage: python packages.py
# It will also generate a requirements.txt for use with pip or virtual environments.
# ========================

REQUIRED_PACKAGES = [
    "ultralytics",
    "tabulate",
    "imageio",
    "numpy",
    "opencv-python",
    "matplotlib",
    "seaborn",
    "torch",
    "scipy",
    "pandas"
]


REQUIREMENTS_TXT = "requirements.txt"

def write_requirements_txt(packages: List[str], filename: str = REQUIREMENTS_TXT):
    """Write all required packages to requirements.txt."""
    with open(filename, "w") as f:
        for pkg in packages:
            f.write(pkg + "\n")
    print(f"requirements.txt generated with all dependencies.")

def is_installed(pkg_name):
    """Check if a package is already installed."""
    try:
        pkg_resources.require(pkg_name)
        return True
    except pkg_resources.DistributionNotFound:
        return False
    except pkg_resources.VersionConflict:
        return True  # Assume installed, even if version mismatch

def pip_available():
    """Check if pip is available in the current Python environment."""
    try:
        check_call([executable, "-m", "pip", "--version"], stdout=open(os.devnull, 'w'), stderr=open(os.devnull, 'w'))
        return True
    except Exception:
        return False

def install_packages(packages: List[str]):
    """Install all required packages, skipping those already installed."""
    for package in packages:
        if is_installed(package):
            print(f"[✓] {package} is already installed.")
        else:
            print(f"[+] Installing {package}...")
            try:
                check_call([executable, "-m", "pip", "install", package])
                print(f"[✓] {package} installed.")
            except CalledProcessError as e:
                print(f"[!] Failed to install {package}: {e}")
            except Exception as e:
                print(f"[!] Unexpected error installing {package}: {e}")

if __name__ == "__main__":
    print("\n=== Droplet AInalysis Dependency Installer ===\n")
    if not pip_available():
        print("[!] pip is not available in this Python environment. Please install pip and try again.")
        sys.exit(1)
    write_requirements_txt(REQUIRED_PACKAGES)
    install_packages(REQUIRED_PACKAGES)
    print("\nAll done! If you encounter issues, try: pip install -r requirements.txt\nRun this script as administrator if you see permissions errors.")