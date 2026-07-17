# setup.py — run this ONCE to create your project structure
import os

folders = [
    "data/enrollment",
    "data/test/genuine",
    "data/test/impostor",
    "data/spoofing/real",
    "data/spoofing/fake",
    "models",
    "outputs"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"Created: {folder}")

print("\nProject structure ready!")