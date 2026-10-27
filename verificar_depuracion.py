import hashlib
from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image, UnidentifiedImageError

SAMPLE_SIZE = None   
RANDOM_STATE = 42

df = pd.read_csv("mura_manifest.csv")
if SAMPLE_SIZE:
    df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=RANDOM_STATE)

corrupt_files = []
color_modes = []
hashes = {}
duplicates = []
blank_images = []

print(f"Verificando {len(df)} imágenes...")

for _, row in df.iterrows():
    path = row["path"]
    try:
        with Image.open(path) as im:
            im.verify()  
        with Image.open(path) as im:
            color_modes.append(im.mode)

            arr = np.asarray(im.convert("L"))
            h = hashlib.md5(arr.tobytes()).hexdigest()
            if h in hashes:
                duplicates.append((hashes[h], path))
            else:
                hashes[h] = path

            if arr.std() < 1.0:
                blank_images.append(path)

    except (UnidentifiedImageError, OSError) as e:
        corrupt_files.append((path, str(e)))

print("\n" + "=" * 50)
print("RESULTADOS DE DEPURACIÓN")
print("=" * 50)

print(f"\n1. Archivos corruptos/ilegibles: {len(corrupt_files)}")
for p, err in corrupt_files[:10]:
    print(f"   - {p}: {err}")

print(f"\n2. Modos de color encontrados: {pd.Series(color_modes).value_counts().to_dict()}")
print("   (Si hay más de un modo, hay inconsistencia a corregir antes de entrenar)")

print(f"\n3. Duplicados exactos detectados: {len(duplicates)}")
for orig, dup in duplicates[:10]:
    print(f"   - {dup}  es copia de  {orig}")

print(f"\n4. Imágenes casi en blanco/negro (posible error de captura): {len(blank_images)}")
for p in blank_images[:10]:
    print(f"   - {p}")

print("\nListo. Si los 4 conteos son 0 o muy bajos, tu dataset está bien depurado.")
