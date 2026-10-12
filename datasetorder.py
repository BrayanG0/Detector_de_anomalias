import pandas as pd
import shutil
from pathlib import Path

# Rutas configuradas con tu directorio real de FracAtlas
base_dir = Path('/home/karva/Descargas/datasets/FracAtlas')
csv_path = base_dir / 'dataset.csv'
origen_imagenes = base_dir / 'images' 

# Directorio donde quedará tu dataset perfectamente organizado
destino_base = Path('/home/karva/Dataset_Organizado_FracAtlas')

# Leer el archivo CSV
print(f"Leyendo archivo CSV desde: {csv_path}")
df = pd.read_csv(csv_path)

# Regiones anatómicas disponibles en el CSV de FracAtlas
regiones = ['hand', 'leg', 'hip', 'shoulder', 'mixed']

movidas = 0
no_encontradas = 0

for index, row in df.iterrows():
    img_name = row['image_id']
    
    # Determinar la región anatómica principal
    region_encontrada = None
    for reg in regiones:
        if row[reg] == 1:
            region_encontrada = reg.upper()
            break
            
    if not region_encontrada:
        region_encontrada = 'OTHER'
        
    # Determinar estado de fractura (1: fractured, 0: non_fractured)
    estado = 'fractured' if row['fractured'] == 1 else 'non_fractured'
    
    # Crear la ruta de destino: ej. /Dataset_Organizado_FracAtlas/HAND/fractured/
    carpeta_destino = destino_base / region_encontrada / estado
    carpeta_destino.mkdir(parents=True, exist_ok=True)
    
    # Buscar la imagen de forma recursiva dentro de la carpeta 'images'
    archivo_encontrado = None
    for archivo_path in origen_imagenes.rglob(img_name):
        archivo_encontrado = archivo_path
        break
        
    if archivo_encontrado and archivo_encontrado.exists():
        dst_path = carpeta_destino / img_name
        shutil.copy(archivo_encontrado, dst_path)
        movidas += 1
    else:
        no_encontradas += 1

print("\n¡Organización completada con éxito!")
print(f"✔ Imágenes organizadas: {movidas}")
print(f"❌ Imágenes no encontradas: {no_encontradas}")
print(f"📁 Tu nuevo dataset ordenado está en: {destino_base}")
