from pathlib import Path
from PIL import Image, ImageOps, ImageFile  # <-- Importar ImageFile
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Habilitar la carga de imágenes truncadas para evitar el error
ImageFile.LOAD_TRUNCATED_IMAGES = True

def procesar_y_sobrescribir(ruta_imagen, tamano_objetivo=(224, 224), color_fondo=(0, 0, 0)):
    """Abre la imagen, la ajusta y la sobrescribe en su misma ubicación."""
    try:
        # Abrimos la imagen original
        img = Image.open(ruta_imagen)
        # Aunque sean JPG, aseguramos que sean RGB (3 canales)
        img_rgb = img.convert("RGB")
        # Ajustamos el tamaño y rellenamos los bordes (padding)
        img_procesada = ImageOps.pad(img_rgb, tamano_objetivo, color=color_fondo)
        # Cerramos el archivo original para liberar el archivo y poder sobrescribirlo
        img.close()
        # Sobrescribimos el archivo original con la nueva imagen procesada
        img_procesada.save(ruta_imagen, quality=95)

    except Exception as e:
        return f"Error en {ruta_imagen.name}: {e}"
    return None

def modificar_dataset_en_sitio(carpeta_principal, tamano_objetivo=(224, 224)):
    """Busca todas las imágenes en las subcarpetas y las procesa sobrescribiéndolas."""
    ruta_principal = Path(carpeta_principal)
    extensiones = ('.jpg', '.jpeg', '.JPG', '.JPEG')
    archivos = []
    for ext in extensiones:
        archivos.extend(ruta_principal.rglob(f'*{ext}'))
        
    print(f"Se encontraron {len(archivos)} imágenes para modificar. Iniciando proceso...")
    
    with ThreadPoolExecutor() as executor:
        tareas = [(ruta, tamano_objetivo) for ruta in archivos]
        resultados = list(tqdm(
            executor.map(lambda args: procesar_y_sobrescribir(*args), tareas),
            total=len(tareas),
            desc="Sobrescribiendo imágenes"
        ))

    errores = [r for r in resultados if r is not None]
    if errores:
        print(f"\nProceso finalizado con {len(errores)} errores:")
        for e in errores:
            print(e)
    else:
        print("\n¡Proceso completado! Todas las imágenes fueron redimensionadas y rellenadas a 224x224.")

# ==========================================
# EJECUCIÓN
# ==========================================
carpeta_dataset = '/home/karva/JPGCONVERSIONDATASETCOPIA/'
modificar_dataset_en_sitio(carpeta_dataset, tamano_objetivo=(224, 224))