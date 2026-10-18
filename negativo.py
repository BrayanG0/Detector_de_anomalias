from pathlib import Path
from PIL import Image, ImageOps, ImageFile
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Habilitar carga de imágenes truncadas por seguridad
ImageFile.LOAD_TRUNCATED_IMAGES = True

CARPETA_DATASET = Path('/home/karva/Copiamura_dataset_completo')
GROSOR_BORDE = 10  # píxeles de los bordes para confirmar el fondo

def estandarizar_polaridad(ruta_imagen):
    """
    Inverte la imagen si tiene más blanco que negro (fondo blanco / negativo),
    dejándola en el estándar de fondo negro y hueso blanco.
    """
    try:
        with Image.open(ruta_imagen) as img:
            img_rgb = img.convert("RGB")
            gris = np.array(img_rgb.convert("L"))

        # 1. Analizar los bordes para confirmar que el fondo es claro
        b = GROSOR_BORDE
        borde = np.concatenate([
            gris[:b, :].flatten(),
            gris[-b:, :].flatten(),
            gris[:, :b].flatten(),
            gris[:, -b:].flatten(),
        ])
        brillo_borde = borde.mean()

        # 2. Tu lógica: Ver si la imagen en general tiene más blanco que negro
        # (Un brillo promedio mayor a 127 indica que predomina el blanco sobre el negro)
        brillo_promedio = gris.mean()
        tiene_mas_blanco = brillo_promedio > 127

        # Está invertida si el borde es claro Y la imagen en general es más blanca que negra
        es_negativo = (brillo_borde > 180) and tiene_mas_blanco

        if es_negativo:
            with Image.open(ruta_imagen) as img_original:
                img_trabajo = img_original.convert("RGB")
                # Invertir colores: el fondo blanco pasa a negro y el hueso oscuro a blanco
                img_invertida = ImageOps.invert(img_trabajo)
                img_invertida.save(ruta_imagen, quality=95)

        return None
    except Exception as e:
        return f"Error en {ruta_imagen.name}: {e}"

def main():
    extensiones = ('*.jpg', '*.jpeg', '*.JPG', '*.JPEG')
    archivos = []
    for ext in extensiones:
        archivos.extend(CARPETA_DATASET.rglob(ext))

    print(f"Analizando {len(archivos)} imágenes para estandarizar a fondo negro...")

    with ThreadPoolExecutor() as executor:
        resultados = list(tqdm(
            executor.map(estandarizar_polaridad, archivos),
            total=len(archivos),
            desc="Estandarizando polaridad"
        ))

    errores = [r for r in resultados if r is not None]
    if errores:
        print(f"\nProceso finalizado con {len(errores)} errores. Primeros 5:")
        for e in errores[:5]:
            print(e)
    else:
        print("\n¡Proceso completado! Las imágenes con más blanco que negro fueron invertidas al estándar.")

if __name__ == "__main__":
    main()