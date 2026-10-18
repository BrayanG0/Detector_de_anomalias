import os
from pathlib import Path
from PIL import Image, ImageFile
from tqdm import tqdm

# Habilitar carga de imágenes truncadas por seguridad
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ----------------------------------------------------------------------
# CONFIGURACIÓN — TRABAJO EN SITIO (IN-PLACE)
# ----------------------------------------------------------------------
CARPETA_DATASET = Path(r"/home/karva/JPGCONVERSIONDATASETCOPIA")
SPLITS = ["train_por_hueso", "valid_por_hueso"]  # O ajusta según tus carpetas
EXTENSIONES_VALIDAS = (".png", ".bmp", ".gif", ".tiff", ".webp", ".jpg", ".jpeg", ".JPG", ".JPEG")


# ----------------------------------------------------------------------
# CONTROL 1 — Verificación de existencia de la carpeta principal
# ----------------------------------------------------------------------
def control_1_verificacion(carpeta_base: Path):
    print("\n" + "=" * 70)
    print(f"CONTROL 1 — VERIFICACIÓN DE RUTA PRINCIPAL")
    print("=" * 70)

    if not carpeta_base.exists():
        print(f"❌ ERROR: La carpeta no existe: {carpeta_base}")
        return False
    
    print(f"  Carpeta objetivo: {carpeta_base}")
    return True


# ----------------------------------------------------------------------
# CONTROL 2 — Escaneo y conteo de archivos a procesar
# ----------------------------------------------------------------------
def control_2_conteo(split_path: Path):
    print("\n" + "=" * 70)
    print(f"CONTROL 2 — CONTEO DE ARCHIVOS ({split_path.name})")
    print("=" * 70)

    archivos = []
    for ext in EXTENSIONES_VALIDAS:
        archivos.extend(list(split_path.rglob(f"*{ext}")))
        archivos.extend(list(split_path.rglob(f"*{ext.upper()}")))
    
    # Eliminar duplicados si los hubiera por mayúsculas/minúsculas
    archivos = list(set(archivos))
    print(f"  Total de archivos compatibles encontrados: {len(archivos)}")
    return archivos


# ----------------------------------------------------------------------
# CONTROL 3 — Conversión, reemplazo y estandarización en la misma carpeta
# ----------------------------------------------------------------------
def control_3_procesar_en_sitio(archivos: list, split_path: Path):
    print("\n" + "=" * 70)
    print(f"CONTROL 3 — CONVERSIÓN Y REEMPLAZO EN SITIO")
    print("=" * 70)

    contador_exitos = 0
    contador_errores = 0
    errores_detalle = []

    for ruta_origen in tqdm(archivos, desc=f"Procesando {split_path.name}"):
        try:
            # Definir la ruta de destino con extensión .jpg en la misma ubicación
            ruta_destino = ruta_origen.with_suffix(".jpg")

            # Abrir, convertir a RGB estricto y guardar como JPG optimizado
            with Image.open(ruta_origen) as img:
                rgb_img = img.convert("RGB")
                rgb_img.save(ruta_destino, "JPEG", quality=95)
            
            # Si el archivo original tenía otra extensión (ej. .png), se elimina para dejar solo el .jpg
            if ruta_origen.resolve() != ruta_destino.resolve() and ruta_origen.exists():
                ruta_origen.unlink()

            contador_exitos += 1
        except Exception as e:
            contador_errores += 1
            errores_detalle.append((str(ruta_origen), str(e)))

    print(f"\n  - Imágenes procesadas con éxito: {contador_exitos}")
    print(f"  - Errores encontrados: {contador_errores}")

    if errores_detalle:
        print("\nPrimeros errores detallados:")
        for r, err in errores_detalle[:5]:
            print(f"    {r} -> {err}")

    return contador_exitos, contador_errores


# ----------------------------------------------------------------------
# EJECUCIÓN PRINCIPAL POR SPLIT
# ----------------------------------------------------------------------
def procesar_split(split_name: str):
    split_path = CARPETA_DATASET / split_name

    print("\n" + "#" * 70)
    print(f"###   PROCESANDO SPLIT EN SITIO: {split_name.upper()}   ###")
    print("#" * 70)

    if not split_path.exists():
        print(f"⚠️ AVISO: no existe la ruta {split_path}, se omite.")
        return

    archivos = control_2_conteo(split_path)
    if not archivos:
        print(f"⚠️ No hay archivos para procesar en {split_name}.")
        return

    exitos, errores = control_3_procesar_en_sitio(archivos, split_path)

    print("\n" + "=" * 70)
    print(f"RESUMEN FINAL — {split_name}")
    print("=" * 70)
    print(f"  Exitosos : {exitos}")
    print(f"  Errores  : {errores}")


def main():
    if not control_1_verificacion(CARPETA_DATASET):
        return

    for split in SPLITS:
        procesar_split(split)

    print("\n✅ ¡Proceso de conversión y reemplazo en sitio completado!")


if __name__ == "__main__":
    main()