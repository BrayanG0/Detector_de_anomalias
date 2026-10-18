import os
import hashlib
from pathlib import Path
from collections import defaultdict, Counter

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError

# ----------------------------------------------------------------------
# CONFIGURACIÓN — NUEVA ESTRUCTURA
# ----------------------------------------------------------------------
DATASET_ROOT = Path(r"/home/karva/JPGCONVERSIONDATASETCOPIA")
SPLITS = ["train_por_hueso", "valid_por_hueso"]  
EXTENSIONES_VALIDAS = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff")
OUTPUT_ROOT = Path(r"./reporte_calidad")
OUTPUT_ROOT.mkdir(exist_ok=True)


# ----------------------------------------------------------------------
# CONTROL 1 — Estructura de carpetas
# ----------------------------------------------------------------------
def control_1_estructura(split_path: Path):
    print("\n" + "=" * 70)
    print(f"CONTROL 1 — ESTRUCTURA DE CARPETAS ({split_path.name})")
    print("=" * 70)

    if not split_path.exists():
        print(f"ERROR: la ruta {split_path} no existe.")
        return

    tipos_hueso = [c for c in sorted(split_path.iterdir()) if c.is_dir()]
    if not tipos_hueso:
        print("No se encontraron carpetas de tipos de hueso.")
        return

    for tipo in tipos_hueso:
        print(f"\nTipo de Hueso: {tipo.name}")
        pacientes = [p for p in sorted(tipo.iterdir()) if p.is_dir()]
        print(f"  Total pacientes/estudios: {len(pacientes)}")
        for paciente in pacientes[:5]:
            n_archivos = sum(1 for f in paciente.rglob("*") if f.is_file())
            print(f"    {paciente.name}: {n_archivos} archivo(s)")
        if len(pacientes) > 5:
            print(f"    ... y {len(pacientes) - 5} paciente(s) más")


# ----------------------------------------------------------------------
# CONTROL 2 — Cantidad de imágenes por tipo de hueso y etiqueta (Normal/Anormal)
# ----------------------------------------------------------------------
def control_2_cantidad(df: pd.DataFrame, split_name: str):
    print("\n" + "=" * 70)
    print(f"CONTROL 2 — CANTIDAD DE IMÁGENES Y ETIQUETAS ({split_name})")
    print("=" * 70)

    total_general = len(df)
    
    # Conteo por tipo de hueso
    conteo_por_hueso = df["tipo_hueso"].value_counts().to_dict()
    print("  Conteo por Tipo de Hueso:")
    for hueso, cant in conteo_por_hueso.items():
        print(f"    {hueso}: {cant} imágenes")

    # Conteo por etiqueta (normal / anormal)
    conteo_etiqueta = df["etiqueta"].value_counts().to_dict()
    print("\n  Conteo por Estado (Normal / Anormal / Desconocido):")
    for etiqueta, cant in conteo_etiqueta.items():
        print(f"    {etiqueta}: {cant} imágenes")

    # Conteo cruzado: Tipo de Hueso x Etiqueta
    print("\n  Desglose cruzado (Tipo de Hueso x Estado):")
    cruzado = pd.crosstab(df["tipo_hueso"], df["etiqueta"])
    print(cruzado.to_string())

    print(f"\n  ➤ TOTAL GENERAL EN {split_name}: {total_general} imágenes")
    return conteo_por_hueso, total_general, cruzado


# ----------------------------------------------------------------------
# CONTROL 3 — Generación del DataFrame maestro (Detectando Normal/Anormal)
# ----------------------------------------------------------------------
def control_3_dataframe(split_path: Path, split_name: str):
    print("\n" + "=" * 70)
    print(f"CONTROL 3 — GENERACIÓN DE DATAFRAME ({split_name})")
    print("=" * 70)

    filas = []
    for tipo_dir in sorted(split_path.iterdir()):
        if not tipo_dir.is_dir():
            continue
        tipo_hueso = tipo_dir.name

        for raiz, _, archivos in os.walk(tipo_dir):
            for archivo in archivos:
                if archivo.lower().endswith(EXTENSIONES_VALIDAS):
                    ruta_completa = Path(raiz) / archivo
                    try:
                        relativo = ruta_completa.relative_to(tipo_dir)
                        # Buscamos en toda la ruta relativa si aparece 'normal' o 'abnormal'/'anormal'
                        partes_str = "/".join(relativo.parts).lower()
                        
                        if "normal" in partes_str and "abnormal" not in partes_str:
                            etiqueta = "normal"
                        elif "abnormal" in partes_str or "anormal" in partes_str:
                            etiqueta = "anormal"
                        else:
                            etiqueta = "desconocido"

                        paciente = relativo.parts[0] if len(relativo.parts) > 1 else "desconocido"
                    except (ValueError, IndexError):
                        paciente = "desconocido"
                        etiqueta = "desconocido"

                    filas.append({
                        "split": split_name,
                        "tipo_hueso": tipo_hueso,
                        "paciente": paciente,
                        "etiqueta": etiqueta,  # <--- NUEVA COLUMNA
                        "archivo": archivo,
                        "ruta": str(ruta_completa),
                        "extension": ruta_completa.suffix.lower(),
                    })

    df = pd.DataFrame(filas)
    print(f"  DataFrame generado con {len(df)} filas y columnas: {list(df.columns)}")
    print("\n  Vista previa:")
    print(df.head(5).to_string(index=False))
    return df


# ----------------------------------------------------------------------
# CONTROL 4 — Formatos de archivo presentes
# ----------------------------------------------------------------------
def control_4_formatos(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("CONTROL 4 — FORMATOS DE ARCHIVO")
    print("=" * 70)

    conteo_ext = Counter(df["extension"])
    for ext, cant in conteo_ext.most_common():
        print(f"  {ext}: {cant} archivo(s)")


# ----------------------------------------------------------------------
# CONTROL 5 — Integridad (imágenes corruptas) con Image.verify()
# ----------------------------------------------------------------------
def control_5_integridad(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("CONTROL 5 — INTEGRIDAD (IMÁGENES CORRUPTAS)")
    print("=" * 70)

    estados = []
    errores = []
    for ruta in df["ruta"]:
        try:
            with Image.open(ruta) as img:
                img.verify()
            estados.append(True)
        except (UnidentifiedImageError, OSError, SyntaxError) as e:
            estados.append(False)
            errores.append((ruta, str(e)))

    df["integra"] = estados
    n_corruptas = (~df["integra"]).sum()

    print(f"  Imágenes válidas: {df['integra'].sum()}")
    print(f"  Imágenes corruptas: {n_corruptas}")
    return df


# ----------------------------------------------------------------------
# CONTROL 6 — Dimensiones y modos de canal
# ----------------------------------------------------------------------
def control_6_dimensiones(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("CONTROL 6 — DIMENSIONES Y MODOS DE CANAL")
    print("=" * 70)

    anchos, altos, modos = [], [], []
    for idx, row in df.iterrows():
        if not row["integra"]:
            anchos.append(None)
            altos.append(None)
            modos.append(None)
            continue
        try:
            with Image.open(row["ruta"]) as img:
                w, h = img.size
                modo = img.mode
            anchos.append(w)
            altos.append(h)
            modos.append(modo)
        except Exception:
            anchos.append(None)
            altos.append(None)
            modos.append(None)

    df["ancho"] = anchos
    df["alto"] = altos
    df["modo"] = modos
    return df


# ----------------------------------------------------------------------
# CONTROL 7 — Duplicados (MD5) y balance por tipo de hueso y etiqueta
# ----------------------------------------------------------------------
def calcular_md5(ruta, bloque=8192):
    h = hashlib.md5()
    try:
        with open(ruta, "rb") as f:
            for chunk in iter(lambda: f.read(bloque), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def control_7_duplicados_balance(df: pd.DataFrame, conteo_por_hueso: dict, output_dir: Path, split_name: str):
    print("\n" + "=" * 70)
    print(f"CONTROL 7 — DUPLICADOS (MD5) Y BALANCE ({split_name})")
    print("=" * 70)

    print("  Calculando hashes MD5...")
    df["md5"] = df["ruta"].apply(calcular_md5)

    hashes = defaultdict(list)
    for _, row in df.iterrows():
        if row["md5"]:
            hashes[row["md5"]].append(row["ruta"])

    duplicados = {h: rutas for h, rutas in hashes.items() if len(rutas) > 1}
    n_duplicados = sum(len(rutas) - 1 for rutas in duplicados.values())
    print(f"  Grupos de duplicados encontrados: {len(duplicados)}")
    print(f"  Archivos duplicados 'sobrantes': {n_duplicados}")

    if duplicados:
        filas_dup = []
        for h, rutas in duplicados.items():
            for r in rutas:
                filas_dup.append({"md5": h, "ruta": r})
        pd.DataFrame(filas_dup).to_csv(output_dir / "duplicados.csv", index=False)

    # Gráfico de barras agrupado por Normal/Anormal si es posible
    plt.figure(figsize=(12, 6))
    cruzado_plot = pd.crosstab(df["tipo_hueso"], df["etiqueta"])
    cruzado_plot.plot(kind="bar", figsize=(10, 5), color=["#4C72B0", "#DD8452", "#555555"])
    plt.title(f"Balance por Tipo de Hueso y Estado ({split_name})")
    plt.xlabel("Tipo de Hueso")
    plt.ylabel("Cantidad de Imágenes")
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="Estado")
    plt.tight_layout()
    plt.savefig(output_dir / "balance_tipos_hueso_etiqueta.png", dpi=150)
    plt.close()
    print(f"  Gráfico guardado en {output_dir / 'balance_tipos_hueso_etiqueta.png'}")

    return df


# ----------------------------------------------------------------------
# EJECUCIÓN DE CADA SPLIT
# ----------------------------------------------------------------------
def analizar_split(split_name: str):
    split_path = DATASET_ROOT / split_name
    output_dir = OUTPUT_ROOT / split_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 70)
    print(f"###  ANALIZANDO: {split_name.upper()}  ###")
    print("#" * 70)

    control_1_estructura(split_path)
    df = control_3_dataframe(split_path, split_name)
    conteo_por_hueso, total, cruzado = control_2_cantidad(df, split_name)
    control_4_formatos(df)
    df = control_5_integridad(df)
    df = control_6_dimensiones(df)
    df = control_7_duplicados_balance(df, conteo_por_hueso, output_dir, split_name)

    reporte_path = output_dir / "reporte_dataset.csv"
    df.to_csv(reporte_path, index=False)

    print("\n" + "=" * 70)
    print(f"RESUMEN FINAL — {split_name}")
    print("=" * 70)
    print(f"  Total imágenes analizadas : {len(df)}")
    print(f"  Imágenes corruptas        : {(~df['integra']).sum()}")
    print(f"  Reporte completo: {reporte_path}")

    return df


def main():
    resultados = {}
    for split in SPLITS:
        if (DATASET_ROOT / split).exists():
            resultados[split] = analizar_split(split)
        else:
            print(f"Advertencia: La carpeta {split} no existe en {DATASET_ROOT}")

    print("\n" + "=" * 70)
    print("COMPARATIVO ENTRE SPLITS")
    print("=" * 70)
    for split, df in resultados.items():
        print(f"  {split}: {len(df)} imágenes, {(~df['integra']).sum()} corruptas")
        if "etiqueta" in df.columns:
            conteo_et = df["etiqueta"].value_counts().to_dict()
            print(f"    Desglose: {conteo_et}")


if __name__ == "__main__":
    main()