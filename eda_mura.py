import re
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

# ------------------------------------------------------------------
# 0. Configuración (protocolo de esterilización visual, diapositiva 3)
# ------------------------------------------------------------------
sns.set_theme(style="whitegrid", palette="muted")
DATASET_DIR = Path("JPGCONVERSIONDATASETCOPIA/train")   # <-- ajusta la ruta
SAMPLE_SIZE = 2000        # submuestreo para el análisis pesado (pixeles)
RANDOM_STATE = 42


# ------------------------------------------------------------------
# 1. Construir el "manifest": un dataframe tabular a partir de las imágenes
#    Esto es lo que reemplaza al CSV en un EDA clásico.
# ------------------------------------------------------------------
def build_manifest(root: Path) -> pd.DataFrame:
    rows = []
    for img_path in root.rglob("*.jpg" if root else "*.png"):
        pass
    for img_path in root.rglob("*.*"):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        study_dir = img_path.parent.name          # e.g. study1_positive
        patient_dir = img_path.parent.parent.name  # e.g. patient05300
        bodypart_dir = img_path.parent.parent.parent.name  # e.g. XR_ELBOW

        m = re.match(r"study(\d+)_(positive|negative)", study_dir)
        study_num = int(m.group(1)) if m else None
        label = m.group(2) if m else None

        rows.append({
            "path": str(img_path),
            "bodypart": bodypart_dir.replace("XR_", ""),
            "patient": patient_dir,
            "study": study_num,
            "label": label,                 # variable categórica (target)
            "file_size_kb": img_path.stat().st_size / 1024,
        })
    return pd.DataFrame(rows)


print("Construyendo manifest (recorriendo carpetas)...")
df = build_manifest(DATASET_DIR)
print(f"Total de imágenes encontradas: {len(df)}")
df.to_csv("mura_manifest.csv", index=False)  # guarda para no repetir el recorrido


# ------------------------------------------------------------------
# 2. ANÁLISIS UNIVARIADO
#    Cualitativas -> bar plot / count plot (diapositiva "Análisis Univariado")
# ------------------------------------------------------------------

# 2.1 Balance de clases (positive vs negative) -> "consistencia con el
#     balanceo de clase de la Semana 6" (pregunta del kahoot)
fig, ax = plt.subplots(figsize=(6, 4))
order = df["label"].value_counts().index
sns.countplot(data=df, x="label", order=order, ax=ax)
ax.set_title("Distribución de clases: Normal vs. Anómalo")
ax.set_xlabel("Diagnóstico")
ax.set_ylabel("Número de imágenes")
plt.tight_layout()
plt.savefig("01_balance_clases.png", dpi=150)
plt.close()

# 2.2 Distribución por parte del cuerpo (bar plot ordenado)
fig, ax = plt.subplots(figsize=(8, 4))
order = df["bodypart"].value_counts().index
sns.countplot(data=df, y="bodypart", order=order, ax=ax)
ax.set_title("Cantidad de imágenes por región anatómica")
plt.tight_layout()
plt.savefig("02_distribucion_bodypart.png", dpi=150)
plt.close()

# 2.3 Tamaño de archivo (cuantitativa continua) -> histograma + KDE
fig, ax = plt.subplots(figsize=(7, 4))
sns.histplot(data=df, x="file_size_kb", kde=True, ax=ax)
skew = df["file_size_kb"].skew()
ax.set_title(f"Tamaño de archivo (KB) — skew = {skew:.2f}")
plt.tight_layout()
plt.savefig("03_hist_filesize.png", dpi=150)
plt.close()


# ------------------------------------------------------------------
# 3. EDA DE PÍXELES (equivalente al "Histograma de canales RGB / Rayos X"
#    de la diapositiva "EDA en Datos No Estructurados")
#    IMPORTANTE: se hace sobre una MUESTRA, no sobre todo el dataset
#    (mismo criterio que la alerta de rendimiento del pairplot).
# ------------------------------------------------------------------
sample_df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=RANDOM_STATE)

def extract_image_stats(path):
    with Image.open(path) as im:
        im = im.convert("L")  # escala de grises (rayos X)
        arr = np.asarray(im)
        w, h = im.size
        return {
            "width": w,
            "height": h,
            "aspect_ratio": w / h,
            "mean_intensity": arr.mean(),
            "std_intensity": arr.std(),
        }

print("Extrayendo estadísticas de píxeles de la muestra...")
stats = sample_df["path"].apply(extract_image_stats).apply(pd.Series)
sample_df = pd.concat([sample_df.reset_index(drop=True), stats.reset_index(drop=True)], axis=1)

# 3.1 Distribución de intensidad de píxel (histograma tipo "Rayos X")
fig, ax = plt.subplots(figsize=(7, 4))
sns.histplot(data=sample_df, x="mean_intensity", hue="label", kde=True,
             alpha=0.5, ax=ax)
ax.set_title("Intensidad media de píxel por clase (posible sesgo de iluminación)")
plt.tight_layout()
plt.savefig("04_hist_intensidad.png", dpi=150)
plt.close()

# 3.2 Dimensiones de imagen: Categórica (bodypart) vs Continua (width) -> violin plot
fig, ax = plt.subplots(figsize=(9, 5))
sns.violinplot(data=sample_df, x="bodypart", y="width", hue="label",
                split=True, inner="quartile", ax=ax)
ax.set_title("Ancho de imagen por región y clase (split=True)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("05_violin_width_bodypart.png", dpi=150)
plt.close()

# 3.3 Numérica vs Numérica: ancho vs. alto, coloreado por clase (scatter)
fig, ax = plt.subplots(figsize=(6, 6))
sns.scatterplot(data=sample_df, x="width", y="height", hue="label",
            alpha=0.4, s=40, ax=ax)
ax.set_title("Dimensiones de imagen (ancho vs. alto)")
plt.tight_layout()
plt.savefig("06_scatter_dimensiones.png", dpi=150)
plt.close()

# 3.4 Boxplot de outliers: intensidad media (regla IQR)
fig, ax = plt.subplots(figsize=(6, 4))
sns.boxplot(data=sample_df, x="label", y="mean_intensity", color="coral", ax=ax)
ax.set_title("Outliers de intensidad media por clase (regla IQR)")
plt.tight_layout()
plt.savefig("07_boxplot_outliers.png", dpi=150)
plt.close()

# 3.5 Heatmap triangular de correlación (multicolinealidad) entre
#     variables numéricas derivadas de la imagen
numeric_cols = ["width", "height", "aspect_ratio", "mean_intensity",
                 "std_intensity", "file_size_kb"]
corr = sample_df[numeric_cols].corr(method="spearman")  # spearman: robusto a outliers
mask = np.triu(np.ones_like(corr, dtype=bool))
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            vmin=-1, vmax=1, ax=ax)
ax.set_title("Correlación (Spearman) entre features derivadas de imagen")
plt.tight_layout()
plt.savefig("08_heatmap_correlacion.png", dpi=150)
plt.close()

print("Listo. Revisa los archivos 01_*.png a 08_*.png y mura_manifest.csv")
