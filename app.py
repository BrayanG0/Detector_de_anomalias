"""
Backend del detector de anomalías en radiografías.

Estructura de la carpeta:
    detector-huesos/
    ├── app.py
    ├── index.html
    └── modelo/
        └── mi_modelo.keras      <-- AQUÍ pones tu modelo

Ejecutar:
    pip install flask pillow numpy tensorflow
    python app.py
Abrir: http://127.0.0.1:5000

MODO DEMOSTRACIÓN: si todavía no existe el archivo del modelo, el servidor
arranca igual y devuelve un resultado simulado, para que puedas ver y probar
la página. En cuanto pongas tu modelo en la carpeta modelo/ y reinicies,
usa el modelo real automáticamente.
"""
import hashlib
import io
import os

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, ImageOps

# ----------------------------------------------------------------------
# 1) CONFIGURACIÓN: ajusta estos valores a como entrenaste tu modelo
# ----------------------------------------------------------------------
MODEL_PATH = "modelo/mi_modelo.keras"  # también sirve .h5 o carpeta SavedModel
IMG_SIZE = 224                         # mismo tamaño que usaste al entrenar
USAR_RGB = True                        # True si el modelo espera 3 canales
UMBRAL = 0.5                           # prob >= UMBRAL => "anormal"


def normalizar(arr):
    """Mismo preprocesamiento del entrenamiento.
    Por defecto escala a [0, 1]. Si usaste otra normalización
    (media/desviación, preprocess_input de un backbone, etc.) cámbiala aquí."""
    return arr / 255.0


# ----------------------------------------------------------------------
# 2) CARGA DEL MODELO (Keras / TensorFlow) o MODO DEMOSTRACIÓN
# ----------------------------------------------------------------------
MODO_DEMO = not os.path.exists(MODEL_PATH)
modelo = None

if MODO_DEMO:
    print("=" * 60)
    print(f"No se encontró el modelo en '{MODEL_PATH}'.")
    print("Arrancando en MODO DEMOSTRACIÓN (resultados simulados).")
    print("=" * 60)
else:
    import tensorflow as tf  # se importa solo si hay modelo

    modelo = tf.keras.models.load_model(MODEL_PATH)
    print("Modelo cargado:", MODEL_PATH)

# --- Si tu modelo es de PyTorch, reemplaza el bloque "else" de arriba y la
# --- función predecir() por algo así:
#
#   import torch
#   modelo = torch.load("modelo/mi_modelo.pt", map_location="cpu")
#   modelo.eval()
#
#   def predecir(x):                      # x: np.array (1, H, W, C)
#       t = torch.from_numpy(x).permute(0, 3, 1, 2).float()
#       with torch.no_grad():
#           salida = modelo(t)
#       return torch.sigmoid(salida).numpy()[0]


def predecir(x):
    return modelo.predict(x, verbose=0)[0]


# ----------------------------------------------------------------------
# 3) PREPROCESAMIENTO DE LA IMAGEN SUBIDA
# ----------------------------------------------------------------------
def preprocesar(bytes_imagen):
    img = Image.open(io.BytesIO(bytes_imagen)).convert("L")
    img = ImageOps.pad(img, (IMG_SIZE, IMG_SIZE))  # igual que en tu preprocesamiento
    if USAR_RGB:
        img = img.convert("RGB")
    arr = np.asarray(img, dtype="float32")
    if not USAR_RGB:
        arr = arr[..., np.newaxis]
    arr = normalizar(arr)
    return np.expand_dims(arr, axis=0)


# ----------------------------------------------------------------------
# 4) SERVIDOR WEB
# ----------------------------------------------------------------------
app = Flask(__name__)


@app.get("/")
def inicio():
    return send_from_directory(".", "index.html")


@app.post("/predict")
def predict():
    archivo = request.files.get("imagen")
    if archivo is None:
        return jsonify(error="No se recibió ninguna imagen."), 400

    contenido = archivo.read()
    try:
        x = preprocesar(contenido)
    except Exception:
        return jsonify(error="No se pudo leer el archivo como imagen."), 400

    if MODO_DEMO:
        # Resultado simulado pero estable: la misma imagen da siempre el mismo valor
        semilla = int(hashlib.md5(contenido).hexdigest(), 16) % 10_000
        p_anormal = 0.05 + 0.90 * (semilla / 10_000)
    else:
        salida = np.asarray(predecir(x)).flatten()
        # Salida sigmoide (1 valor) o softmax (2 valores: [normal, anormal])
        p_anormal = float(salida[0]) if salida.size == 1 else float(salida[1])

    return jsonify(
        probabilidad=p_anormal,
        etiqueta="anormal" if p_anormal >= UMBRAL else "normal",
        demo=MODO_DEMO,
    )


if __name__ == "__main__":
    app.run(debug=True)
