from math import isfinite
from pathlib import Path

import torch
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from model import RedColaborativaMURA
from validacion import Puerta

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "modelo" / "best.pt"
IMAGE_SIZE = 320                      # igual que en el entrenamiento
THRESHOLD = 0.5
MAX_IMAGES = 8                        # radiografías por estudio
MAX_UPLOAD_BYTES = 64 * 1024 * 1024   # límite de toda la petición
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAMES = ["DenseNet121", "ResNet50", "EfficientNet-B0", "MobileNetV2"]
AVISO = ("Herramienta educativa / de investigación. No es un dispositivo médico y no "
         "reemplaza el diagnóstico de un profesional de la salud.")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


class ResizePad:
    """Redimensiona conservando la proporción y rellena con negro hasta cuadrado."""

    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        w, h = img.size
        scale = self.size / max(w, h)
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BILINEAR)
        canvas = Image.new("RGB", (self.size, self.size), (0, 0, 0))
        canvas.paste(img, ((self.size - img.width) // 2, (self.size - img.height) // 2))
        return canvas


preprocess = transforms.Compose([
    ResizePad(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No encuentro el modelo en {MODEL_PATH}. "
                                "Copia best.pt dentro de la carpeta 'modelo'.")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if isinstance(checkpoint.get(key), dict):
                checkpoint = checkpoint[key]
                break

    if not isinstance(checkpoint, dict):
        raise TypeError("best.pt debe contener un state_dict de PyTorch.")

    state_dict = {
        key.removeprefix("module."): value
        for key, value in checkpoint.items()
    }
    if "atencion.0.weight" not in state_dict or "clasificador.3.weight" not in state_dict:
        raise KeyError(
            "El archivo no es el modelo colaborativo de 4 IAs (faltan las capas "
            "'atencion' y 'clasificador'). ¿Es el model.pth antiguo de MobileNet?"
        )

    model = RedColaborativaMURA(pretrained=False, full_finetune=True)
    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()
    return model


model = load_model()
puerta = Puerta(DEVICE)


@app.get("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.get("/estado")
def estado():
    return jsonify(demo=False, modelos=MODEL_NAMES)


@app.post("/predict")
def predict():
    # Acepta una o varias radiografías de un mismo estudio (campo 'imagen' repetido, o 'imagenes')
    files = [f for f in request.files.getlist("imagen") if f and f.filename]
    if not files:
        files = [f for f in request.files.getlist("imagenes") if f and f.filename]
    if not files:
        return jsonify(error="Selecciona una imagen en el campo 'imagen'."), 400
    if len(files) > MAX_IMAGES:
        return jsonify(error=f"Máximo {MAX_IMAGES} imágenes por estudio."), 400

    tensors = []
    imagenes = []
    for uploaded in files:
        try:
            with Image.open(uploaded.stream) as image:
                rgb_original = image.convert("RGB")
                imagenes.append(rgb_original)
                # Elimina cualquier tinte sutil de pantalla/cámara para que la IA reciba la radiografía pura
                gris_limpio = image.convert("L").convert("RGB")
                tensors.append(preprocess(gris_limpio))
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
            return jsonify(error=f"No se pudo leer '{uploaded.filename}' como imagen válida."), 400

    # Puerta de validación: rechaza lo que no parece una radiografía (gatos, fotos, etc.)
    rechazo = puerta.revisar(imagenes, [f.filename for f in files])
    if rechazo:
        return jsonify(error=rechazo), 422

    batch = torch.stack(tensors).to(DEVICE)
    with torch.inference_mode():
        # promedio de la imagen y su versión volteada, como en la evaluación
        logits_a, weights = model.forward_con_pesos(batch)
        logits_b, _ = model.forward_con_pesos(torch.flip(batch, dims=[3]))
        logits = ((logits_a + logits_b) / 2).reshape(-1)
        probability = float(torch.sigmoid(logits.mean()))      # logit medio del estudio
        per_image = torch.sigmoid(logits).tolist()
        contribution = weights.mean(dim=0).tolist()

    if not isfinite(probability) or not 0.0 <= probability <= 1.0:
        app.logger.error("El modelo devolvió una probabilidad inválida: %s", probability)
        return jsonify(error="La salida del modelo no es una probabilidad entre 0 y 1."), 500

    label = "Anormal" if probability >= THRESHOLD else "Normal"
    return jsonify(
        probabilidad=probability,
        etiqueta=label,
        demo=False,
        imagenes=[{"archivo": f.filename, "puntaje": round(p, 4)} for f, p in zip(files, per_image)],
        aporte_por_modelo={n: round(w, 4) for n, w in zip(MODEL_NAMES, contribution)},
        modelo="Red colaborativa de 4 IAs: " + ", ".join(MODEL_NAMES),
        aviso=AVISO,
    )


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify(error="Las imágenes superan el límite de 64 MB en total."), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)

    app.run(host="127.0.0.1", port=5000, debug=False)