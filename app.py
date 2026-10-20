from math import isfinite
from pathlib import Path

import torch
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, UnidentifiedImageError
from torch import nn
from torchvision import transforms
from torchvision.models import mobilenet_v2

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pth"
IMAGE_SIZE = 224
THRESHOLD = 0.5
MAX_UPLOAD_BYTES = 16 * 1024 * 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict"):
            if key in checkpoint:
                checkpoint = checkpoint[key]
                break

    if not isinstance(checkpoint, dict):
        raise TypeError("model.pth debe contener un state_dict de PyTorch.")

    state_dict = {
        key.removeprefix("module."): value
        for key, value in checkpoint.items()
    }
    final_weight_key = next(
        (
            key
            for key in ("classifier.1.0.weight", "classifier.1.weight")
            if key in state_dict
        ),
        None,
    )
    if final_weight_key is None:
        raise KeyError(
            "model.pth no contiene la capa final MobileNet-V2 esperada "
            "(classifier[1])."
        )

    final_weights = state_dict[final_weight_key]
    if final_weights.ndim != 2 or final_weights.shape[0] != 1:
        raise ValueError(
            "La capa final classifier[1] debe producir una única salida binaria."
        )

    model = mobilenet_v2(weights=None)
    if final_weight_key == "classifier.1.0.weight":
        if final_weights.shape[1] != model.last_channel:
            raise ValueError("La capa final de model.pth no corresponde a MobileNet-V2.")
        model.classifier[1] = nn.Sequential(
            nn.Linear(model.last_channel, 1),
            nn.Sigmoid(),
        )
        output_is_probability = True
    else:
        if final_weights.shape[1] != model.last_channel:
            raise ValueError("La capa final de model.pth no corresponde a MobileNet-V2.")
        model.classifier[1] = nn.Linear(model.last_channel, 1)
        output_is_probability = False

    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()
    return model, output_is_probability


model, output_is_probability = load_model()


@app.get("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.get("/estado")
def estado():
    return jsonify(demo=False)


@app.post("/predict")
def predict():
    uploaded = request.files.get("imagen")
    if uploaded is None or not uploaded.filename:
        return jsonify(error="Selecciona una imagen en el campo 'imagen'."), 400

    try:
        with Image.open(uploaded.stream) as image:
            tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        return jsonify(error="No se pudo leer el archivo como imagen válida."), 400

    with torch.inference_mode():
        output = model(tensor)
        if not isinstance(output, torch.Tensor):
            app.logger.error("El modelo devolvió una salida que no es un tensor.")
            return jsonify(error="La salida del modelo no es un tensor válido."), 500
        output = output.reshape(-1)
        if not output_is_probability:
            output = torch.sigmoid(output)

    if output.numel() != 1:
        app.logger.error(
            "Se esperaba una salida escalar; el modelo devolvió %s valores.",
            output.numel(),
        )
        return jsonify(error="La salida del modelo no tiene el formato binario esperado."), 500

    probability = float(output.item())
    if not isfinite(probability) or not 0.0 <= probability <= 1.0:
        app.logger.error("El modelo devolvió una probabilidad inválida: %s", probability)
        return jsonify(error="La salida del modelo no es una probabilidad entre 0 y 1."), 500

    label = "Anormal" if probability >= THRESHOLD else "Normal"
    return jsonify(probabilidad=probability, etiqueta=label, demo=False)


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify(error="La imagen supera el límite de 16 MB."), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
