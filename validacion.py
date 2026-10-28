"""Puerta de validación multicapa previa al modelo de diagnóstico.

Asegura que las imágenes subidas sean efectivamente radiografías médicas
de huesos (musculoesqueléticas) y con calidad suficiente para el análisis.

Capas de filtrado:
1. Resolución mínima (evita imágenes diminutas o pixeladas).
2. Filtro de color (las radiografías son monocromáticas; descarta fotos comunes a color).
3. Contraste mínimo (descarta imágenes lisas, vacías o sin rango dinámico).
4. Fondo blanco / Documento (descarta hojas escaneadas, recetas y PDFs).
5. Nitidez / Desenfoque (descarta imágenes excesivamente borrosas o movidas).
6. Perfil radiológico óseo (descarta fotos en blanco y negro de ojos, rostros,
   paisajes u objetos cotidianos que carecen del fondo oscuro de rayos X y la anatomía ósea).
7. Distancia estadística OOD de Mahalanobis (si existe modelo/ood_stats.npz).
"""
import logging
from pathlib import Path

import numpy as np
from scipy.ndimage import laplace

log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
STATS_PATH = BASE_DIR / "modelo" / "ood_stats.npz"

# Parámetros de validación calibrados
MIN_RESOLUCION = 160          # ancho y alto mínimos en píxeles
TOL_COLOR = 26.0              # diferencia media máxima entre canales RGB (permite fotos de celular a pantallas/negatoscopios)
MIN_STD_CONTRASTE = 14.0      # desviación estándar mínima de intensidad
MAX_PROP_BLANCO = 0.55        # proporción máxima de píxeles casi blancos (>220)
MAX_BRILLO_ESQUINAS = 185.0   # brillo medio máximo en las 4 esquinas
MAX_PERCENTIL_10 = 130.0      # el 10% más oscuro debe estar por debajo de este valor
MIN_VAR_NITIDEZ = 12.0        # varianza mínima del Laplaciano
MIN_PROP_FONDO_OSCURO = 0.14  # proporción mínima de fondo oscuro (< 55) típico de rayos X
MAX_BORDE_MEDIO = 95.0        # brillo medio máximo del marco exterior de la placa
MAX_MEDIA_GLOBAL = 125.0      # una radiografía de huesos no está dominada por tonos claros
MAX_MEDIANA_GLOBAL = 135.0    # la mediana no puede ser de tonos predominantemente claros
MARGEN_OOD = 1.10             # holgura sobre el umbral OOD calibrado
MARGEN = MARGEN_OOD           # alias de compatibilidad

_extractor = None
_tf = None


def es_gris(img, tol=TOL_COLOR):
    """True si la imagen es monocromática (canales R, G y B prácticamente iguales)."""
    pequena = np.asarray(img.convert("RGB").resize((128, 128)), dtype=np.int16)
    diferencia_canales = float((pequena.max(axis=2) - pequena.min(axis=2)).mean())
    return diferencia_canales < tol


def tiene_contraste(img, min_std=MIN_STD_CONTRASTE):
    """True si la imagen tiene variación de tonos y no es plana, vacía o monocroma lisa."""
    gris = np.asarray(img.convert("L").resize((128, 128)), dtype=np.float32)
    return float(gris.std()) >= min_std


def es_documento_o_fondo_blanco(img):
    """
    True si la imagen tiene las características típicas de un documento,
    dibujo o captura con fondo blanco, en lugar del fondo negro de una radiografía.
    """
    gris = np.asarray(img.convert("L").resize((128, 128)), dtype=np.float32)

    # 1. Más de la mitad de la imagen es blanco puro
    prop_blanco = float((gris > 220).mean())
    if prop_blanco > MAX_PROP_BLANCO:
        return True

    # 2. Las esquinas exteriores son claras (en radiografías el contorno es negro)
    c1 = gris[:16, :16]
    c2 = gris[:16, -16:]
    c3 = gris[-16:, :16]
    c4 = gris[-16:, -16:]
    media_esquinas = float((c1.mean() + c2.mean() + c3.mean() + c4.mean()) / 4.0)
    if media_esquinas > MAX_BRILLO_ESQUINAS:
        return True

    # 3. Ausencia total de tonos oscuros
    percentil_10 = float(np.percentile(gris, 10))
    if percentil_10 > MAX_PERCENTIL_10:
        return True

    return False


def esta_suficientemente_enfocada(img, min_var=MIN_VAR_NITIDEZ):
    """True si la imagen tiene bordes y detalles anatómicos discernibles."""
    gris = np.asarray(img.convert("L").resize((256, 256)), dtype=np.float32)
    var_lap = float(laplace(gris).var())
    return var_lap >= min_var


def tiene_perfil_radiografico(img):
    """
    Verifica que la imagen cumpla con la física de una radiografía musculoesquelética:
    1. Presencia de fondo oscuro (aire libre donde los rayos X impactan el chasis sin atenuación).
    2. Marco perimetral predominantemente oscuro (el contorno de la placa alrededor del cuerpo).
    3. Distribución global de luminancia no saturada en tonos claros (como ocurre en fotos de piel,
       ojos, rostros o paisajes en blanco y negro).
    """
    gris = np.asarray(img.convert("L").resize((160, 160)), dtype=np.float32)
    h, w = gris.shape

    # 1. Proporción de fondo oscuro (aire exterior)
    prop_oscuro = float((gris < 55).mean())
    if prop_oscuro < MIN_PROP_FONDO_OSCURO:
        return False, (f"no contiene el fondo oscuro característico de una radiografía "
                       f"({prop_oscuro*100:.1f}% vs mínimo {MIN_PROP_FONDO_OSCURO*100:.0f}%)")

    # 2. Marco exterior perimetral (los bordes de una radiografía tocan el aire)
    py, px = max(1, int(h * 0.08)), max(1, int(w * 0.08))
    borde = np.concatenate([
        gris[:py, :].ravel(),
        gris[-py:, :].ravel(),
        gris[:, :px].ravel(),
        gris[:, -px:].ravel()
    ])
    borde_medio = float(borde.mean())
    borde_mediana = float(np.median(borde))

    if borde_medio > MAX_BORDE_MEDIO and borde_mediana > MAX_BORDE_MEDIO:
        return False, (f"los bordes de la imagen son demasiado claros ({borde_medio:.0f}/255); "
                       f"en una radiografía el contorno debe ser el fondo oscuro de la placa")

    # 3. Luminancia global: una radiografía de huesos no es una foto clara de piel/paisaje
    media_global = float(gris.mean())
    mediana_global = float(np.median(gris))
    if media_global > MAX_MEDIA_GLOBAL or mediana_global > MAX_MEDIANA_GLOBAL:
        return False, (f"la luminosidad general es excesivamente alta para una placa radiográfica "
                       f"(mediana {mediana_global:.0f}/255)")

    return True, ""


def _cargar_extractor(device):
    global _extractor, _tf
    if _extractor is None:
        import torch
        from torchvision import models, transforms

        modelo = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        modelo.classifier = torch.nn.Identity()
        _extractor = modelo.eval().to(device)
        _tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return _extractor, _tf


def caracteristicas(imagenes, device):
    """Vectores de 1280 características (MobileNetV2 preentrenado en ImageNet)."""
    import torch

    ext, tf = _cargar_extractor(device)
    x = torch.stack([tf(i.convert("RGB")) for i in imagenes]).to(device)
    with torch.inference_mode():
        return ext(x).cpu().numpy().astype(np.float64)


def ajustar_estadisticas(F):
    """Media y precisión (inversa de la covarianza con regularización)."""
    mu = F.mean(axis=0)
    cov = np.cov(F, rowvar=False)
    cov = cov + 0.01 * np.mean(np.diag(cov)) * np.eye(cov.shape[0])
    return mu, np.linalg.pinv(cov)


def distancias(F, mu, prec):
    d = F - mu
    return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, prec, d), 0.0))


class Puerta:
    def __init__(self, device, ruta=STATS_PATH):
        self.device = device
        self.mu = self.prec = self.umbral = None
        if Path(ruta).exists():
            s = np.load(ruta)
            self.mu, self.prec, self.umbral = s["mu"], s["prec"], float(s["umbral"])
            _cargar_extractor(device)
        else:
            log.warning("No encuentro %s: se aplican los filtros analíticos completos "
                        "(color, resolución, contraste, fondo oscuro, nitidez y perfil óseo). "
                        "Ejecuta calibrar_ood.py para activar además la detección OOD profunda.", ruta)

    def revisar(self, imagenes, nombres):
        """
        Aplica la cadena de filtros en secuencia.
        Devuelve None si todas las imágenes pasan todos los filtros;
        de lo contrario, devuelve un mensaje explicativo para el usuario.
        """
        for img, nombre in zip(imagenes, nombres):
            # 1. Filtro de resolución mínima
            w, h = img.size
            if w < MIN_RESOLUCION or h < MIN_RESOLUCION:
                return (f"'{nombre}' tiene una resolución demasiado baja ({w}x{h} px). "
                        f"Sube una imagen de al menos {MIN_RESOLUCION}x{MIN_RESOLUCION} píxeles.")

            # 2. Filtro de color (las radiografías son en escala de grises)
            if not es_gris(img):
                return (f"'{nombre}' no parece una radiografía médica (contiene color). "
                        "Sube una imagen de radiografía en escala de grises.")

            # 3. Filtro de imagen plana / vacía
            if not tiene_contraste(img):
                return (f"'{nombre}' no tiene suficiente contraste o parece una imagen plana/vacía. "
                        "Asegúrate de subir una radiografía legible.")

            # 4. Filtro de documento / texto con fondo blanco
            if es_documento_o_fondo_blanco(img):
                return (f"'{nombre}' parece un documento, hoja escaneada o imagen con fondo blanco. "
                        "Las radiografías médicas deben mostrar fondo oscuro y estructuras óseas.")

            # 5. Filtro de desenfoque / nitidez
            if not esta_suficientemente_enfocada(img):
                return (f"'{nombre}' está excesivamente desenfocada o movida para un análisis confiable. "
                        "Sube una toma más nítida.")

            # 6. Filtro de perfil radiológico óseo (descarta ojos, rostros, paisajes y objetos en B/N)
            es_radio, detalle = tiene_perfil_radiografico(img)
            if not es_radio:
                return (f"'{nombre}' no parece una radiografía de huesos ni una imagen médica. "
                        f"Detalle: {detalle}. Por favor sube una radiografía musculoesquelética válida.")

        # 7. Filtro estadístico OOD profundo (si fue calibrado con radiografías reales)
        if self.umbral is not None:
            dist = distancias(caracteristicas(imagenes, self.device), self.mu, self.prec)
            for nombre, d in zip(nombres, dist):
                limite = self.umbral * MARGEN_OOD
                if d > limite:
                    log.info("Rechazada '%s': distancia %.1f > umbral %.1f", nombre, d, limite)
                    return (f"'{nombre}' no coincide con el patrón anatómico de una radiografía musculoesquelética. "
                            "Por favor sube una radiografía válida.")

        return None
