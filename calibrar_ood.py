"""Calibra la puerta de validación con radiografías reales y permite probar imágenes.

Calibrar (una sola vez, con GPU tarda pocos minutos):
    python calibrar_ood.py --carpeta RUTA/A/IMAGENES_MURA --n 3000

Probar imágenes que NO deberían pasar (gato, tórax, captura de pantalla...):
    python calibrar_ood.py --probar gato.jpg torax.png captura.png
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from validacion import (MARGEN, STATS_PATH, ajustar_estadisticas, caracteristicas,
                        distancias, es_gris)

EXT = {".png", ".jpg", ".jpeg", ".bmp"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def listar(carpeta, n):
    rutas = [p for p in Path(carpeta).rglob("*") if p.suffix.lower() in EXT]
    random.Random(0).shuffle(rutas)
    return rutas[:n]


def extraer(rutas, lote=32):
    salida = []
    for i in range(0, len(rutas), lote):
        imgs = []
        for p in rutas[i:i + lote]:
            with Image.open(p) as im:
                imgs.append(im.convert("RGB").copy())
        salida.append(caracteristicas(imgs, DEVICE))
        print(f"  {min(i + lote, len(rutas))}/{len(rutas)}", end="\r")
    print()
    return np.concatenate(salida)


def calibrar(carpeta, n, percentil):
    rutas = listar(carpeta, n)
    if len(rutas) < 500:
        raise SystemExit(f"Solo encontré {len(rutas)} imágenes; usa al menos 500 (ideal 2000+).")
    print(f"Extrayendo características de {len(rutas)} imágenes en {DEVICE}...")
    F = extraer(rutas)

    corte = int(len(F) * 0.7)                  # 70 % ajusta, 30 % fija el umbral
    mu, prec = ajustar_estadisticas(F[:corte])
    d_val = distancias(F[corte:], mu, prec)
    umbral = float(np.percentile(d_val, percentil))

    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(STATS_PATH, mu=mu, prec=prec, umbral=umbral)
    print(f"Umbral (percentil {percentil} de validación): {umbral:.2f}  -> guardado en {STATS_PATH}")
    print(f"Distancias de validación: mediana {np.median(d_val):.2f}, máx {d_val.max():.2f}")


def probar(archivos):
    s = np.load(STATS_PATH)
    mu, prec, umbral = s["mu"], s["prec"], float(s["umbral"])
    limite = umbral * MARGEN
    print(f"Límite de aceptación: {limite:.2f}\n")
    for a in archivos:
        with Image.open(a) as im:
            img = im.convert("RGB").copy()
        d = float(distancias(caracteristicas([img], DEVICE), mu, prec)[0])
        gris = es_gris(img)
        veredicto = "ACEPTADA" if (gris and d <= limite) else "RECHAZADA"
        print(f"{veredicto:10s} dist={d:8.2f}  gris={gris}  {a}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--carpeta", help="carpeta con radiografías reales (busca en subcarpetas)")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--percentil", type=float, default=99)
    ap.add_argument("--probar", nargs="+", help="imágenes de prueba")
    args = ap.parse_args()
    if args.probar:
        probar(args.probar)
    elif args.carpeta:
        calibrar(args.carpeta, args.n, args.percentil)
    else:
        ap.print_help()

