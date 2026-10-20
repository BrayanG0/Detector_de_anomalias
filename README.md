# OsteoScan · Detector de anomalías en radiografías

Página web que recibe una radiografía musculoesquelética y usa un modelo de
aprendizaje profundo (entrenado con el conjunto MURA) para estimar la
probabilidad de que el estudio sea anormal.

> Herramienta académica. No reemplaza el diagnóstico de un profesional de la salud.

## Requisitos

Coloca el archivo de pesos `model.pth` en la raíz del proyecto, junto a
`app.py` e `index.html`. El checkpoint debe contener el `state_dict` de un
MobileNet-V2 con una capa final binaria en `classifier[1]`.

Instala las dependencias y ejecuta el servidor:

```bash
pip install -r requirements.txt -r requirements-modelo.txt
python app.py
```

Abre http://127.0.0.1:5000 (no abras el HTML con doble clic: necesita el servidor).
El servidor procesa imágenes RGB redimensionadas a 224 × 224 y normalizadas
con la media y desviación estándar de ImageNet.

## Personalizar la presentación

En el `<script>` de `index.html`, bloque `PROYECTO`: tu nombre, programa y las
métricas del modelo.

## Notas

- La salida del modelo se interpreta como probabilidad de anormalidad y el
  umbral de clasificación es `0.5`.
- La página es una herramienta académica y no reemplaza el diagnóstico de un
  profesional de la salud.
