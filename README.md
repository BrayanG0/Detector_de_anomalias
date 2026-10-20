# OsteoScan · Detector de anomalías en radiografías

Página web que recibe una radiografía musculoesquelética y usa un modelo de
aprendizaje profundo (entrenado con el conjunto MURA) para estimar la
probabilidad de que el estudio sea anormal.

> Herramienta académica. No reemplaza el diagnóstico de un profesional de la salud.

## Dos modos de uso

| Modo | Cuándo | Qué instalar |
|------|--------|--------------|
| **Demostración** (sin modelo) | No existe `modelo/mi_modelo.keras` | `requirements.txt` |
| **Con modelo** | Ya copiaste tu modelo en `modelo/` | `requirements.txt` + `requirements-modelo.txt` |

En modo demostración la página funciona completa pero los resultados son
**simulados** (aparece un aviso en el panel de resultado). Sirve para ver y presentar el diseño.

## Ejecutar sin modelo

```bash
pip install -r requirements.txt
python app.py
```

## Ejecutar con modelo

1. Copia tu modelo: `cp tu_modelo.keras modelo/mi_modelo.keras`
   (si tiene otro nombre, cambia `MODEL_PATH` en `app.py`).
2. Ajusta en `app.py` la sección CONFIGURACIÓN (`IMG_SIZE`, `USAR_RGB`,
   `UMBRAL`, `normalizar()`) a como entrenaste.
3. Instala TensorFlow y ejecuta:

```bash
pip install -r requirements.txt -r requirements-modelo.txt
python app.py
```

Abre http://127.0.0.1:5000 (no abras el HTML con doble clic: necesita el servidor).

## Personalizar la presentación

En el `<script>` de `index.html`, bloque `PROYECTO`: tu nombre, programa y las
métricas del modelo.

## Notas

- Si usas PyTorch, en `app.py` hay un bloque comentado para reemplazar la carga
  y la predicción.
- Si `pip install tensorflow` falla, usa Python 3.11 o 3.12 (por ejemplo con conda).
