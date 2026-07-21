---
title: Alerta DCRNN Temperatura DMQ
emoji: 🌡️
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# Alerta temprana de temperatura — DCRNN (TFM, REMMAQ/DMQ Quito)

API FastAPI + front propio (HTML/CSS/JS sin frameworks ni build step) que
sirve el modelo DCRNN del TFM para las 6 estaciones REMMAQ del Distrito
Metropolitano de Quito, en sus 3 horizontes entrenados y seleccionables desde
la interfaz: corto (h=3h, ventana 24h), medio (h=48h, ventana 96h) y largo
(h=72h, ventana 168h). Corre **inferencia real** (no gráficos de arrays
precalculados) sobre ventanas históricas tomadas del split de test
(2024–2026, fuera de entrenamiento), muestra un mapa geográfico con las
estaciones coloreadas por temperatura pronosticada y marca alertas de
frío/calor por umbral.

Arquitectura: `api.py` (FastAPI) expone `/api/meta` y `/api/predict` y sirve
`static/` (el front) en la misma app/puerto — sin CORS, un solo contenedor.
Los umbrales de alerta se comparan **en el navegador** contra la última
predicción ya cargada, así que moverlos no llama al backend ni recalcula el
modelo (a diferencia de una versión previa en Streamlit, donde cualquier
widget —incluidos los umbrales— forzaba un rerun completo de la página).

Carpeta autocontenida: no depende de `artefactos/` ni `resultados/` del
repo del TFM en build time — todo lo necesario está en `model_data/`
(generado una vez con `prepare_data.py`, ver raíz del proyecto).

## Correr localmente sin Docker (desarrollo)

```bash
pip install -r requirements.txt  # + torch/torch_geometric_temporal del .venv del TFM
uvicorn api:app --reload --port 8000
```

Abrir http://localhost:8000

## Correr localmente con Docker

```bash
docker build -t dcrnn-alerta .
docker run -p 7860:7860 dcrnn-alerta
```

Abrir http://localhost:7860

## Desplegar en Hugging Face Spaces

1. Crear un Space nuevo en https://huggingface.co/new-space con **SDK: Docker**.
2. Inicializar esta carpeta (`app/`) como su propio repo git y apuntarlo al
   remoto del Space:

   ```bash
   cd app
   git init
   git lfs install
   git lfs track "model_data/*.npy" "model_data/*.pt"
   git add .
   git commit -m "Deploy dashboard DCRNN"
   git remote add space https://huggingface.co/spaces/<usuario>/<nombre-space>
   git push space main
   ```

   (Alternativa sin git: `huggingface_hub.upload_folder(repo_id=..., repo_type="space", folder_path="app")`.)
3. HF Spaces detecta el `Dockerfile` y este `README.md` (frontmatter
   `sdk: docker`, `app_port: 7860`) y construye la imagen automáticamente.

> Nota: `model_data/*.npy` y el checkpoint `.pt` pesan ~6.9MB en total — no
> es estrictamente necesario Git LFS, pero se recomienda si el Space va a
> versionarse con más historial.
