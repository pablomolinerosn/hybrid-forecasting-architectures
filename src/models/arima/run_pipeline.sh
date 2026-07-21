#!/bin/bash
# Corre las 4 etapas del pipeline ARIMA/SARIMAX en orden, mas el
# reentrenamiento de modelos faltantes y el analisis final. Cada etapa se
# salta sola si su salida ya existe (feature_engineering.py sobreescribe;
# tuning_modelos.py reanuda modelo por modelo si el .pkl ya esta guardado).
#
# Uso: bash src/models/arima/run_pipeline.sh   (desde la raiz del repo,
# o desde cualquier lado: el script se ubica solo con su propia ruta)
set -e
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source "$ROOT/../../../.venv/bin/activate" 2>/dev/null || true

# Acotar paralelismo a nivel de proceso y BLAS a 1 hilo por proceso: evita
# la sobresuscripcion de CPU/memoria documentada en README.md (seccion
# ARIMA/SARIMA) al entrenar SARIMA/SARIMAX con estacionalidad.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python3 feature_engineering.py
python3 train_modelos_base.py
python3 retrain_sarimax_faltantes.py    # completa SARIMAX_H* si train_modelos_base.py se quedo corto por memoria
python3 tuning_modelos.py               # reanuda solo, salta los .pkl que ya existan
python3 predicciones_tuning.py
python3 analisis_resultados.py
