"""
retrain_sarimax_faltantes.py
=============================
Reentrena, de forma SECUENCIAL (n_jobs=1), los modelos base SARIMAX_H*
que no llegaron a guardarse en la corrida de `train_modelos_base.py`
(2026-07-14) por un OOM: con n_jobs=8, SARIMA + los 3 SARIMAX_H*
corrían a la vez (4 modelos con seasonal_order=(1,0,1,24), ~1-6GB de
RSS cada uno durante el ajuste) y el sistema (WSL2, ~15GB de RAM
visibles, no 32GB) mató uno de los procesos por falta de memoria.

Aquí no hay paralelismo entre modelos: se entrenan uno a la vez para
mantener el pico de memoria acotado a un solo modelo SARIMAX a la vez.
"""

import logging
import time
from pathlib import Path

from joblib import load

from forecasting_arima_utils import entrenar_modelos_base

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

FILE_PATH = Path("../../resultados/arima/feature_engineering.pkl")
OUTPUT_DIR = Path("../../resultados/arima/")

P_FIJO, Q_FIJO = 1, 1
SEASONAL_ORDER = (1, 0, 1, 24)

TIPOS_FALTANTES = ["SARIMAX_H3", "SARIMAX_H48", "SARIMAX_H72"]


def main() -> None:
    """Reentrena, uno por uno, los modelos base SARIMAX_H* faltantes."""
    feature_data = load(FILE_PATH)
    logger.info("Feature store cargado desde: %s", FILE_PATH)

    train_y = feature_data["train_y"]
    train_x_h = feature_data["train_x_h"]
    train_x = feature_data["train_x"]
    es_estacionaria = feature_data["es_estacionaria"]

    d_fijo = 0 if es_estacionaria else 1
    logger.info("Serie estacionaria: %s -> d=%d", es_estacionaria, d_fijo)
    logger.info("Reentrenando %d modelos faltantes, en secuencia (sin Parallel): %s", len(TIPOS_FALTANTES), TIPOS_FALTANTES)

    t0 = time.perf_counter()
    for i, tipo in enumerate(TIPOS_FALTANTES, start=1):
        ruta_esperada = (OUTPUT_DIR / f"modelo_{tipo.lower()}_base").with_suffix(".pkl")
        if ruta_esperada.exists():
            logger.info("[%d/%d] %s ya existe en %s -> se reutiliza, no se reentrena.", i, len(TIPOS_FALTANTES), tipo, ruta_esperada)
            continue

        ruta = entrenar_modelos_base(
            tipo=tipo,
            y_train=train_y,
            x_train_h=train_x_h,
            exog_train=train_x,
            p=P_FIJO,
            d=d_fijo,
            q=Q_FIJO,
            seasonal_order=SEASONAL_ORDER,
            output_dir=OUTPUT_DIR,
        )
        transcurrido = (time.perf_counter() - t0) / 60
        logger.info("[%d/%d] %s guardado -> %s | %.1f min transcurridos", i, len(TIPOS_FALTANTES), tipo, ruta, transcurrido)

    duracion = (time.perf_counter() - t0) / 60
    logger.info("Reentrenamiento de modelos faltantes completo en %.1f min", duracion)


if __name__ == "__main__":
    main()
