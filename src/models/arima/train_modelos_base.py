"""
02_train_modelos_base.py
==========================
Entrena los modelos base ARIMA, SARIMA, ARIMAX y SARIMAX (uno por cada
horizonte de pronóstico) usando órdenes (p, d, q) fijos, sin búsqueda de
hiperparámetros. El orden de diferenciación `d` se decide automáticamente
según el resultado del test de estacionariedad calculado en la etapa de
feature engineering. Los modelos entrenados se guardan en disco.
"""

import logging
import time
from pathlib import Path

from joblib import Parallel, delayed, load
from tqdm import tqdm

from forecasting_arima_utils import entrenar_modelos_base, tqdm_joblib

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parámetros
# ---------------------------------------------------------------------------
FILE_PATH = Path("../../resultados/arima/feature_engineering.pkl")
OUTPUT_DIR = Path("../../resultados/arima/")

ENTRENAR_MODELOS = True

P_FIJO, Q_FIJO = 1, 1
SEASONAL_ORDER = (1, 0, 1, 24)
HORIZONTES = [3, 48, 72]

TIPOS_MODELOS = ["ARIMA", "SARIMA"]
TIPOS_MODELOS += [f"{m}_H{h}" for m in ["ARIMAX", "SARIMAX"] for h in HORIZONTES]


def main() -> None:
    """Carga el feature store y entrena, en paralelo, los modelos base."""
    feature_data = load(FILE_PATH)
    logger.info("Feature store cargado desde: %s", FILE_PATH)

    train_y = feature_data["train_y"]
    train_x_h = feature_data["train_x_h"]
    train_x = feature_data["train_x"]
    es_estacionaria = feature_data["es_estacionaria"]

    # d=0 si la serie ya es estacionaria (según el test ADF de la etapa
    # anterior); d=1 si requiere una diferenciación.
    d_fijo = 0 if es_estacionaria else 1
    logger.info("Serie estacionaria: %s -> d=%d", es_estacionaria, d_fijo)

    if not ENTRENAR_MODELOS:
        logger.info("ENTRENAR_MODELOS=False: no se entrena ningún modelo.")
        return

    # Reanudación: si el .pkl de un tipo ya existe (p. ej. de una corrida
    # previa), se salta -- evita rehacer ajustes ya completos y, sobre
    # todo, reduce cuántos ajustes SARIMA/SARIMAX concurrentes se piden a
    # la vez (ver nota de memoria más abajo).
    pendientes = [
        tipo for tipo in TIPOS_MODELOS
        if not (OUTPUT_DIR / f"modelo_{tipo.lower()}_base").with_suffix(".pkl").exists()
    ]
    ya_listos = [t for t in TIPOS_MODELOS if t not in pendientes]
    if ya_listos:
        logger.info("Ya entrenados, se saltan: %s", ya_listos)
    if not pendientes:
        logger.info("Los %d modelos base ya están entrenados. Nada que hacer.", len(TIPOS_MODELOS))
        return

    # Paralelismo a nivel de PROCESO, acotado y sin sobresuscripción
    # (auditoría 2026-07-14/15): con n_jobs=-1 y BLAS libre, joblib pedía
    # hasta 16 procesos `loky`, cada uno ADEMÁS con su propio BLAS
    # multi-hilo por debajo -> sobresuscripción severa (procesos `loky`
    # que quedaban vivos a >90% de CPU incluso tras matar el proceso
    # principal). Además, entrenar SARIMA + los 3 SARIMAX_H* a la vez (4
    # modelos con seasonal_order=(1,0,1,24), hasta ~6-7GB de RSS cada uno
    # durante el ajuste) llegó a tumbar un worker por falta de memoria en
    # una máquina con ~15GB de RAM visibles (WSL2). Por eso aquí se acota
    # a 4, no a `len(pendientes)`: como mucho 4 modelos pesados a la vez,
    # cada uno con 1 solo hilo de BLAS (ver OMP_NUM_THREADS=1 etc. en
    # run_pipeline.sh). Si aun así un worker muere por memoria, esta misma
    # función es reanudable: correr de nuevo salta los que ya se guardaron
    # (y `retrain_sarimax_faltantes.py` cubre el caso ya conocido de los
    # SARIMAX_H* como reintento secuencial explícito).
    n_jobs = min(len(pendientes), 4)
    logger.info("Entrenando %d tipos de modelo (n_jobs=%d): %s", len(pendientes), n_jobs, pendientes)
    t0 = time.perf_counter()
    with tqdm_joblib(tqdm(total=len(pendientes), desc="modelos base", unit="modelo")):
        rutas_guardadas = Parallel(n_jobs=n_jobs)(
            delayed(entrenar_modelos_base)(
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
            for tipo in pendientes
        )
    duracion = (time.perf_counter() - t0) / 60

    for tipo, ruta in zip(pendientes, rutas_guardadas):
        logger.info("Modelo %s guardado en: %s", tipo, ruta)
    logger.info("Tiempo total de entrenamiento de modelos base: %.1f min", duracion)


if __name__ == "__main__":
    main()
