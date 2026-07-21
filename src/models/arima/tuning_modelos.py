"""
03_tuning_modelos.py
======================
Búsqueda de hiperparámetros (p, d, q) para los modelos ARIMA/SARIMA y para
sus variantes con exógenas ARIMAX/SARIMAX. A diferencia de los modelos
base, aquí ya no existe una versión "normal" de ARIMAX/SARIMAX: solo se
entrenan las versiones rezagadas por horizonte de pronóstico (H3, H48,
H72), ya que son las que se usan en producción para pronóstico directo
multi-step.
"""

import logging
import time
from pathlib import Path

import pandas as pd
from joblib import load

from forecasting_arima_utils import grid_search_model_parallel

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parámetros
# ---------------------------------------------------------------------------
FILE_PATH = Path("../../resultados/arima/feature_engineering.pkl")
OUTPUT_DIR = Path("../../resultados/arima/tuning/")

ENTRENAR_MODELOS = True

GRID_P_MAX, GRID_Q_MAX = 2, 2
SEASONAL_ORDER = (1, 0, 1, 24)
HORIZONTES = [3, 48, 72]

# Paralelismo acotado, sin sobresuscripción (auditoría 2026-07-14): con
# n_jobs=-1 y BLAS libre, joblib pedía hasta 16 procesos `loky` que ADEMÁS
# multi-hilaban internamente -> sobresuscripción severa (quedaron `loky`
# vivos a >90% CPU tras matar el proceso principal). Probamos también
# n_jobs=1 + BLAS a 1 hilo: un solo ajuste SARIMAX con estacionalidad no
# terminó ni en 10 min (el álgebra lineal de estos modelos es pequeña, el
# multi-hilo de BLAS no compensa perder el paralelismo entre ajustes). La
# grilla es 2x2 = 4 combinaciones -> como mucho 4 procesos, cada uno con 1
# solo hilo de BLAS (ver OMP_NUM_THREADS=1 etc. en run_pipeline.sh): nunca
# más de 4 núcleos ocupados a la vez, de los 16 disponibles.
N_JOBS = 4


def _ruta_tuning(nombre_modelo: str) -> Path:
    """Ruta esperada del .pkl de tuning de un modelo, para chequeo de reanudación."""
    return (OUTPUT_DIR / f"modelo_{nombre_modelo.lower()}_tuning").with_suffix(".pkl")


def alinear_horizonte(y_h: pd.Series, x: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """
    Alinea una serie objetivo rezagada (con NaN al final por el
    `shift(-h)` aplicado en la etapa de feature engineering) con sus
    exógenas correspondientes, descartando las filas sin valor real.

    Parámetros
    ----------
    y_h : pd.Series
        Serie objetivo desplazada por un horizonte h.
    x : pd.DataFrame
        Exógenas completas (sin desplazar), indexadas por fecha.

    Retorno
    -------
    tuple[pd.Series, pd.DataFrame]
        (y alineada sin NaN, exógenas alineadas al mismo índice)
    """
    y_valido = y_h.dropna()
    x_alineado = x.loc[y_valido.index]
    return y_valido, x_alineado


def main() -> None:
    """Carga el feature store y ejecuta la búsqueda de hiperparámetros."""
    inicio = time.perf_counter()

    feature_data = load(FILE_PATH)
    logger.info("Feature store cargado desde: %s", FILE_PATH)

    train_y = feature_data["train_y"]
    train_x_h = feature_data["train_x_h"]
    train_x = feature_data["train_x"]
    val_y = feature_data["val_y"]
    val_x_h = feature_data["val_x_h"]
    val_x = feature_data["val_x"]
    es_estacionaria = feature_data["es_estacionaria"]

    d_busqueda = 0 if es_estacionaria else 1
    logger.info("Serie estacionaria: %s -> d=%d", es_estacionaria, d_busqueda)

    if not ENTRENAR_MODELOS:
        logger.info("ENTRENAR_MODELOS=False: no se ejecuta la búsqueda.")
        return

    rutas_guardadas: dict[str, Path] = {}
    TOTAL_MODELOS = 2 + 2 * len(HORIZONTES)  # ARIMA, SARIMA, (ARIMAX_H*, SARIMAX_H*) x horizontes
    contador = 0

    def _log_avance(nombre: str, ruta: Path) -> None:
        nonlocal contador
        contador += 1
        transcurrido = (time.perf_counter() - inicio) / 60
        logger.info(
            "[%d/%d] %s listo -> %s | %.1f min transcurridos desde el inicio del tuning",
            contador, TOTAL_MODELOS, nombre, ruta, transcurrido,
        )

    def _entrenar_o_reusar(nombre_modelo: str, **kwargs) -> Path:
        """
        Reanudación (auditoría 2026-07-15): si el .pkl de este modelo ya
        existe (de una corrida anterior interrumpida por el reinicio del
        sistema), se reutiliza tal cual y NO se vuelve a correr el grid
        search — evita rehacer horas de cómputo ya completadas.
        """
        ruta = _ruta_tuning(nombre_modelo)
        if ruta.exists():
            logger.info("%s ya existe en %s -> se reutiliza, no se reentrena.", nombre_modelo, ruta)
            return ruta
        return grid_search_model_parallel(nombre_modelo=nombre_modelo, output_dir=OUTPUT_DIR, n_jobs=N_JOBS, **kwargs)

    # --- ARIMA -----------------------------------------------------------
    rutas_guardadas["ARIMA"] = _entrenar_o_reusar(
        "ARIMA", y_train=train_y, y_val=val_y,
        p_max=GRID_P_MAX, q_max=GRID_Q_MAX, d_busqueda=d_busqueda,
    )
    _log_avance("ARIMA", rutas_guardadas["ARIMA"])

    # --- SARIMA ------------------------------------------------------------
    rutas_guardadas["SARIMA"] = _entrenar_o_reusar(
        "SARIMA", y_train=train_y, y_val=val_y,
        p_max=GRID_P_MAX, q_max=GRID_Q_MAX, d_busqueda=d_busqueda,
        usar_sarima=True, seasonal_order=SEASONAL_ORDER,
    )
    _log_avance("SARIMA", rutas_guardadas["SARIMA"])

    # --- ARIMAX y SARIMAX rezagados por horizonte ---------------------------
    for h in HORIZONTES:
        y_train_h, x_train_h = alinear_horizonte(train_y, train_x_h[h])
        y_val_h, x_val_h = alinear_horizonte(val_y, val_x_h[h])

        rutas_guardadas[f"ARIMAX_H{h}"] = _entrenar_o_reusar(
            f"ARIMAX_H{h}", y_train=y_train_h, y_val=y_val_h, exog_train=x_train_h, exog_val=x_val_h,
            p_max=GRID_P_MAX, q_max=GRID_Q_MAX, d_busqueda=d_busqueda,
        )
        _log_avance(f"ARIMAX_H{h}", rutas_guardadas[f"ARIMAX_H{h}"])

        rutas_guardadas[f"SARIMAX_H{h}"] = _entrenar_o_reusar(
            f"SARIMAX_H{h}", y_train=y_train_h, y_val=y_val_h, exog_train=x_train_h, exog_val=x_val_h,
            p_max=GRID_P_MAX, q_max=GRID_Q_MAX, d_busqueda=d_busqueda,
            usar_sarima=True, seasonal_order=SEASONAL_ORDER,
        )
        _log_avance(f"SARIMAX_H{h}", rutas_guardadas[f"SARIMAX_H{h}"])

    for nombre, ruta in rutas_guardadas.items():
        logger.info("Modelo %s guardado en: %s", nombre, ruta)

    duracion = time.perf_counter() - inicio
    logger.info("Tiempo total de búsqueda de hiperparámetros: %.1f min", duracion / 60)


if __name__ == "__main__":
    main()
