"""
01_feature_engineering.py
==========================
Prepara la serie objetivo (temperatura media) y sus exógenas a partir del
feature store, genera los desplazamientos por horizonte de pronóstico,
segmenta train/val/test, evalúa estacionariedad y persiste todo en un
único archivo para las siguientes etapas del pipeline (entrenamiento de
modelos ARIMA/SARIMAX).
"""

import logging
from pathlib import Path

import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from forecasting_arima_utils import cargar_datos, segmentar_datos, test_estacionariedad

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parámetros
# ---------------------------------------------------------------------------
FILE_PATH = Path("../../data/processed/mdt_feature_store_2008.parquet")
OUTPUT_PATH = Path("../../resultados/arima/feature_engineering.pkl")

TARGET_COL = "temperaturaMedia"
EXOG_COLS = [
    "humedadRelativa", "precipitacion", "presionBarometrica", "radiacionSolar",
    "viento_u", "viento_v", "precipitacion_log",
    "hora_sin", "hora_cos", "mes_sin", "mes_cos",
    "dia_semana_sin", "dia_semana_cos",
    "radiacionSolar_lag1", "radiacionSolar_lag2",
]

TRAIN_START, TRAIN_END = "2008-01-01", "2021-12-31 23:00:00"
VAL_START, VAL_END = "2022-01-01", "2023-12-31 23:00:00"
TEST_START, TEST_END = "2024-01-01", "2026-03-31 23:00:00"

HORIZONTES = [3, 48, 72]


def main() -> None:
    """Ejecuta el pipeline de preparación de datos y guarda los resultados."""
    # --- Carga de la base ---------------------------------------------------
    df_original = cargar_datos(FILE_PATH)
    logger.info("Base de datos cargada: %d filas", len(df_original))

    # --- Preparación de la serie objetivo y exógenas ------------------------
    # No se hace .copy() antes del groupby: `.mean()` ya devuelve un
    # DataFrame nuevo, por lo que copiar antes solo duplicaba memoria.
    df_agrupado = df_original.groupby("fechaHora", observed=False)[
        [TARGET_COL] + EXOG_COLS
    ].mean()

    serie_y = df_agrupado[TARGET_COL].asfreq("h").astype("float32")
    exogenas_x = df_agrupado[EXOG_COLS].asfreq("h").astype("float32")

    # Desplazamientos por horizonte de pronóstico.
    x_desplazada = {h: exogenas_x.shift(h) for h in HORIZONTES}

    # 2. Construir un índice común: solo las horas donde TODAS las series tienen datos válidos
    indices_validos = serie_y.dropna().index
    indices_validos = indices_validos.intersection(exogenas_x.dropna().index)

    for h, df in x_desplazada.items():
        indices_validos = indices_validos.intersection(df.dropna().index)

    # 3. Filtrar todas las series con el mismo índice
    serie_y = serie_y.loc[indices_validos]
    exogenas_x = exogenas_x.loc[indices_validos]
    x_desplazada = {h: df.loc[indices_validos] for h, df in x_desplazada.items()}

    # Nulos sobre la base agrupada completa (antes de segmentar), para
    # detectar problemas de calidad de datos previos al split.
    n_nulos = df_agrupado[[TARGET_COL] + EXOG_COLS].isnull().any(axis=1).sum()
    logger.info(
        "Nulos en base agrupada: %d obs (%.1f%%)", n_nulos, 100 * n_nulos / len(df_agrupado)
    )

    # --- Segmentación train / val / test ------------------------------------
    scaler = StandardScaler()
    train_y, train_x_h, train_x = segmentar_datos(
        serie_y, x_desplazada, exogenas_x, TRAIN_START, TRAIN_END, HORIZONTES, scaler, fit=True
    )
    val_y, val_x_h, val_x = segmentar_datos(
        serie_y, x_desplazada, exogenas_x, VAL_START, VAL_END, HORIZONTES, scaler
    )
    test_y, test_x_h, test_x = segmentar_datos(
        serie_y, x_desplazada, exogenas_x, TEST_START, TEST_END, HORIZONTES, scaler
    )

    # Verificación de la distribución de los segmentos (train/val/test).
    total = len(train_y) + len(val_y) + len(test_y)
    logger.info(
        "Distribución de segmentos -> Train: %d (%.1f%%) | Val: %d (%.1f%%) | Test: %d (%.1f%%)",
        len(train_y), 100 * len(train_y) / total,
        len(val_y), 100 * len(val_y) / total,
        len(test_y), 100 * len(test_y) / total,
    )

    # --- Análisis de estacionariedad -----------------------------------------
    sns.set_style("whitegrid")
    plt.rcParams.update({"font.size": 12, "figure.facecolor": "white"})
    logger.info("Analizando estacionariedad de la serie de entrenamiento...")
    es_estacionaria = test_estacionariedad(train_y, "temperatura (train)")

    # --- Persistencia ---------------------------------------------------------
    # `scaler` se guarda junto con los datos: cualquier exógena que se
    # escale más adelante (p. ej. por estación, en la etapa de
    # predicciones) debe usar este mismo scaler ya ajustado, no uno
    # nuevo, para mantener la misma escala que vieron los modelos
    # entrenados.
    feature_data = {
        "train_y": train_y,
        "train_x_h": train_x_h,
        "train_x": train_x,
        "val_y": val_y,
        "val_x_h": val_x_h,
        "val_x": val_x,
        "test_y": test_y,
        "test_x_h": test_x_h,
        "test_x": test_x,
        "es_estacionaria": es_estacionaria,
        "scaler": scaler,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(feature_data, OUTPUT_PATH, compress=3, protocol=4)
    logger.info("Feature store guardado en: %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
