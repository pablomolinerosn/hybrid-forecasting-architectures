"""
04_predicciones_tuning.py
===========================
Genera las predicciones "rolling" (walk-forward) de los 8 modelos de
tuning (ARIMA, SARIMA, ARIMAX_H*, SARIMAX_H*), en dos modalidades
independientes (activables por separado con GENERAR_GENERAL /
GENERAR_POR_ESTACION):

- General: sobre la serie promedio entre todas las estaciones, tal como
  se entrenaron los modelos (comportamiento original). Guarda un solo
  archivo: predicciones_tuning.parquet.
- Por estación: filtra la base cruda por cada estación (en vez de
  promediar) y corre un walk-forward independiente por estación, las 6
  en paralelo. Guarda un archivo por estación:
  predicciones_tuning_<estacion>.parquet.

En ambos casos se usa la misma función `generar_predicciones_tuning`
(en `forecasting_arima_utils.py`), que no necesita saber si los datos
que recibe son el promedio o una estación puntual — solo recorre, en un
mismo `for`, los 8 modelos de tuning:

- ARIMA / SARIMA (sin exógenas): pronóstico recursivo hasta 72 horas,
  extrayendo los horizontes 3, 48 y 72 de cada ventana.
- ARIMAX_H*/SARIMAX_H* (entrenados sobre la serie ya desplazada -h, con
  exógenas sin desplazar): pronóstico directo con steps=1, reajustando
  el índice +h horas para volver a la hora real que representa.

Importante para el modo por estación: las exógenas de cada estación se
escalan con el MISMO `scaler` ajustado sobre el promedio de estaciones
en la etapa de feature engineering (guardado en el feature store) — no
se ajusta un scaler nuevo por estación.
"""

import logging
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed, load

from forecasting_arima_utils import cargar_datos, generar_predicciones_tuning, segmentar_datos

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parámetros
# ---------------------------------------------------------------------------
RAW_FILE_PATH = Path("../../data/processed/mdt_feature_store_2008.parquet")
FEATURE_STORE_PATH = Path("../../resultados/arima/feature_engineering.pkl")
TUNING_DIR = Path("../../resultados/arima/tuning/")
PREDICTIONS_DIR = Path("../../resultados/arima/predicciones/")

# Mismos valores que en 01_feature_engineering.py: deben mantenerse
# sincronizados, ya que el modo por estación vuelve a construir la
# serie objetivo y exógenas desde la base cruda.
TARGET_COL = "temperaturaMedia"
EXOG_COLS = [
    "humedadRelativa", "precipitacion", "presionBarometrica", "radiacionSolar",
    "viento_u", "viento_v", "precipitacion_log",
    "hora_sin", "hora_cos", "mes_sin", "mes_cos",
    "dia_semana_sin", "dia_semana_cos",
    "radiacionSolar_lag1", "radiacionSolar_lag2",
]
VAL_START, VAL_END = "2022-01-01", "2023-12-31 23:00:00"
TEST_START, TEST_END = "2024-01-01", "2026-03-31 23:00:00"

HORIZONTES = (3, 48, 72)
FECHA_INICIO = "2022-01-01 00:00:00"  # inicio de val_y
FECHA_HASTA = "2026-03-31"  # fin de test_y

# Qué generar en esta corrida.
GENERAR_GENERAL = True
GENERAR_POR_ESTACION = True
# Acotado a 3, no 6 (auditoría 2026-07-15): cada proceso carga los 8
# modelos de tuning completos (varios cientos de MB c/u; SARIMA y
# SARIMAX_H48 ya guardados pesan 898MB y 833MB respectivamente en disco).
# Con 6 estaciones en paralelo, el conjunto de modelos cargados podía
# rondar 13-17GB solo en objetos de modelo -> mismo riesgo de OOM que ya
# tumbó un worker en la etapa de modelos base. Con 3 procesos el pico
# esperado baja a ~6-9GB, dejando margen sobre los ~15GB reales de RAM
# visibles en WSL2.
N_JOBS_ESTACIONES = 3


def predecir_general(feature_data: dict) -> None:
    """
    Genera las predicciones de tuning sobre la serie promedio entre
    estaciones (comportamiento original, sin filtrar por estación).

    Parámetros
    ----------
    feature_data : dict
        Diccionario cargado desde el feature store
        (`feature_engineering.pkl`), con las llaves val_y/test_y/etc.
    """
    val_y = feature_data["val_y"]
    test_y = feature_data["test_y"]
    val_x_h = feature_data["val_x_h"]
    test_x_h = feature_data["test_x_h"]
    val_x = feature_data["val_x"]
    test_x = feature_data["test_x"]

    y_total = pd.concat([val_y, test_y]).asfreq("h")
    x_total_h = {h: pd.concat([val_x_h[h], test_x_h[h]]).asfreq("h") for h in HORIZONTES}
    exog_total = pd.concat([val_x, test_x]).astype("float32").reindex(y_total.index)

    logger.info(
        "[general] Rango a recorrer: %s -> %s (%d horas)",
        y_total.index.min(), y_total.index.max(), len(y_total),
    )

    output_path = PREDICTIONS_DIR / "predicciones_tuning.parquet"
    df_predicciones = generar_predicciones_tuning(
        y_total=y_total,
        x_total_h=x_total_h,
        exog_total=exog_total,
        tuning_dir=TUNING_DIR,
        output_path=output_path,
        horizontes=HORIZONTES,
        fecha_inicio=FECHA_INICIO,
        fecha_hasta=FECHA_HASTA,
        descripcion="Rolling forecast tuning [general]",
    )
    logger.info("[general] Predicciones generadas: %d filas -> %s", len(df_predicciones), output_path)


def preparar_datos_estacion(
    df_original: pd.DataFrame, estacion: str, scaler
) -> tuple[pd.Series, dict[int, pd.Series], pd.DataFrame]:
    """
    Filtra la base cruda a una sola estación y arma la serie objetivo +
    exógenas (val y test concatenados), escalando las exógenas con el
    scaler ya ajustado en la etapa de feature engineering.

    Parámetros
    ----------
    df_original : pd.DataFrame
        Base cruda completa (con columna 'estacion'), ya cargada con
        `cargar_datos`.
    estacion : str
        Nombre de la estación a filtrar.
    scaler : StandardScaler
        Scaler ya ajustado sobre el promedio de estaciones en
        `01_feature_engineering.py` (se reutiliza, no se reajusta).

    Retorno
    -------
    tuple[pd.Series, dict[int, pd.Series], pd.DataFrame]
        (y_total, x_total_h, exog_total) de esta estación, cubriendo
        val + test.
    """
    df_estacion = df_original.loc[df_original["estacion"] == estacion]
    df_agrupado = df_estacion.groupby("fechaHora", observed=False)[
        [TARGET_COL] + EXOG_COLS
    ].mean()

    serie_y = df_agrupado[TARGET_COL].asfreq("h").astype("float32")
    exogenas_x = df_agrupado[EXOG_COLS].asfreq("h").astype("float32")
    x_desplazada = {h: exogenas_x.shift(h) for h in HORIZONTES}

    val_y, val_x_h, val_x = segmentar_datos(
        serie_y, x_desplazada, exogenas_x, VAL_START, VAL_END, list(HORIZONTES), scaler
    )
    test_y, test_x_h, test_x = segmentar_datos(
        serie_y, x_desplazada, exogenas_x, TEST_START, TEST_END, list(HORIZONTES), scaler
    )

    y_total = pd.concat([val_y, test_y]).asfreq("h")
    x_total_h = {h: pd.concat([val_x_h[h], test_x_h[h]]).asfreq("h") for h in HORIZONTES}
    exog_total = pd.concat([val_x, test_x]).astype("float32").reindex(y_total.index)
    return y_total, x_total_h, exog_total


def predecir_estacion(estacion: str, df_original: pd.DataFrame, scaler) -> pd.DataFrame:
    """
    Genera y guarda las predicciones de tuning de una sola estación.

    Parámetros
    ----------
    estacion : str
        Nombre de la estación a procesar.
    df_original : pd.DataFrame
        Base cruda completa (con columna 'estacion').
    scaler : StandardScaler
        Scaler ya ajustado, reutilizado para escalar las exógenas de
        esta estación.

    Retorno
    -------
    pd.DataFrame
        Predicciones de esta estación (ya con la columna "estacion"),
        para poder combinarlas con las de las demás estaciones.
    """
    y_total, x_total_h, exog_total = preparar_datos_estacion(df_original, estacion, scaler)
    logger.info(
        "[%s] Rango a recorrer: %s -> %s (%d horas)",
        estacion, y_total.index.min(), y_total.index.max(), len(y_total),
    )

    output_path = PREDICTIONS_DIR / f"predicciones_tuning_{estacion}.parquet"
    df_predicciones = generar_predicciones_tuning(
        y_total=y_total,
        x_total_h=x_total_h,
        exog_total=exog_total,
        tuning_dir=TUNING_DIR,
        output_path=output_path,
        horizontes=HORIZONTES,
        fecha_inicio=FECHA_INICIO,
        fecha_hasta=FECHA_HASTA,
        descripcion=f"Rolling forecast tuning [{estacion}]",
        estacion=estacion,
    )
    logger.info("[%s] Predicciones generadas: %d filas -> %s", estacion, len(df_predicciones), output_path)
    return df_predicciones


def predecir_por_estacion(feature_data: dict) -> None:
    """
    Genera las predicciones de tuning de las 6 estaciones en paralelo, y
    además guarda un archivo combinado con todas juntas (columna
    "estacion") para facilitar el análisis agregado en el notebook.

    Parámetros
    ----------
    feature_data : dict
        Diccionario cargado desde el feature store, del cual se
        reutiliza el `scaler` ya ajustado.
    """
    df_original = cargar_datos(RAW_FILE_PATH)
    scaler = feature_data["scaler"]

    estaciones = sorted(df_original["estacion"].dropna().unique())
    logger.info("Estaciones a procesar (%d): %s", len(estaciones), estaciones)

    resultados = Parallel(n_jobs=min(N_JOBS_ESTACIONES, len(estaciones)))(
        delayed(predecir_estacion)(estacion, df_original, scaler) for estacion in estaciones
    )

    df_combinado = pd.concat(resultados, ignore_index=True)
    ruta_combinada = PREDICTIONS_DIR / "predicciones_tuning_estaciones.parquet"
    df_combinado.to_parquet(ruta_combinada, engine="pyarrow", index=False)
    logger.info("Predicciones combinadas de todas las estaciones guardadas en: %s", ruta_combinada)


def main() -> None:
    """Carga el feature store y genera las predicciones habilitadas (general y/o por estación)."""
    if not GENERAR_GENERAL and not GENERAR_POR_ESTACION:
        logger.info("GENERAR_GENERAL=False y GENERAR_POR_ESTACION=False: no hay nada que generar.")
        return

    feature_data = load(FEATURE_STORE_PATH)
    logger.info("Feature store cargado desde: %s", FEATURE_STORE_PATH)

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    if GENERAR_GENERAL:
        predecir_general(feature_data)

    if GENERAR_POR_ESTACION:
        predecir_por_estacion(feature_data)


if __name__ == "__main__":
    main()
