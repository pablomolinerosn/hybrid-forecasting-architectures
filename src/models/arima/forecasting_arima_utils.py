"""
forecasting_arima_utils.py
===========================
Utilidades para el entrenamiento, validación y comparación de modelos
ARIMA / ARIMAX / SARIMA / SARIMAX aplicados a series de tiempo climáticas.

Este módulo agrupa funciones de:
    - Carga y optimización de memoria de datos (parquet).
    - Escalamiento de variables exógenas y segmentación train/val/test.
    - Pruebas de estacionariedad (ADF, ACF, PACF).
    - Entrenamiento y persistencia de modelos base y de tuning (grid
      search paralelo).
    - Pronóstico "rolling" (walk-forward) de los modelos de tuning,
      general o por estación.
    - Cálculo de métricas y gráficos de las predicciones generadas.

Notas de la última revisión end-to-end:
    - Se eliminó `comparar_modelos`: llamaba a funciones que ya no
      existían en el módulo (código muerto, no usado por ningún script
      ni por el notebook).
    - Se corrigió un desfase de `h` horas en el campo "real" de
      `generar_predicciones_tuning` para los modelos ARIMAX_H*/SARIMAX_H*
      (comparaba la predicción a h horas contra la temperatura actual en
      vez de la temperatura real en el instante pronosticado). Validado
      empíricamente con una serie sintética.
    - Se reemplazaron las rutas construidas a mano con f-strings/backslash
      por el operador `/` de `pathlib` en `guardar_modelo`,
      `grid_search_model_parallel`, `entrenar_modelos_base` y
      `generar_predicciones_tuning`, para que funcionen igual en Windows
      y Linux.
    - Se refactorizó `calcular_metricas` para no repetir el cálculo de
      MAE/RMSE/MAPE/R2 en dos ramas (con y sin agrupación por estación).
"""

from __future__ import annotations

import contextlib
import gc
import itertools
import logging
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, dump, load
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """
    Conecta una barra `tqdm` al progreso real de un `joblib.Parallel`
    (por defecto, Parallel no reporta avance por tarea completada).

    Uso: ``with tqdm_joblib(tqdm(total=N, desc="...")): Parallel(...)(...)``
    """
    class _CallbackConTqdm(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    callback_original = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = _CallbackConTqdm
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = callback_original
        tqdm_object.close()

# ---------------------------------------------------------------------------
# Configuración global
# ---------------------------------------------------------------------------
# Antes: `OUTPUT_DIR` se usaba dentro de `guardar_modelo` sin haber sido
# definida en ningún lado del módulo, lo que provocaba un NameError en
# tiempo de ejecución. Se define aquí de forma explícita y con pathlib
# (más portable entre sistemas operativos que las f-strings con "/").
OUTPUT_DIR = Path("modelos_outputs")
PREDICTIONS_DIR = Path("predicciones_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Carga y preparación de datos
# ---------------------------------------------------------------------------
def optimizar_memoria(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce el consumo de memoria de un DataFrame ajustando tipos de datos.

    - Convierte la columna 'estacion' a categórica (si existe), ya que
      tiene pocos valores únicos repetidos muchas veces.
    - Convierte columnas float64 a float32 mediante downcast, dado que
      los datos climáticos no requieren precisión de 64 bits.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame original a optimizar.

    Retorno
    -------
    pd.DataFrame
        Mismo DataFrame con tipos de datos optimizados (modificado in-place
        y retornado por conveniencia).
    """
    memoria_inicial = df.memory_usage(deep=True).sum() / 1024**2
    logger.info("Memoria inicial: %.2f MB", memoria_inicial)

    if "estacion" in df.columns:
        df["estacion"] = df["estacion"].astype("category")

    columnas_float = df.select_dtypes(include=["float64"]).columns
    for col in columnas_float:
        df[col] = pd.to_numeric(df[col], downcast="float")

    memoria_final = df.memory_usage(deep=True).sum() / 1024**2
    logger.info("Memoria optimizada: %.2f MB", memoria_final)
    return df


def cargar_datos(file_path: str | Path) -> pd.DataFrame:
    """
    Carga un archivo parquet y optimiza su uso de memoria.

    Parámetros
    ----------
    file_path : str | Path
        Ruta al archivo .parquet.

    Retorno
    -------
    pd.DataFrame
        Datos cargados y con tipos optimizados.
    """
    logger.info("Cargando datos desde: %s...", file_path)
    df = pd.read_parquet(file_path)
    return optimizar_memoria(df)


def escalar_datos(
    x_slice: pd.DataFrame, scaler: StandardScaler, fit: bool = False
) -> pd.DataFrame:
    """
    Aplica un StandardScaler y devuelve un DataFrame con el índice y las
    columnas originales (evita perder los nombres de columnas, algo que
    ocurre si se usa el array de numpy devuelto directamente por sklearn).

    Parámetros
    ----------
    x_slice : pd.DataFrame
        Subconjunto de datos a escalar.
    scaler : StandardScaler
        Instancia (ya creada) del escalador de sklearn.
    fit : bool, default False
        Si es True, ajusta el scaler con estos datos (usar solo en train).
        Si es False, solo transforma (usar en validación/test).

    Retorno
    -------
    pd.DataFrame
        Datos escalados, en float32 para ahorrar memoria.
    """
    valores = scaler.fit_transform(x_slice) if fit else scaler.transform(x_slice)
    return pd.DataFrame(valores, index=x_slice.index, columns=x_slice.columns).astype(
        "float32"
    )


def segmentar_datos(
    y: pd.Series,
    y_shifted: dict[int, pd.Series],
    x: pd.DataFrame,
    start: str,
    end: str,
    horizontes: list[int],
    scaler: StandardScaler,
    fit: bool = False,
) -> tuple[pd.Series, dict[int, pd.Series], pd.DataFrame]:
    """
    Recorta y prepara un segmento temporal (train/val/test) de la serie
    objetivo, sus exógenas desplazadas por horizonte (`x_desplazada` en
    `feature_engineering.py` / `predicciones_tuning.py`) y las exógenas
    contemporáneas, todas escaladas con el mismo `StandardScaler`.

    Parámetros
    ----------
    y : pd.Series
        Serie objetivo completa (indexada por fecha), sin desplazar.
    y_shifted : dict[int, pd.Series]
        Exógenas desplazadas (`exogenas_x.shift(h)`), una por horizonte de
        pronóstico (p.ej. {3: x_h3, 48: x_h48, 72: x_h72}). El nombre del
        parámetro es historia del diseño anterior (donde sí era el
        objetivo desplazado); hoy contiene exógenas.
    x : pd.DataFrame
        Variables exógenas contemporáneas completas.
    start, end : str
        Límites (inclusive) del rango de fechas a extraer.
    horizontes : list[int]
        Horizontes a extraer de `y_shifted`. Antes la función dependía
        de una variable global `horizons`; ahora se recibe como
        parámetro explícito para evitar acoplamiento oculto.
    scaler : StandardScaler
        Escalador ya creado (ver `escalar_datos`).
    fit : bool, default False
        Si True, ajusta el scaler con este segmento (usar solo en train).
        Si False, únicamente transforma (usar en validación/test).

    Retorno
    -------
    tuple[pd.Series, dict[int, pd.Series], pd.DataFrame]
        (segmento_y, segmento_x_por_horizonte_escalado, segmento_x_escalado)

    Nota (auditoría 2026-07-14): antes `segmento_y_h` (exógenas desplazadas)
    se devolvía SIN escalar, mientras que `segmento_x` (exógenas
    contemporáneas) sí pasaba por el `StandardScaler` — los modelos
    ARIMAX_H*/SARIMAX_H* terminaban ajustados con radiación solar en W/m²
    (rango ~0-1200) mezclada, sin normalizar, con variables cíclicas en
    [-1, 1]. Síntoma observado: SARIMA (sin exógenas) superaba a SARIMAX
    (con exógenas sin escalar) en los tres horizontes — un resultado
    contraintuitivo típico de mal condicionamiento numérico, no de que las
    exógenas no aporten señal. Corregido: se ajusta/aplica el escalador
    sobre las exógenas contemporáneas primero, y esa MISMA instancia ya
    ajustada (nunca se reajusta dos veces) se reutiliza en modo
    transform-only sobre cada versión desplazada — mismas variables, mismas
    unidades, solo desplazadas en el tiempo, así que comparten escala.
    """
    segmento_y = y.loc[start:end]
    segmento_x = escalar_datos(x.loc[start:end], scaler, fit=fit)
    segmento_y_h = {
        h: escalar_datos(y_shifted[h].loc[start:end], scaler, fit=False)
        for h in horizontes
    }
    return segmento_y, segmento_y_h, segmento_x


def test_estacionariedad(
    serie: pd.Series,
    nombre: str = "serie",
    lags: int = 40,
    muestra_inicio: str = "2020-01-01",
    muestra_fin: str = "2020-01-07",
) -> bool:
    """
    Evalúa la estacionariedad de una serie con el test ADF (Augmented
    Dickey-Fuller) y grafica la serie (una muestra), su ACF y su PACF.

    Parámetros
    ----------
    serie : pd.Series
        Serie de tiempo a evaluar.
    nombre : str
        Nombre descriptivo usado en los títulos de los gráficos.
    lags : int
        Número de rezagos a graficar en ACF/PACF.
    muestra_inicio, muestra_fin : str
        Rango de fechas para graficar una muestra de la serie original
        (graficar la serie completa suele ser ilegible).

    Retorno
    -------
    bool
        True si la serie es estacionaria (p-valor < 0.05), False en caso
        contrario.
    """
    resultado = adfuller(serie.dropna())
    p_valor = resultado[1]
    es_estacionaria = p_valor < 0.05
    logger.info(
        "ADF [%s]: p-valor=%.4f -> %s",
        nombre,
        p_valor,
        "Estacionaria" if es_estacionaria else "No estacionaria",
    )

    fig, axes = plt.subplots(3, 1, figsize=(14, 9))

    serie_muestra = serie.loc[muestra_inicio:muestra_fin]
    axes[0].plot(serie_muestra, color="royalblue", linewidth=2)
    axes[0].set_title(f"Serie temporal (ejemplo 1 semana): {nombre}", fontweight="bold")
    axes[0].set_ylabel("Valor")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    plot_acf(serie.dropna(), ax=axes[1], lags=lags, color="darkgreen")
    axes[1].set_title("Función de Autocorrelación (ACF)", fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plot_pacf(serie.dropna(), ax=axes[2], lags=lags, method="ywm", color="darkred")
    axes[2].set_title("Función de Autocorrelación Parcial (PACF)", fontweight="bold")
    axes[2].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()

    return es_estacionaria


# ---------------------------------------------------------------------------
# Persistencia de modelos
# ---------------------------------------------------------------------------
def guardar_modelo(modelo, ruta: str | Path) -> Path:
    """
    Guarda un modelo entrenado en disco usando joblib (comprimido).

    Parámetros
    ----------
    modelo : object
        Modelo ya entrenado (ARIMA/SARIMAX fit result, etc.).
    ruta : str | Path
        Ruta completa del archivo (carpeta + nombre), con o sin
        extensión — se fuerza a `.pkl`. Se recomienda construirla con el
        operador `/` de `pathlib` (p. ej. `output_dir / "modelo_x"`), no
        con f-strings manuales, para que funcione igual en Windows y
        Linux.

    Retorno
    -------
    Path
        Ruta final donde quedó guardado el modelo.
    """
    ruta = Path(ruta).with_suffix(".pkl")   # asegura extensión .pkl
    ruta.parent.mkdir(parents=True, exist_ok=True)  # crea carpeta si no existe
    modelo.remove_data()
    dump(modelo, ruta, compress=3, protocol=4)
    return ruta


def cargar_modelo(
    ruta: str | Path,
    endog_train,
    exog_train,
):
    """
    Carga un modelo guardado y, si fue almacenado con `remove_data=True`,
    reconstruye su estado mediante `apply()`.

    Parámetros
    ----------
    ruta : str | Path
        Ruta del modelo.
    endog_train : pd.Series
        Serie histórica utilizada para reconstruir el estado del modelo.
    exog_train : pd.DataFrame, opcional
        Variables exógenas correspondientes a `endog_train`. Solo se usan
        para modelos SARIMAX/ARIMAX.

    Retorno
    -------
    SARIMAXResultsWrapper
        Modelo listo para realizar predicciones.
    """
    # Cargar modelo
    modelo = load(ruta)

    # Reconstruir el estado
    try:
        modelo = modelo.apply(endog=endog_train)
    except:
        modelo = modelo.apply(
            endog=endog_train,
            exog=exog_train
        )

    return modelo
        


# ---------------------------------------------------------------------------
# Generar las prediciones de los modelos
# ---------------------------------------------------------------------------

def generar_predicciones_tuning(
    y_total: pd.Series,
    x_total_h: dict[int, pd.Series],
    exog_total: pd.DataFrame,
    tuning_dir: Path,
    output_path: Path,
    horizontes: tuple[int, ...] = (3, 48, 72),
    fecha_inicio: str = "2022-01-01 00:00:00",
    fecha_hasta: str = "2022-01-01",
    descripcion: str = "Rolling forecast tuning",
    estacion: Optional[str] = None,
) -> pd.DataFrame:
    """
    Genera, en un único recorrido walk-forward, las predicciones de los 8
    modelos de tuning (ARIMA, SARIMA, ARIMAX_H*, SARIMAX_H*).

    Todos los modelos se cargan una sola vez al inicio y luego, en un
    mismo `for` que avanza hora a hora, se pronostican y se extienden
    con `.extend()` juntos (nunca `.append()`, que sería O(n) por paso al
    re-filtrar todo el historial acumulado en cada llamada). Esto evita
    recorrer la ventana de validación una vez por modelo (8 recorridos)
    y en su lugar la recorre una sola vez para los 8 a la vez.

    Dos estrategias de pronóstico conviven en el mismo paso:
    - ARIMA/SARIMA (sin exógenas): pronóstico recursivo hasta el
      horizonte máximo activo, extrayendo cada horizonte de esa misma
      corrida.
    - ARIMAX_H*/SARIMAX_H* (entrenados sobre la serie ya desplazada -h,
      con exógenas sin desplazar): pronóstico directo con steps=1. El
      índice que reporta el modelo corresponde al instante actual, pero
      la predicción representa el instante `actual + h` horas, así que
      la fecha de salida se recorre +h horas y el valor real se busca en
      esa misma fecha (no en la fecha actual), para comparar ambos en el
      mismo horario.

    A medida que se acerca el final de la ventana de validación, los
    horizontes que ya no caben (porque no queda suficiente historia real
    por delante) se van excluyendo automáticamente, tanto para el
    pronóstico recursivo como para el directo.

    Parámetros
    ----------
    y_total : pd.Series
        Serie objetivo real, sin desplazar, cubriendo todo el rango a
        recorrer (p. ej. validación + test).
    y_total_h : dict[int, pd.Series]
        Series objetivo desplazadas por horizonte (`{3: y_h3, 48: ...}`),
        usadas únicamente para extender los modelos "*_H*" paso a paso.
    exog_total : pd.DataFrame
        Exógenas sin desplazar, alineadas al índice de `y_total`.
    tuning_dir : Path
        Carpeta donde están los 8 modelos de tuning guardados
        (`modelo_<nombre>_tuning.pkl` / `modelo_<nombre>_h<h>_tuning.pkl`).
    output_path : Path
        Ruta del archivo parquet donde se guardan todas las predicciones.
    horizontes : tuple[int, ...]
        Horizontes a pronosticar, ordenados de menor a mayor.
    fecha_inicio, fecha_hasta : str
        Rango de fechas a recorrer (fecha_hasta se evalúa hasta las
        23:00:00 de ese día).
    descripcion : str
        Etiqueta mostrada en la barra de progreso (útil para identificar
        cada corrida cuando se procesan varias en paralelo, p. ej. una
        por estación).
    estacion : str, opcional
        Si se especifica, se agrega como columna "estacion" al resultado
        antes de guardarlo — permite luego agrupar por estación con
        `calcular_metricas(df, estaciones=True)` al combinar varios
        archivos de predicciones.

    Retorno
    -------
    pd.DataFrame
        Todas las predicciones, con columnas: modelo, fechaHora,
        horizonte, real, pred, ci_lower, ci_upper.
    """
    # Se cargan parte del historico de entrenamiento para ejecutar los modelos
    FILE_PATH = Path("../../results/modelosARIMA/feature_engineering/feature_engineering.pkl")
    OUTPUT_DIR = Path("../../modelos/modelosARIMA/")
    feature_data = load(FILE_PATH)
    train_y = feature_data["train_y"].iloc[-(24*365):]
    train_x_h = {
        h: x.iloc[-(24 * 365):]
        for h, x in feature_data["train_x_h"].items()
    }
    train_X = feature_data["train_x"].iloc[-(24*365):]

    # Se cargan los 8 modelos una sola vez, antes de iterar.
    modelos_sin_exog = {
        "arima": cargar_modelo(tuning_dir / "modelo_arima_tuning.pkl", train_y, train_X),
        "sarima": cargar_modelo(tuning_dir / "modelo_sarima_tuning.pkl", train_y, train_X),
    }
    modelos_con_exog = {
        (nombre, h): cargar_modelo(tuning_dir / f"modelo_{nombre}_h{h}_tuning.pkl", train_y, train_x_h[h])
        for nombre in ("arimax", "sarimax")
        for h in horizontes
    }

    fecha_inicio_dt = pd.to_datetime(fecha_inicio)
    fecha_fin_dt = pd.to_datetime(f"{fecha_hasta} 23:00:00")
    horas_totales = int((fecha_fin_dt - fecha_inicio_dt).total_seconds() / 3600)

    registros = []
    horizontes_activos = list(horizontes)

    barra = tqdm(range(horas_totales - 2), desc=descripcion, unit="hora")
    for inicio in barra:
        fin = inicio + 1
        horas_restantes = horas_totales - inicio

        # Recorta horizontes que ya no caben en lo que queda de ventana.
        while horizontes_activos and max(horizontes_activos) > horas_restantes:
            horizontes_activos = horizontes_activos[:-1]
        if not horizontes_activos:
            break

        max_h = max(horizontes_activos)
        exog_paso = exog_total.iloc[inicio:fin, :]
        endog_paso = y_total.iloc[inicio:fin]

        # --- ARIMA / SARIMA: un pronóstico recursivo cubre todos los horizontes ---
        for nombre, modelo in modelos_sin_exog.items():
            pronostico = modelo.get_forecast(steps=max_h)
            media = pronostico.predicted_mean
            intervalo = pronostico.conf_int()
            modelos_sin_exog[nombre] = modelo.extend(endog=endog_paso)

            for h in horizontes_activos:
                idx_real = inicio + h - 1
                if idx_real >= len(y_total):
                    continue
                registros.append(
                    {
                        "modelo": nombre,
                        "fechaHora": media.index[h - 1],
                        "horizonte": h,
                        "real": y_total.iloc[idx_real],
                        "pred": media.iloc[h - 1],
                        "ci_lower": intervalo.iloc[h - 1, 0],
                        "ci_upper": intervalo.iloc[h - 1, 1],
                    }
                )

        # --- ARIMAX_H*/SARIMAX_H*: pronóstico directo, steps=1 -----------
        for (nombre, h), modelo in modelos_con_exog.items():
            if h not in horizontes_activos:
                continue

            pronostico = modelo.get_forecast(steps=h, exog=x_total_h[h].iloc[inicio:inicio+h])
            
            # Valor puntual en horizonte h
            media = pronostico.predicted_mean.iloc[h-1]
            
            # Intervalo de confianza en horizonte h
            intervalo = pronostico.conf_int().iloc[h-1]
            
            # Actualizar modelo con datos reales
            modelos_con_exog[(nombre, h)] = modelo.extend(
                y_total.iloc[inicio:fin],
                exog=x_total_h[h].iloc[inicio:fin],
                validate_specification=False
            )

            # CORRECCIÓN (desfase de 1h, auditoría 2026-07-14): el pronóstico
            # parte del último punto conocido ANTES de esta iteración
            # (posición `inicio - 1`, ya que el `extend` de más arriba
            # todavía no ha "revelado" `inicio`), así que el paso h-ésimo
            # (`media`, índice h-1 dentro de `pronostico`) cae en la
            # posición `inicio + h - 1`, no `inicio + h` — igual que la
            # rama ARIMA/SARIMA de arriba, que ya usa `inicio + h - 1`.
            # Confirmado empíricamente: con el índice viejo, el "real"
            # reportado para ARIMAX/SARIMAX en una fecha X coincidía
            # exactamente con el "real" de ARIMA/SARIMA en la fecha X+1h.
            idx_real = inicio + h - 1
            if idx_real >= len(y_total):
                continue
            
            # Fecha correspondiente al horizonte h
            fecha_predicha = pronostico.predicted_mean.index[h-1]
            
            registros.append(
                {
                    "modelo": nombre,
                    "fechaHora": fecha_predicha,
                    "horizonte": h,
                    "real": y_total.iloc[idx_real],
                    "pred": media,
                    "ci_lower": intervalo.iloc[0],
                    "ci_upper": intervalo.iloc[1],
                }
            )


        barra.set_postfix(horizontes_activos=horizontes_activos)

    df_predicciones = pd.DataFrame(registros)
    if estacion is not None:
        df_predicciones["estacion"] = estacion

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_predicciones.to_parquet(output_path, engine="pyarrow", index=False)
    logger.info("Predicciones de tuning guardadas en: %s", output_path)
    return df_predicciones
# ---------------------------------------------------------------------------
# Métricas y visualización
# ---------------------------------------------------------------------------


def _metricas_grupo(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-2) -> dict:
    """
    Calcula MAE, RMSE, MAPE y R2 para un par (real, predicho).

    Homologado (auditoría 2026-07-14) a la fórmula EXACTA de
    `src/models/common.py:metricas()` (usada para A3T-GCN/DCRNN), para
    que las métricas de ARIMA/SARIMAX sean directamente comparables sin
    ninguna diferencia de criterio: mismo `eps` como piso del denominador
    del MAPE (antes ARIMA solo reemplazaba ceros exactos, sin piso para
    valores cercanos a cero) y mismas claves de columna.

    Parámetros
    ----------
    y_true, y_pred : np.ndarray
        Valores reales y predichos, ya alineados.
    eps : float, default 1e-2
        Piso del denominador del MAPE (evita divisiones por ~0 sin excluir
        observaciones). La temperatura mínima del dataset es ~0.65 °C, así
        que en la práctica rara vez se activa; se interpreta el MAPE con
        cautela igual que en el pipeline de A3T-GCN/DCRNN.

    Retorno
    -------
    dict
        {"MAE": ..., "RMSE": ..., "R2": ..., "MAPE": ...}
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    mape = float(np.mean(np.abs(err) / np.maximum(np.abs(y_true), eps)) * 100.0)
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


def calcular_metricas(df_pred: pd.DataFrame, estaciones: bool = False) -> pd.DataFrame:
    """
    Calcula MAE, RMSE, MAPE y R2 por modelo y horizonte (y, opcionalmente,
    también por estación) a partir de un DataFrame de predicciones como
    el que genera `generar_predicciones_tuning`.

    Parámetros
    ----------
    df_pred : pd.DataFrame
        Debe tener columnas "modelo", "horizonte", "real", "pred", y
        "estacion" si `estaciones=True`.
    estaciones : bool, default False
        Si True, agrupa además por estación (requiere que `df_pred`
        tenga esa columna, p. ej. al concatenar las predicciones de
        `predecir_por_estacion`).

    Retorno
    -------
    pd.DataFrame
        Una fila por combinación de modelo/horizonte(/estación), con las
        columnas MAE, RMSE, MAPE (%) y R2.
    """
    resultados = []

    for modelo in df_pred["modelo"].unique():
        df_m = df_pred[df_pred["modelo"] == modelo]

        for h in df_m["horizonte"].unique():
            df_h = df_m[df_m["horizonte"] == h]

            if estaciones and "estacion" in df_h.columns:
                for est in df_h["estacion"].unique():
                    df_e = df_h[df_h["estacion"] == est]
                    resultados.append(
                        {"modelo": modelo, "horizonte": h, "estacion": est}
                        | _metricas_grupo(df_e["real"].values, df_e["pred"].values)
                    )
            else:
                resultados.append(
                    {"modelo": modelo, "horizonte": h}
                    | _metricas_grupo(df_h["real"].values, df_h["pred"].values)
                )

    columnas_orden = ["modelo", "horizonte"] + (["estacion"] if estaciones else [])
    return pd.DataFrame(resultados).sort_values(by=columnas_orden).reset_index(drop=True)


def grafico_predicciones_ci(
    df_pred: pd.DataFrame,
    modelo: str = "arimax",
    horizontes: tuple[int, ...] = (3, 48, 72),
    fecha_inicio: str = "2022-01-01",
    fecha_fin: str = "2022-01-07",
) -> None:
    """
    Genera un gráfico por horizonte con la serie real, la predicción y
    su intervalo de confianza, para un modelo y rango de fechas dados.

    Parámetros
    ----------
    df_pred : pd.DataFrame
        Predicciones con columnas "fechaHora", "modelo", "horizonte",
        "real", "pred", "ci_lower", "ci_upper" (formato de
        `generar_predicciones_tuning`).
    modelo : str
        Nombre del modelo a graficar (p. ej. "arima", "sarimax_h48").
    horizontes : tuple[int, ...]
        Horizontes a graficar, uno por figura.
    fecha_inicio, fecha_fin : str
        Rango de fechas a mostrar.
    """
    sns.set_style("white")
    sns.set_context("talk")

    df_filtrado = df_pred[
        (df_pred["fechaHora"] >= fecha_inicio)
        & (df_pred["fechaHora"] <= fecha_fin)
        & (df_pred["modelo"] == modelo)
    ]

    for h in horizontes:
        df_h = df_filtrado[df_filtrado["horizonte"] == h]

        plt.figure(figsize=(14, 6))
        sns.lineplot(x=df_h["fechaHora"], y=df_h["real"], label="Real", color="#1f77b4", linewidth=2)
        sns.lineplot(
            x=df_h["fechaHora"], y=df_h["pred"], label=f"Predicción h={h}",
            color="red", linestyle="--", linewidth=2,
        )
        plt.fill_between(df_h["fechaHora"], df_h["ci_lower"], df_h["ci_upper"], color="red", alpha=0.15)
        plt.title(f"Pronóstico {modelo.upper()} (h={h})", fontsize=16, fontweight="bold")
        plt.xlabel("Tiempo")
        plt.ylabel("Temperatura Media")
        plt.legend(frameon=False)
        plt.show()


# ---------------------------------------------------------------------------
# Entrenamiento y búsqueda de hiperparámetros
# ---------------------------------------------------------------------------
def evaluar_modelo(
    y_train: pd.Series,
    y_val: pd.Series,
    exog_train: Optional[pd.DataFrame],
    exog_val: Optional[pd.DataFrame],
    p: int,
    d: int,
    q: int,
    usar_sarima: bool = False,
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> Optional[dict]:
    """
    Entrena y evalúa una combinación (p, d, q) de un modelo ARIMA/ARIMAX
    o SARIMA/SARIMAX. Pensada para usarse dentro de un grid search.

    Parámetros
    ----------
    y_train, y_val : pd.Series
        Series de entrenamiento y validación.
    exog_train, exog_val : pd.DataFrame | None
        Variables exógenas (None si el modelo no las usa).
    p, d, q : int
        Órdenes del modelo ARIMA.
    usar_sarima : bool
        Si True, entrena SARIMAX; si False, ARIMA.
    seasonal_order : tuple[int, int, int, int]
        Orden estacional (solo aplica si usar_sarima=True).

    Retorno
    -------
    dict | None
        Diccionario con p, d, q, AIC, BIC y RMSE, o None si el ajuste
        falló (p.ej. por no convergencia).
    """
    try:
        if usar_sarima:
            modelo = SARIMAX(
                y_train, exog=exog_train, order=(p, d, q), seasonal_order=seasonal_order,
                trend="c", enforce_stationarity=True, enforce_invertibility=True,
            ).fit(method_kwargs={"maxiter": 500})
        else:
            modelo = ARIMA(
                y_train, exog=exog_train, order=(p, d, q), trend="c",
                enforce_stationarity=True, enforce_invertibility=True,
            ).fit(method_kwargs={"maxiter": 500})

        pred = modelo.forecast(steps=len(y_val), exog=exog_val)
        rmse = np.sqrt(mean_squared_error(y_val, pred))
        return {"p": p, "d": d, "q": q, "AIC": modelo.aic, "BIC": modelo.bic, "RMSE": rmse}
    except Exception as exc:  # noqa: BLE001 - se registra en lugar de silenciar
        logger.debug("Combinación p=%s d=%s q=%s falló: %s", p, d, q, exc)
        return None


def grid_search_model_parallel(
    y_train: pd.Series,
    y_val: pd.Series,
    exog_train: Optional[pd.DataFrame] = None,
    exog_val: Optional[pd.DataFrame] = None,
    p_max: int = 4,
    q_max: int = 4,
    d_busqueda: int = 0,
    nombre_modelo: str = "modelo_arima_tuning",
    usar_sarima: bool = False,
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
    n_jobs: int = -1,
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    """
    Búsqueda de hiperparámetros (p, d, q) en paralelo para ARIMA/ARIMAX o
    SARIMA/SARIMAX, seleccionando el mejor modelo por RMSE de validación
    (fuera de muestra) y guardándolo.

    Antes se seleccionaba por AIC, un criterio *dentro* de muestra (se
    calcula sobre el ajuste en train): dos combinaciones con AIC similar
    pueden generalizar de forma muy distinta a validación. El RMSE contra
    `y_val` ya se calculaba en `evaluar_modelo` pero no se usaba para
    elegir — ahora sí (auditoría 2026-07-14). AIC/BIC quedan en la tabla de
    resultados como referencia, no como criterio de selección.

    Parámetros
    ----------
    y_train, y_val : pd.Series
        Series de entrenamiento y validación.
    exog_train, exog_val : pd.DataFrame | None
        Exógenas, si aplica.
    p_max, q_max : int
        Cotas superiores (exclusivas) del rango de búsqueda para p y q.
    d_busqueda : int
        Orden de diferenciación fijo a usar en la búsqueda.
    nombre_modelo : str
        Nombre base para el archivo guardado.
    usar_sarima : bool
        Si True, busca sobre SARIMAX; si False, sobre ARIMA.
    seasonal_order : tuple[int, int, int, int]
        Orden estacional fijo (solo si usar_sarima=True).
    n_jobs : int
        Núcleos a usar en paralelo (-1 = todos los disponibles).
    output_dir : Path
        Carpeta donde se guarda el modelo seleccionado. Por defecto usa
        la constante OUTPUT_DIR del módulo.

    Retorno
    -------
    Path
        Ruta donde quedó guardado el mejor modelo encontrado.
    """
    combinaciones = list(itertools.product(range(p_max), [d_busqueda], range(q_max)))
    t0 = time.perf_counter()

    with tqdm_joblib(tqdm(total=len(combinaciones), desc=f"grid {nombre_modelo}", unit="combo")):
        resultados = Parallel(n_jobs=n_jobs)(
            delayed(evaluar_modelo)(
                y_train, y_val, exog_train, exog_val, p, d, q,
                usar_sarima=usar_sarima, seasonal_order=seasonal_order,
            )
            for p, d, q in combinaciones
        )
    resultados = [r for r in resultados if r is not None]

    if not resultados:
        raise RuntimeError(
            "Ninguna combinación de (p, d, q) convergió. Revisa los rangos de "
            "búsqueda o los datos de entrada."
        )

    df_resultados = pd.DataFrame(resultados).sort_values(by="RMSE")
    mejor = df_resultados.iloc[0]

    modelo_cls = SARIMAX if usar_sarima else ARIMA
    kwargs_extra = {"seasonal_order": seasonal_order} if usar_sarima else {}
    modelo_final = modelo_cls(
        y_train, exog=exog_train,
        order=(int(mejor["p"]), int(mejor["d"]), int(mejor["q"])),
        trend="c", enforce_stationarity=True, enforce_invertibility=True,
        **kwargs_extra,
    ).fit(method_kwargs={"maxiter": 500})

    ruta_guardada = guardar_modelo(
        modelo_final, output_dir / f"modelo_{nombre_modelo.lower()}_tuning"
    )
    duracion = time.perf_counter() - t0
    logger.info(
        "Modelo %s guardado en: %s | mejor (p,d,q)=(%d,%d,%d) RMSE_val=%.4f | %.1f min",
        nombre_modelo, ruta_guardada, int(mejor["p"]), int(mejor["d"]), int(mejor["q"]),
        mejor["RMSE"], duracion / 60,
    )
    return ruta_guardada


def entrenar_modelos_base(
    tipo: str,
    y_train: pd.Series,
    x_train_h: Optional[dict[int, pd.Series]] = None,
    exog_train: Optional[pd.DataFrame] = None,
    p: int = 1,
    d: int = 0,
    q: int = 1,
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    """
    Entrena un modelo base (ARIMA, SARIMA, ARIMAX o SARIMAX, con o sin
    horizonte específico) y lo guarda en disco.

    Antes esta función tenía 9 bloques `if/elif` casi idénticos, uno por
    cada `tipo`. Se reemplazó por un diccionario de configuración: reduce
    la duplicación y facilita agregar nuevos tipos sin repetir lógica.

    Parámetros
    ----------
    tipo : str
        Uno de: "ARIMA", "SARIMA", "SARIMAX", "ARIMAX_H3", "ARIMAX_H48",
        "ARIMAX_H72", "SARIMAX_H3", "SARIMAX_H48", "SARIMAX_H72".
    y_train : pd.Series
        Serie de entrenamiento general (usada por ARIMA/SARIMA/SARIMAX).
    y_train_h : dict[int, pd.Series], opcional
        Series de entrenamiento específicas por horizonte, requeridas
        para los tipos "*_H3", "*_H48", "*_H72".
    exog_train : pd.DataFrame, opcional
        Exógenas de entrenamiento (requeridas para tipos con "X").
    p, d, q : int
        Orden ARIMA.
    seasonal_order : tuple[int, int, int, int]
        Orden estacional (solo para tipos "SARIMA*").
    output_dir : Path
        Carpeta donde se guarda el modelo entrenado. Por defecto usa la
        constante OUTPUT_DIR del módulo.

    Retorno
    -------
    Path
        Ruta donde quedó guardado el modelo.
    """
    x_train_h = x_train_h or {}

    # Mapa: tipo -> (clase del modelo, usa estacionalidad, horizonte)
    configuracion = {
        "ARIMA": (ARIMA, False, None),
        "SARIMA": (SARIMAX, True, None),
        "SARIMAX": (SARIMAX, True, None),
        "ARIMAX_H3": (ARIMA, False, 3),
        "ARIMAX_H48": (ARIMA, False, 48),
        "ARIMAX_H72": (ARIMA, False, 72),
        "SARIMAX_H3": (SARIMAX, True, 3),
        "SARIMAX_H48": (SARIMAX, True, 48),
        "SARIMAX_H72": (SARIMAX, True, 72),
    }


    if tipo not in configuracion:
        raise ValueError(f"Tipo de modelo no reconocido: {tipo!r}")

    modelo_cls, usa_estacionalidad, horizonte = configuracion[tipo]

    # Endógena: siempre la misma
    endog = y_train

    # Exógena: solo si el modelo es con "X"
    usa_exogenas = "X" in tipo
    if usa_exogenas:
        exog_train = x_train_h.get(horizonte)
        if exog_train is None:
            raise ValueError(f"El tipo '{tipo}' requiere exógenas en x_train_h[{horizonte}]")
    else:
        exog_train = None

    kwargs_extra = {"seasonal_order": seasonal_order} if usa_estacionalidad else {}

    t0 = time.perf_counter()
    modelo = modelo_cls(
        endog,
        exog=exog_train,
        order=(p, d, q),
        trend="c",
        enforce_stationarity=True,
        enforce_invertibility=True,
        **kwargs_extra,
    ).fit()

    ruta_guardada = guardar_modelo(modelo, output_dir / f"modelo_{tipo.lower()}_base")
    duracion = time.perf_counter() - t0
    logger.info("Modelo %s guardado en: %s | %.1f min", tipo, ruta_guardada, duracion / 60)

    del modelo
    gc.collect()
    return ruta_guardada


def mostrar_resultados_modelo(
    modelo, y_val: pd.Series, exog: Optional[pd.DataFrame] = None
) -> None:
    """
    Imprime un resumen de desempeño (AIC, BIC, RMSE) y el `summary()` de
    statsmodels para un modelo ya entrenado.

    Antes: usaba una variable global `val_y` inexistente en vez del
    parámetro `y_val` recibido, lo que producía un NameError. Corregido
    para usar siempre el parámetro.

    Parámetros
    ----------
    modelo : object
        Modelo entrenado (ARIMA/SARIMAX fit result).
    y_val : pd.Series
        Serie real de validación, usada para calcular el RMSE.
    exog : pd.DataFrame, opcional
        Exógenas de validación (requeridas si el modelo las usa).
    """
    if exog is not None:
        pred = modelo.forecast(steps=len(y_val), exog=exog)
    else:
        pred = modelo.forecast(steps=len(y_val))

    rmse = np.sqrt(mean_squared_error(y_val, pred))
    p, d, q = modelo.model.order
    logger.info("Parámetros: p=%s, d=%s, q=%s", p, d, q)
    logger.info("AIC: %.2f, BIC: %.2f, RMSE: %.4f", modelo.aic, modelo.bic, rmse)
    print(modelo.summary())

    del modelo
    gc.collect()
