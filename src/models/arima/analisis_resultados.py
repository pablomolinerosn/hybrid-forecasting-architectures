"""
analisis_resultados.py
========================
Réplica en script (headless, sin notebook) de la sección "6. Análisis de
resultados" de `entrenamiento_analisis_arima.ipynb`: calcula métricas
generales y por estación sobre las predicciones de tuning, genera gráficos
de ejemplo (real vs. predicho + intervalo de confianza) y guarda las tablas
de métricas por periodo (test completo 2024-2026 y marzo 2026) en un Excel
con varias hojas — igual que la celda final del notebook original.
"""

import matplotlib
matplotlib.use("Agg")  # headless: nunca abre ventana, solo guarda a disco

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from forecasting_arima_utils import calcular_metricas, grafico_predicciones_ci

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PREDICTIONS_DIR = Path("../../resultados/arima/predicciones/")
FIGURAS_DIR = PREDICTIONS_DIR / "figuras"

TEST_START, TEST_END = "2024-01-01", "2026-03-31"
MARZO_START, MARZO_END = "2026-03-01", "2026-03-31"


def guardar_figuras(df_pred: pd.DataFrame, modelo: str, fecha_inicio: str, fecha_fin: str, sufijo: str) -> None:
    """Genera los gráficos de `grafico_predicciones_ci` y los guarda a disco (headless)."""
    plt.close("all")
    grafico_predicciones_ci(df_pred, modelo=modelo, fecha_inicio=fecha_inicio, fecha_fin=fecha_fin)
    horizontes = (3, 48, 72)
    figuras = [plt.figure(n) for n in plt.get_fignums()]
    for h, fig in zip(horizontes, figuras):
        ruta = FIGURAS_DIR / f"prediccion_{modelo}_h{h}_{sufijo}.png"
        fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info("Figuras de %s (%s) guardadas en: %s", modelo, sufijo, FIGURAS_DIR)


def main() -> None:
    """Calcula métricas (general y por estación), genera figuras de ejemplo y exporta el Excel de métricas."""
    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Análisis general (promedio entre estaciones) ---
    df_pred = pd.read_parquet(PREDICTIONS_DIR / "predicciones_tuning.parquet")
    logger.info("Predicciones generales cargadas: %d filas", len(df_pred))

    for modelo in ["arima", "sarima", "arimax", "sarimax"]:
        guardar_figuras(df_pred, modelo, "2022-01-01", "2022-01-07", sufijo="general")

    metricas_general = calcular_metricas(df_pred)
    logger.info("Métricas generales (todo el rango):\n%s", metricas_general.to_string(index=False))

    # --- Análisis por estación ---
    df_pred_estaciones = pd.read_parquet(PREDICTIONS_DIR / "predicciones_tuning_estaciones.parquet")
    logger.info("Predicciones por estación cargadas: %d filas", len(df_pred_estaciones))

    metricas_por_estacion = calcular_metricas(df_pred_estaciones, estaciones=True)
    logger.info("Métricas por estación (todo el rango), primeras filas:\n%s", metricas_por_estacion.head(10).to_string(index=False))

    estacion_ejemplo = df_pred_estaciones["estacion"].unique()[0]
    guardar_figuras(
        df_pred_estaciones[df_pred_estaciones["estacion"] == estacion_ejemplo],
        "sarimax", MARZO_START, MARZO_END, sufijo=f"estacion_{estacion_ejemplo}",
    )

    # --- Métricas por periodo: test completo vs. marzo 2026 ---
    df_metricas_test_general = calcular_metricas(
        df_pred[(df_pred["fechaHora"] >= TEST_START) & (df_pred["fechaHora"] <= TEST_END)]
    )
    df_metricas_test_marzo = calcular_metricas(
        df_pred[(df_pred["fechaHora"] >= MARZO_START) & (df_pred["fechaHora"] <= MARZO_END)]
    )
    df_estaciones_metricas_test_general = calcular_metricas(
        df_pred_estaciones[(df_pred_estaciones["fechaHora"] >= TEST_START) & (df_pred_estaciones["fechaHora"] <= TEST_END)],
        estaciones=True,
    )
    df_estaciones_test_marzo = calcular_metricas(
        df_pred_estaciones[(df_pred_estaciones["fechaHora"] >= MARZO_START) & (df_pred_estaciones["fechaHora"] <= MARZO_END)],
        estaciones=True,
    )

    ruta_excel = PREDICTIONS_DIR / "predicciones_metricas.xlsx"
    with pd.ExcelWriter(ruta_excel, engine="openpyxl") as writer:
        df_metricas_test_general.to_excel(writer, sheet_name="General_Test", index=False)
        df_metricas_test_marzo.to_excel(writer, sheet_name="Marzo_Test", index=False)
        df_estaciones_metricas_test_general.to_excel(writer, sheet_name="General_Estaciones", index=False)
        df_estaciones_test_marzo.to_excel(writer, sheet_name="Marzo_Estaciones", index=False)

    logger.info("Métricas por periodo (test 2024-2026 y marzo 2026) guardadas en: %s", ruta_excel)
    logger.info("--- Comparativa marzo 2026 (general) ---\n%s", df_metricas_test_marzo.to_string(index=False))


if __name__ == "__main__":
    main()
