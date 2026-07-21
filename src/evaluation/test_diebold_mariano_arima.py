#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
 test_diebold_mariano_arima.py
----------------------------------------------------------------------------
 Test de Diebold-Mariano (DM) entre DCRNN (mejor modelo de la familia
 GNN, según analisis_comparativo_modelos.ipynb) y SARIMA (mejor variante
 de la familia ARIMA/SARIMAX, según resultados/arima/predicciones/).

 A diferencia de test_diebold_mariano.py (A3T-GCN vs DCRNN, que comparan
 predicciones ya alineadas 1:1 por construcción -- mismo ventaneo,
 mismos índices de test), aquí los dos modelos vienen de pipelines
 completamente distintos:
   - DCRNN predice sobre ventanas (seq_len, horizon) indexadas por
     posición dentro del split de test (ver artefactos/splits.json +
     tiempos.npy).
   - SARIMA predice con un walk-forward hora a hora sobre fechaHora real,
     con horizontes que se recortan cerca del final de la ventana.

 Para que la comparación sea válida, se alinean AMBOS por 'fechaHora'
 real (no por posición) y por estación, quedándose solo con el
 subconjunto de instantes que ambos modelos pronosticaron para el mismo
 horizonte -- un inner join, no una suposición de índices iguales.

 Reutiliza las funciones estadísticas genéricas de test_diebold_mariano.py
 (no reimplementa el test DM).
============================================================================
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from test_diebold_mariano import dm_desde_diferencial, perdida  # noqa: E402

ESCENARIOS = {"corto": {"horizon": 3, "seq_len": 24}, "medio": {"horizon": 48, "seq_len": 96}, "largo": {"horizon": 72, "seq_len": 168}}
M1, M2 = "SARIMA", "DCRNN"  # convención: dbar > 0 => SARIMA peor => DCRNN mejor


def veredicto(res: dict, alpha: float) -> str:
    """
    Igual que test_diebold_mariano.veredicto, pero con las etiquetas
    (M1, M2) de ESTE módulo -- la versión importada usa los globals de
    test_diebold_mariano.py (siempre "A3T-GCN"/"DCRNN"), que aquí darían
    un veredicto con la etiqueta equivocada para el modelo 1.
    """
    if "error" in res:
        return "indeterminado"
    if res["p_value"] < alpha:
        return f"{M2} mejor" if res["dbar"] > 0 else f"{M1} mejor"
    return "equivalentes"


def cargar_dcrnn_df(dcrnn_dir: Path, art_dir: Path, escenario: str, horizon: int, seq_len: int) -> pd.DataFrame:
    """Reconstruye (fechaHora, estacion, real, pred) de DCRNN a partir de los .npy + índice temporal del split de test."""
    tiempos = pd.to_datetime(np.load(art_dir / "tiempos.npy", allow_pickle=True))
    splits = json.load(open(art_dir / "splits.json"))
    station_order = json.load(open(art_dir / "station_order.json"))
    test_a, _ = splits["test"]

    pred = np.load(dcrnn_dir / f"dcrnn_{escenario}_h{horizon}_pred_test.npy")
    real = np.load(dcrnn_dir / f"dcrnn_{escenario}_h{horizon}_true_test.npy")
    idx = test_a + np.arange(pred.shape[0]) + seq_len + horizon - 1
    fechas = tiempos[idx]

    filas = [
        pd.DataFrame({"fechaHora": fechas, "estacion": est, "real": real[:, j], "pred": pred[:, j]})
        for j, est in enumerate(station_order)
    ]
    return pd.concat(filas, ignore_index=True), station_order


def cargar_arima_df(arima_dir: Path, escenario: str, horizon: int, modelo: str = "sarima") -> pd.DataFrame:
    """Filtra las predicciones de tuning por estación al modelo y horizonte pedidos."""
    df = pd.read_parquet(arima_dir / "predicciones_tuning_estaciones.parquet")
    df = df[(df["modelo"] == modelo) & (df["horizonte"] == horizon)].copy()
    return df[["fechaHora", "estacion", "real", "pred"]]


def alinear(df_dcrnn: pd.DataFrame, df_arima: pd.DataFrame, station_order: list) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Inner join por (fechaHora, estacion). Devuelve matrices (T, N) de
    error para cada modelo, ya alineadas al mismo subconjunto de instantes,
    y T (tamaño de muestra efectivo).
    """
    m = df_dcrnn.merge(df_arima, on=["fechaHora", "estacion"], suffixes=("_dcrnn", "_arima"), how="inner")
    if not np.allclose(m["real_dcrnn"], m["real_arima"], atol=1e-2):
        raise ValueError("Las series 'real' de DCRNN y ARIMA no coinciden en los instantes alineados -- revisar fuente de datos.")

    tabla = m.pivot_table(index="fechaHora", columns="estacion", values=["real_dcrnn", "pred_dcrnn", "pred_arima"])
    tabla = tabla.reindex(columns=station_order, level=1).dropna()

    real = tabla["real_dcrnn"].to_numpy()
    pred_dcrnn = tabla["pred_dcrnn"].to_numpy()
    pred_arima = tabla["pred_arima"].to_numpy()
    return real, pred_dcrnn, pred_arima, len(tabla)


def main() -> None:
    ap = argparse.ArgumentParser(description="Diebold-Mariano SARIMA vs DCRNN (alineado por fechaHora)")
    ap.add_argument("--arima", default="resultados/arima/predicciones")
    ap.add_argument("--dcrnn", default="resultados/dcrnn")
    ap.add_argument("--artefactos", default="artefactos")
    ap.add_argument("--modelo-arima", default="sarima")
    ap.add_argument("--test-inicio", default="2024-01-01")
    ap.add_argument("--test-fin", default="2026-03-31 23:00:00")
    ap.add_argument("--loss", default="ae", choices=["ae", "se"])
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--salida", default="resultados/diebold_mariano")
    args = ap.parse_args()

    arima_dir, dcrnn_dir, art_dir = Path(args.arima), Path(args.dcrnn), Path(args.artefactos)
    os.makedirs(args.salida, exist_ok=True)

    print("=" * 76)
    print(f" TEST DE DIEBOLD-MARIANO  |  {M1} (modelo 1)  vs  {M2} (modelo 2)")
    print(f" Alineación: inner join por (fechaHora, estacion), no por posición")
    print(f" Pérdida principal: {'|error|' if args.loss=='ae' else 'error^2'} | alpha={args.alpha}")
    print("=" * 76)

    salida, resumen = {}, []
    for escenario, cfg in ESCENARIOS.items():
        df_dcrnn, station_order = cargar_dcrnn_df(dcrnn_dir, art_dir, escenario, cfg["horizon"], cfg["seq_len"])
        df_arima = cargar_arima_df(arima_dir, escenario, cfg["horizon"], modelo=args.modelo_arima)
        df_arima = df_arima[(df_arima["fechaHora"] >= args.test_inicio) & (df_arima["fechaHora"] <= args.test_fin)]

        real, pred_dcrnn, pred_arima, T = alinear(df_dcrnn, df_arima, station_order)
        cobertura = T / max(df_dcrnn["fechaHora"].nunique(), 1)
        print(f"\n### {escenario.upper()} (h={cfg['horizon']})  instantes alineados: {T}  (cobertura DCRNN: {cobertura:.1%})")

        if T < 30:
            print(f"  [OMITIDO] Muestra insuficiente para DM (T={T} < 30).")
            salida[escenario] = {"error": "muestra insuficiente", "T": int(T)}
            continue

        eA = pred_arima - real   # M1 = SARIMA
        eD = pred_dcrnn - real   # M2 = DCRNN
        maeA, maeD = float(np.mean(np.abs(eA))), float(np.mean(np.abs(eD)))

        LA, LD = perdida(eA, args.loss), perdida(eD, args.loss)
        d_global = (LA - LD).mean(axis=1)
        res_g = dm_desde_diferencial(d_global, cfg["horizon"])
        vg = veredicto(res_g, args.alpha)

        por_est = {}
        for j, est in enumerate(station_order):
            dj = perdida(eA[:, j], args.loss) - perdida(eD[:, j], args.loss)
            rj = dm_desde_diferencial(dj, cfg["horizon"])
            rj["veredicto"] = veredicto(rj, args.alpha)
            rj["mae_SARIMA"] = float(np.mean(np.abs(eA[:, j])))
            rj["mae_DCRNN"] = float(np.mean(np.abs(eD[:, j])))
            por_est[est] = rj

        salida[escenario] = {
            "horizon": cfg["horizon"], "T_alineado": int(T), "cobertura_dcrnn": cobertura,
            "mae_global": {M1: maeA, M2: maeD}, "global": {**res_g, "veredicto": vg},
            "por_estacion": por_est,
        }
        resumen.append({"escenario": escenario, "h": cfg["horizon"], "T": T, "mae_SARIMA": maeA, "mae_DCRNN": maeD,
                        "dm_stat": res_g.get("dm_stat"), "p_value": res_g.get("p_value"), "veredicto": vg})

        print(f"  MAE global: {M1} {maeA:.4f} °C | {M2} {maeD:.4f} °C")
        print(f"  GLOBAL: dbar={res_g['dbar']:+.5f} DM={res_g['dm_stat']:+.3f} p={res_g['p_value']:.3g} -> {vg}")
        for est in station_order:
            r = por_est[est]
            print(f"    {est:<12}{r['dm_stat']:>9.3f}{r['p_value']:>12.3g}   {r['veredicto']}")

    with open(os.path.join(args.salida, "dm_resultados_arima.json"), "w", encoding="utf-8") as f:
        json.dump(salida, f, ensure_ascii=False, indent=2, default=str)
    cols = ["escenario", "h", "T", "mae_SARIMA", "mae_DCRNN", "dm_stat", "p_value", "veredicto"]
    with open(os.path.join(args.salida, "dm_resumen_arima.csv"), "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in resumen:
            f.write(",".join(str(r[c]) for c in cols) + "\n")

    print("\n" + "=" * 76)
    print(" RESUMEN")
    for r in resumen:
        print(f" {r['escenario']:<9}{r['h']:>4}  T={r['T']:<8}MAE_SARIMA={r['mae_SARIMA']:.4f}  "
              f"MAE_DCRNN={r['mae_DCRNN']:.4f}  DM={r['dm_stat']:.3f}  p={r['p_value']:.3g}  {r['veredicto']}")
    print(f"\n[OK] Resultados en: {os.path.abspath(args.salida)}")


if __name__ == "__main__":
    main()
