#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
 test_diebold_mariano_todos.py
----------------------------------------------------------------------------
 Test de Diebold-Mariano por pares entre las TRES familias de modelos
 comparadas en notebooks/analisis_comparativo_modelos.ipynb: DCRNN, ClimaX
 y SARIMA. A diferencia de test_diebold_mariano_arima.py (que alinea SARIMA
 contra DCRNN sobre el 100% del test de DCRNN), aquí se **iguala la
 densidad de muestreo entre los tres modelos**: ClimaX solo predice sobre
 ventanas sin solapamiento (`stride = pred_len`), mucho más espaciadas que
 el ventaneo hora a hora de DCRNN y el walk-forward hora a hora de SARIMA.

 Para que las tres comparaciones (DCRNN-ClimaX, DCRNN-SARIMA,
 ClimaX-SARIMA) usen exactamente la MISMA muestra -- requisito para que
 sean comparables entre sí, no solo válidas por separado -- se calcula la
 intersección de instantes (fechaHora, estacion) donde los TRES modelos
 tienen predicción, y esa intersección (necesariamente al ritmo más
 disperso de ClimaX) es la que se usa para las tres pruebas DM de cada
 escenario.

 Reutiliza las funciones estadísticas genéricas de test_diebold_mariano.py.
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
MODELOS = ["DCRNN", "ClimaX", "SARIMA"]
PARES = [("DCRNN", "ClimaX"), ("DCRNN", "SARIMA"), ("ClimaX", "SARIMA")]


def veredicto_par(res: dict, alpha: float, m1: str, m2: str) -> str:
    """dbar = perdida(m1) - perdida(m2); dbar>0 => m1 peor => m2 mejor."""
    if "error" in res:
        return "indeterminado"
    if res["p_value"] < alpha:
        return f"{m2} mejor" if res["dbar"] > 0 else f"{m1} mejor"
    return "equivalentes"


def cargar_dcrnn_df(dcrnn_dir: Path, art_dir: Path, escenario: str, horizon: int, seq_len: int) -> pd.DataFrame:
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


def cargar_climax_df(climax_dir: Path, escenario: str) -> pd.DataFrame:
    df = pd.read_parquet(climax_dir / "predicciones" / f"predictions_{escenario}_plazo.parquet")
    df = df[df["horizon"] == df["horizon"].max()].copy()  # último paso -> mismo instante que DCRNN
    return df.rename(columns={"real_temp": "real", "pred_temp": "pred"})[["fechaHora", "estacion", "real", "pred"]]


def cargar_arima_df(arima_dir: Path, escenario: str, horizon: int, modelo: str, test_inicio: str, test_fin: str) -> pd.DataFrame:
    df = pd.read_parquet(arima_dir / "predicciones_tuning_estaciones.parquet")
    df = df[(df["modelo"] == modelo) & (df["horizonte"] == horizon)].copy()
    df = df[(df["fechaHora"] >= test_inicio) & (df["fechaHora"] <= test_fin)]
    return df[["fechaHora", "estacion", "real", "pred"]]


def alinear_tres(dfs: dict, station_order: list) -> tuple[dict, dict, int]:
    """
    Intersección estricta de (fechaHora, estacion) entre los 3 modelos.
    Devuelve {modelo: matriz pred (T,N)}, matriz real (T,N) y T.
    """
    m = dfs["DCRNN"].merge(dfs["ClimaX"], on=["fechaHora", "estacion"], suffixes=("_DCRNN", "_ClimaX"))
    m = m.merge(dfs["SARIMA"].rename(columns={"real": "real_SARIMA", "pred": "pred_SARIMA"}), on=["fechaHora", "estacion"])

    for a, b in [("real_DCRNN", "real_ClimaX"), ("real_DCRNN", "real_SARIMA")]:
        if not np.allclose(m[a], m[b], atol=1e-1):
            raise ValueError(f"Series 'real' no coinciden entre modelos ({a} vs {b}) en los instantes alineados.")

    tabla = m.pivot_table(index="fechaHora", columns="estacion",
                          values=["real_DCRNN", "pred_DCRNN", "pred_ClimaX", "pred_SARIMA"])
    tabla = tabla.reindex(columns=station_order, level=1).dropna()

    real = tabla["real_DCRNN"].to_numpy()
    preds = {modelo: tabla[f"pred_{modelo}"].to_numpy() for modelo in MODELOS}
    return preds, real, len(tabla)


def dm_par(errA: np.ndarray, errB: np.ndarray, h: int, loss: str, alpha: float, nombreA: str, nombreB: str) -> dict:
    LA, LB = perdida(errA, loss), perdida(errB, loss)
    d_global = (LA - LB).mean(axis=1)
    res = dm_desde_diferencial(d_global, h)
    res["veredicto"] = veredicto_par(res, alpha, nombreA, nombreB)
    res["mae_A"], res["mae_B"] = float(np.mean(np.abs(errA))), float(np.mean(np.abs(errB)))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Diebold-Mariano por pares: DCRNN, ClimaX, SARIMA (densidad de muestreo igualada)")
    ap.add_argument("--dcrnn", default="resultados/dcrnn")
    ap.add_argument("--climax", default="resultados/climax")
    ap.add_argument("--arima", default="resultados/arima/predicciones")
    ap.add_argument("--artefactos", default="artefactos")
    ap.add_argument("--modelo-arima", default="sarima")
    ap.add_argument("--test-inicio", default="2024-01-01")
    ap.add_argument("--test-fin", default="2026-03-31 23:00:00")
    ap.add_argument("--loss", default="ae", choices=["ae", "se"])
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--salida", default="resultados/diebold_mariano")
    args = ap.parse_args()

    dcrnn_dir, climax_dir = Path(args.dcrnn), Path(args.climax)
    arima_dir, art_dir = Path(args.arima), Path(args.artefactos)
    os.makedirs(args.salida, exist_ok=True)

    print("=" * 84)
    print(" TEST DE DIEBOLD-MARIANO POR PARES  |  DCRNN vs ClimaX vs SARIMA")
    print(" Densidad de muestreo igualada: interseccion estricta de instantes de los 3 modelos")
    print(f" Perdida principal: {'|error|' if args.loss=='ae' else 'error^2'} | alpha={args.alpha}")
    print("=" * 84)

    salida, resumen = {}, []
    for escenario, cfg in ESCENARIOS.items():
        h = cfg["horizon"]
        df_dcrnn, station_order = cargar_dcrnn_df(dcrnn_dir, art_dir, escenario, h, cfg["seq_len"])
        df_climax = cargar_climax_df(climax_dir, escenario)
        df_sarima = cargar_arima_df(arima_dir, escenario, h, args.modelo_arima, args.test_inicio, args.test_fin)

        preds, real, T = alinear_tres({"DCRNN": df_dcrnn, "ClimaX": df_climax, "SARIMA": df_sarima}, station_order)
        cobertura_climax = T / max(df_climax["fechaHora"].nunique(), 1)
        print(f"\n### {escenario.upper()} (h={h})  instantes alineados (los 3 modelos): {T}"
              f"  (= {cobertura_climax:.1%} de las ventanas de ClimaX)")

        if T < 30:
            print(f"  [OMITIDO] Muestra insuficiente para DM (T={T} < 30).")
            salida[escenario] = {"error": "muestra insuficiente", "T": int(T)}
            continue

        errores = {m: preds[m] - real for m in MODELOS}
        salida[escenario] = {"horizon": h, "T_alineado": int(T), "pares": {}}
        for a, b in PARES:
            res = dm_par(errores[a], errores[b], h, args.loss, args.alpha, a, b)
            salida[escenario]["pares"][f"{a}-{b}"] = res
            resumen.append({"escenario": escenario, "h": h, "T": T, "par": f"{a}-{b}",
                            "mae_A": res["mae_A"], "mae_B": res["mae_B"],
                            "dm_stat": res.get("dm_stat"), "p_value": res.get("p_value"), "veredicto": res["veredicto"]})
            print(f"  {a:<7}vs {b:<7} MAE {res['mae_A']:.4f} vs {res['mae_B']:.4f}"
                  f"  DM={res['dm_stat']:+.3f}  p={res['p_value']:.3g}  -> {res['veredicto']}")

    with open(os.path.join(args.salida, "dm_resultados_todos.json"), "w", encoding="utf-8") as f:
        json.dump(salida, f, ensure_ascii=False, indent=2, default=str)
    cols = ["escenario", "h", "T", "par", "mae_A", "mae_B", "dm_stat", "p_value", "veredicto"]
    with open(os.path.join(args.salida, "dm_resumen_todos.csv"), "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in resumen:
            f.write(",".join(str(r[c]) for c in cols) + "\n")

    print("\n" + "=" * 84)
    print(" RESUMEN — veredictos por escenario y par")
    for r in resumen:
        print(f" {r['escenario']:<7}h={r['h']:<4}{r['par']:<16}MAE {r['mae_A']:.4f} vs {r['mae_B']:.4f}"
              f"   DM={r['dm_stat']:.3f}   p={r['p_value']:.3g}   {r['veredicto']}")

    print("\n RESUMEN — victorias de DCRNN (con significancia estadística)")
    for r in resumen:
        if "DCRNN" in r["par"] and r["veredicto"] == "DCRNN mejor":
            print(f"   [{r['escenario']}] DCRNN mejor que {r['par'].replace('DCRNN-', '').replace('-DCRNN', '')}"
                  f"  (p={r['p_value']:.3g})")

    print(f"\n[OK] Resultados en: {os.path.abspath(args.salida)}")


if __name__ == "__main__":
    main()
