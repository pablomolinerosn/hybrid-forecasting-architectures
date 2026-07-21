#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
 tabla_resultados.py
----------------------------------------------------------------------------
 Genera tablas de métricas (MAE, RMSE, MAPE, R²) para A3T-GCN y DCRNN en los
 tres escenarios (corto/medio/largo), en cuatro cortes:
   1. TEST general              2. TEST por estación
   3. Marzo 2026 general        4. Marzo 2026 por estación

 Fuente de datos:
   - TEST: se toma de metricas_<modelo>.json (los valores ya reportados).
   - Marzo 2026: se recalcula desde los .npy de predicción, filtrando el mes.

 Salidas (en --salida):
   tabla_test_general.csv, tabla_test_estaciones.csv,
   tabla_marzo_general.csv, tabla_marzo_estaciones.csv,
   resultados_modelos.xlsx  (4 hojas)

 Uso:
   python tabla_resultados.py --artefactos artefactos \
       --a3t resultados/a3tgcn --dcrnn resultados/dcrnn --mes 2026-03
============================================================================
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

ESCENARIOS = ["corto", "medio", "largo"]
MODELOS = {                       # nombre -> (carpeta, prefijo_archivo, json)
    "A3T-GCN": ("a3tgcn", "metricas_a3tgcn.json"),
    "DCRNN":   ("dcrnn",  "metricas_dcrnn.json"),
}


def metricas(y_true, y_pred, eps=1e-2):
    yt = np.asarray(y_true, float).ravel()
    yp = np.asarray(y_pred, float).ravel()
    err = yp - yt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    sstot = float(np.sum((yt - yt.mean()) ** 2))
    r2 = float(1 - np.sum(err ** 2) / sstot) if sstot > 0 else float("nan")
    mape = float(np.mean(np.abs(err) / np.maximum(np.abs(yt), eps)) * 100)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


def cargar_preds(res_dir, prefijo, esc, met, tiempos, test_a):
    h = met[esc]["config"]["horizon"]; L = met[esc]["config"]["seq_len"]
    tag = f"{prefijo}_{esc}_h{h}"
    pred = np.load(os.path.join(res_dir, f"{tag}_pred_test.npy"))
    true = np.load(os.path.join(res_dir, f"{tag}_true_test.npy"))
    S = pred.shape[0]
    start = test_a + L + h - 1
    ts = tiempos[start:start + S]
    return pred, true, ts, h


def mascara_mes(ts, anio, mes):
    ini = np.datetime64(f"{anio}-{mes:02d}-01")
    fin = np.datetime64(f"{anio+1}-01-01") if mes == 12 else np.datetime64(f"{anio}-{mes+1:02d}-01")
    return (ts >= ini) & (ts < fin)


def main():
    ap = argparse.ArgumentParser(description="Tablas de resultados STGNN")
    ap.add_argument("--artefactos", default="artefactos")
    ap.add_argument("--resultados", default="resultados",
                    help="Carpeta base que contiene a3tgcn/ y dcrnn/")
    ap.add_argument("--mes", default="2026-03", help="Mes a resumir (YYYY-MM)")
    ap.add_argument("--salida", default="resultados/tablas")
    args = ap.parse_args()

    os.makedirs(args.salida, exist_ok=True)
    anio, mes = int(args.mes[:4]), int(args.mes[5:7])

    estaciones = json.load(open(os.path.join(args.artefactos, "station_order.json")))
    tiempos = np.load(os.path.join(args.artefactos, "tiempos.npy"))
    splits = json.load(open(os.path.join(args.artefactos, "splits.json")))
    test_a = splits["test"][0]

    mets = {}
    for modelo, (carpeta, jname) in MODELOS.items():
        mets[modelo] = json.load(open(os.path.join(args.resultados, carpeta, jname)))

    # ------------------------------------------------------------------ #
    # 1-2. TEST (desde los JSON ya reportados)
    # ------------------------------------------------------------------ #
    test_gen, test_est = [], []
    for modelo in MODELOS:
        met = mets[modelo]
        for esc in ESCENARIOS:
            h = met[esc]["config"]["horizon"]
            g = met[esc]["global"]
            test_gen.append({"Modelo": modelo, "Escenario": esc, "Horizonte (h)": h,
                             "MAE (°C)": g["MAE"], "RMSE (°C)": g["RMSE"],
                             "MAPE (%)": g["MAPE"], "R²": g["R2"]})
            for est in estaciones:
                e = met[esc]["por_estacion"][est]
                test_est.append({"Modelo": modelo, "Escenario": esc, "Horizonte (h)": h,
                                 "Estación": est, "MAE (°C)": e["MAE"], "RMSE (°C)": e["RMSE"],
                                 "MAPE (%)": e["MAPE"], "R²": e["R2"]})

    # ------------------------------------------------------------------ #
    # 3-4. MARZO (recalculado desde los .npy)
    # ------------------------------------------------------------------ #
    marzo_gen, marzo_est = [], []
    faltan_preds = False
    for modelo, (carpeta, _) in MODELOS.items():
        met = mets[modelo]; res_dir = os.path.join(args.resultados, carpeta)
        prefijo = carpeta                       # "a3tgcn" / "dcrnn"
        for esc in ESCENARIOS:
            h = met[esc]["config"]["horizon"]
            try:
                pred, true, ts, h = cargar_preds(res_dir, prefijo, esc, met, tiempos, test_a)
            except FileNotFoundError:
                faltan_preds = True
                continue
            m = mascara_mes(ts, anio, mes)
            if m.sum() == 0:
                continue
            g = metricas(true[m], pred[m])
            marzo_gen.append({"Modelo": modelo, "Escenario": esc, "Horizonte (h)": h,
                              "MAE (°C)": g["MAE"], "RMSE (°C)": g["RMSE"],
                              "MAPE (%)": g["MAPE"], "R²": g["R2"]})
            for j, est in enumerate(estaciones):
                e = metricas(true[m, j], pred[m, j])
                marzo_est.append({"Modelo": modelo, "Escenario": esc, "Horizonte (h)": h,
                                  "Estación": est, "MAE (°C)": e["MAE"], "RMSE (°C)": e["RMSE"],
                                  "MAPE (%)": e["MAPE"], "R²": e["R2"]})

    # ------------------------------------------------------------------ #
    # ensamblar, redondear, guardar
    # ------------------------------------------------------------------ #
    def _df(filas):
        if not filas:
            return pd.DataFrame()
        df = pd.DataFrame(filas)
        for c in ["MAE (°C)", "RMSE (°C)", "MAPE (%)", "R²"]:
            df[c] = df[c].round(4)
        return df

    tablas = {
        "test_general":    _df(test_gen),
        "test_estaciones": _df(test_est),
        "marzo_general":   _df(marzo_gen),
        "marzo_estaciones": _df(marzo_est),
    }

    for nombre, df in tablas.items():
        df.to_csv(os.path.join(args.salida, f"tabla_{nombre}.csv"), index=False)

    xlsx_path = os.path.join(args.salida, "resultados_modelos.xlsx")
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xw:
            hoja = {"test_general": "TEST general", "test_estaciones": "TEST por estación",
                    "marzo_general": f"{args.mes} general", "marzo_estaciones": f"{args.mes} por estación"}
            for nombre, df in tablas.items():
                (df if not df.empty else pd.DataFrame({"info": ["sin datos"]})
                 ).to_excel(xw, sheet_name=hoja[nombre][:31], index=False)
        print(f"[OK] Excel: {xlsx_path}")
    except Exception as e:
        print(f"[AVISO] No se pudo escribir xlsx ({e}); quedan los CSV.")

    # ------------------------------------------------------------------ #
    # impresión
    # ------------------------------------------------------------------ #
    pd.set_option("display.width", 160, "display.max_rows", 100)
    print("\n" + "=" * 70 + "\n TEST — GENERAL\n" + "=" * 70)
    print(tablas["test_general"].to_string(index=False))
    print("\n" + "=" * 70 + f"\n {args.mes} — GENERAL\n" + "=" * 70)
    print(tablas["marzo_general"].to_string(index=False) if not tablas["marzo_general"].empty
          else "  (faltan .npy de predicción)")
    print("\n" + "=" * 70 + "\n TEST — POR ESTACIÓN\n" + "=" * 70)
    print(tablas["test_estaciones"].to_string(index=False))
    print("\n" + "=" * 70 + f"\n {args.mes} — POR ESTACIÓN\n" + "=" * 70)
    print(tablas["marzo_estaciones"].to_string(index=False) if not tablas["marzo_estaciones"].empty
          else "  (faltan .npy de predicción)")

    if faltan_preds:
        print("\n[AVISO] Faltaron algunos .npy de predicción; las tablas de marzo "
              "pueden estar incompletas.")
    print(f"\n[OK] CSVs y Excel en: {os.path.abspath(args.salida)}")


if __name__ == "__main__":
    main()
