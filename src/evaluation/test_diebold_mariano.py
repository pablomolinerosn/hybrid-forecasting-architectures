#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
 test_diebold_mariano.py
----------------------------------------------------------------------------
 Test de Diebold-Mariano (DM) entre A3T-GCN y DCRNN usando las predicciones
 de TEST ya guardadas por el pipeline (una sola semilla). No reentrena nada.

 Qué hace:
   Para cada escenario (corto/medio/largo) compara la precisión de ambos
   modelos sobre la MISMA serie de test y decide si la diferencia es
   estadísticamente significativa o si los modelos son equivalentes.

 Detalles metodológicos:
   - Diferencial de pérdida por instante:  d_t = L(e_A3T) - L(e_DCRNN)
       con L = error absoluto ('ae') o cuadrático ('se').
       Convención: dbar > 0  =>  A3T-GCN tiene más pérdida  =>  DCRNN mejor.
   - Varianza de largo plazo con autocovarianzas hasta el lag (h-1), que es
     la truncación teórica para pronósticos a h pasos (errores ~ MA(h-1)).
     Si la estimación rectangular resultara no positiva, se usa kernel de
     Bartlett (Newey-West), que garantiza positividad.
   - Corrección de muestra pequeña de Harvey-Leybourne-Newbold (1997) y
     referencia t de Student con (T-1) g.l. (con T ~ 19k, t ≈ Normal).

 Entradas (arrays que ya produce common.py):
   {a3t}/a3tgcn_{esc}_h{H}_pred_test.npy  y  _true_test.npy
   {dcrnn}/dcrnn_{esc}_h{H}_pred_test.npy y  _true_test.npy
   (H se lee de metricas_*.json)

 Uso:
   python test_diebold_mariano.py --a3t resultados/a3tgcn --dcrnn resultados/dcrnn
   python test_diebold_mariano.py --loss se --alpha 0.05
============================================================================
"""

import argparse
import json
import math
import os

import numpy as np

ESCENARIOS = ["corto", "medio", "largo"]
M1, M2 = "A3T-GCN", "DCRNN"          # modelo 1 y modelo 2 (orden del diferencial)


# ---------------------------------------------------------------------------
# ESTADÍSTICA
# ---------------------------------------------------------------------------
def normal_sf(x):
    """1 - Phi(x) sin dependencias (via erfc)."""
    return 0.5 * math.erfc(x / math.sqrt(2))


def p_valor_dos_colas(stat, df):
    """p-valor de dos colas; usa t de Student si hay scipy, si no Normal."""
    try:
        from scipy import stats
        return float(2 * stats.t.sf(abs(stat), df=df))
    except Exception:
        return float(2 * normal_sf(abs(stat)))


def _varianza_largo_plazo(d, h):
    """Var de largo plazo de d_t con lags hasta h-1 (rectangular; Bartlett si <=0)."""
    dc = d - d.mean()
    gamma0 = float(np.mean(dc * dc))
    K = max(int(h) - 1, 0)
    gammas = [float(np.mean(dc[k:] * dc[:-k])) for k in range(1, K + 1)]
    lrv = gamma0 + 2.0 * sum(gammas)
    pesos = "rectangular"
    if lrv <= 0 and K > 0:                       # fallback que garantiza positividad
        lrv = gamma0 + 2.0 * sum((1 - k / (K + 1)) * g
                                 for k, g in zip(range(1, K + 1), gammas))
        pesos = "bartlett"
    return lrv, K, pesos


def dm_desde_diferencial(d, h):
    """Test DM sobre una serie 1-D de diferencial de pérdida d_t."""
    d = np.asarray(d, dtype=float)
    T = d.size
    dbar = float(d.mean())
    lrv, K, pesos = _varianza_largo_plazo(d, h)
    var_dbar = lrv / T
    if var_dbar <= 0:
        return {"error": "varianza no positiva", "T": int(T), "dbar": dbar}
    dm = dbar / math.sqrt(var_dbar)
    hln = math.sqrt(max((T + 1 - 2 * h + h * (h - 1) / T) / T, 1e-12))
    dm_corr = dm * hln
    return {
        "T": int(T), "dbar": dbar,
        "dm_stat": float(dm_corr), "dm_sin_correccion": float(dm),
        "p_value": p_valor_dos_colas(dm_corr, df=T - 1),
        "lags": int(K), "pesos": pesos, "hln_factor": float(hln),
    }


def perdida(e, loss):
    return e ** 2 if loss == "se" else np.abs(e)


def veredicto(res, alpha):
    if "error" in res:
        return "indeterminado"
    if res["p_value"] < alpha:
        return f"{M2} mejor" if res["dbar"] > 0 else f"{M1} mejor"
    return "equivalentes"


# ---------------------------------------------------------------------------
# CARGA / ALINEACIÓN
# ---------------------------------------------------------------------------
def cargar_par(a3t_dir, dcrnn_dir, esc, met_a, met_d):
    hA = met_a[esc]["config"]["horizon"]
    hD = met_d[esc]["config"]["horizon"]
    if hA != hD:
        raise ValueError(f"[{esc}] horizontes distintos entre modelos: {hA} vs {hD}")
    tagA = f"a3tgcn_{esc}_h{hA}"
    tagD = f"dcrnn_{esc}_h{hD}"
    pa = np.load(os.path.join(a3t_dir, f"{tagA}_pred_test.npy"))
    ta = np.load(os.path.join(a3t_dir, f"{tagA}_true_test.npy"))
    pd_ = np.load(os.path.join(dcrnn_dir, f"{tagD}_pred_test.npy"))
    td_ = np.load(os.path.join(dcrnn_dir, f"{tagD}_true_test.npy"))

    if pa.shape != pd_.shape:
        raise ValueError(f"[{esc}] formas de predicción distintas: {pa.shape} vs {pd_.shape}")
    if not np.allclose(ta, td_, atol=1e-3):
        raise ValueError(f"[{esc}] las series observadas no coinciden entre modelos "
                         f"(¿distinto seq_len/horizon o split?)")
    return pa, pd_, ta, hA


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Diebold-Mariano A3T-GCN vs DCRNN")
    ap.add_argument("--a3t", default="resultados/a3tgcn")
    ap.add_argument("--dcrnn", default="resultados/dcrnn")
    ap.add_argument("--estaciones", default="artefactos/station_order.json")
    ap.add_argument("--loss", default="ae", choices=["ae", "se"],
                    help="Función de pérdida principal: ae=|error|, se=error^2")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--salida", default="resultados/diebold_mariano")
    args = ap.parse_args()

    os.makedirs(args.salida, exist_ok=True)
    met_a = json.load(open(os.path.join(args.a3t, "metricas_a3tgcn.json")))
    met_d = json.load(open(os.path.join(args.dcrnn, "metricas_dcrnn.json")))
    estaciones = json.load(open(args.estaciones))

    print("=" * 76)
    print(" TEST DE DIEBOLD-MARIANO  |  A3T-GCN (modelo 1)  vs  DCRNN (modelo 2)")
    print(f" Pérdida principal: {'|error|' if args.loss=='ae' else 'error^2'} "
          f"| alpha={args.alpha}")
    print(" Convención: dbar>0 => A3T-GCN peor => DCRNN mejor")
    print("=" * 76)

    salida = {}
    resumen = []
    for esc in ESCENARIOS:
        pa, pd_, ta, h = cargar_par(args.a3t, args.dcrnn, esc, met_a, met_d)
        eA = pa - ta                      # (S, N)
        eD = pd_ - ta
        maeA = float(np.mean(np.abs(eA)))
        maeD = float(np.mean(np.abs(eD)))

        # --- global: diferencial de pérdida promediado en el espacio ---
        LA = perdida(eA, args.loss); LD = perdida(eD, args.loss)
        d_global = (LA - LD).mean(axis=1)            # (S,)
        res_g = dm_desde_diferencial(d_global, h)
        vg = veredicto(res_g, args.alpha)

        # --- por estación ---
        por_est = {}
        for j, est in enumerate(estaciones):
            dj = perdida(eA[:, j], args.loss) - perdida(eD[:, j], args.loss)
            rj = dm_desde_diferencial(dj, h)
            rj["veredicto"] = veredicto(rj, args.alpha)
            rj["mae_A3T"] = float(np.mean(np.abs(eA[:, j])))
            rj["mae_DCRNN"] = float(np.mean(np.abs(eD[:, j])))
            por_est[est] = rj

        salida[esc] = {"horizon": h, "mae_global": {M1: maeA, M2: maeD},
                       "global": {**res_g, "veredicto": vg},
                       "por_estacion": por_est}
        resumen.append({"escenario": esc, "h": h, "mae_A3T": maeA, "mae_DCRNN": maeD,
                        "dm_stat": res_g.get("dm_stat"), "p_value": res_g.get("p_value"),
                        "veredicto": vg})

        # --- impresión ---
        print(f"\n### {esc.upper()}  (h={h})  lags={res_g.get('lags')} "
              f"pesos={res_g.get('pesos')}")
        print(f"  MAE global: {M1} {maeA:.4f} °C | {M2} {maeD:.4f} °C")
        print(f"  GLOBAL (promedio espacial): dbar={res_g['dbar']:+.5f} "
              f"DM={res_g['dm_stat']:+.3f} p={res_g['p_value']:.3g} -> {vg}")
        print(f"  {'estación':<12}{'DM':>9}{'p-valor':>12}   veredicto")
        for est in estaciones:
            r = por_est[est]
            print(f"  {est:<12}{r['dm_stat']:>9.3f}{r['p_value']:>12.3g}   {r['veredicto']}")

    # --- guardar ---
    with open(os.path.join(args.salida, "dm_resultados.json"), "w", encoding="utf-8") as f:
        json.dump(salida, f, ensure_ascii=False, indent=2)
    cols = ["escenario", "h", "mae_A3T", "mae_DCRNN", "dm_stat", "p_value", "veredicto"]
    with open(os.path.join(args.salida, "dm_resumen.csv"), "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in resumen:
            f.write(",".join(str(r[c]) for c in cols) + "\n")

    print("\n" + "=" * 76)
    print(" RESUMEN")
    print(f" {'escenario':<9}{'h':>4}{'MAE A3T':>10}{'MAE DCRNN':>11}"
          f"{'DM':>9}{'p-valor':>11}   veredicto")
    for r in resumen:
        print(f" {r['escenario']:<9}{r['h']:>4}{r['mae_A3T']:>10.4f}{r['mae_DCRNN']:>11.4f}"
              f"{r['dm_stat']:>9.3f}{r['p_value']:>11.3g}   {r['veredicto']}")
    print(f"\n[OK] Resultados en: {os.path.abspath(args.salida)}")
    print(" Nota: con T ~ 19k, la t(T-1) coincide con la Normal; la corrección")
    print("       HLN es ~1. La significancia proviene del tamaño y estructura")
    print("       de autocorrelación de los errores reales.")


if __name__ == "__main__":
    main()
