#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
 preprocesamiento_canonico.py
----------------------------------------------------------------------------
 Produce la BASE ÚNICA que consumen A3T-GCN, DCRNN y (más adelante) los
 baselines. Un solo preprocesador -> comparación justa entre modelos.

 Hace:
   1. Elimina variables descartadas y fija el manifiesto de 14 features
      (8 climáticas + 6 cíclicas). Target = temperaturaMedia (idx 0).
   2. Ordena las 6 estaciones de forma fija (alfabética) -> mismo eje N que
      la matriz de adyacencia.
   3. Arma el tensor (T, N, F).
   4. Splits temporales: TRAIN 2008-2021 | VAL 2022-2023 | TEST 2024-2026.
   5. StandardScaler ajustado SOLO en train, aplicado a las 8 climáticas.
      Las 6 cíclicas se dejan intactas (se rompería su geometría circular).
   6. Reconstruye la adyacencia Haversine + kernel gaussiano (sigma = media
      de distancias, totalmente conectado) con el orden de estaciones fijo.

 Artefactos (en --salida):
   X.npy, tiempos.npy, feature_order.json, station_order.json,
   scaler.json, target_scaler.json, splits.json,
   adyacencia.npy, edge_index.npy, edge_weight.npy, distancias_km.npy,
   estaciones_coords_usadas.csv, resumen_preprocesamiento.json
============================================================================
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# MANIFIESTO DE FEATURES  (target primero -> TARGET_IDX = 0)
# ---------------------------------------------------------------------------
CLIMATICAS = [
    "temperaturaMedia",       # <- TARGET (idx 0)
    "humedadRelativa",
    "presionBarometrica",
    "radiacionSolar",
    "viento_u",
    "viento_v",
    "precipitacion_log",
    "radiacionSolar_lag1",
]
CICLICAS = [
    "hora_sin", "hora_cos",
    "mes_sin", "mes_cos",
    "dia_semana_sin", "dia_semana_cos",
]
FEATURE_ORDER = CLIMATICAS + CICLICAS          # 14 en total
TARGET = "temperaturaMedia"
N_CLIM = len(CLIMATICAS)                        # 8 -> índices 0..7 se escalan

# Estaciones en ORDEN FIJO (alfabético). Debe coincidir con el eje N de A.
ESTACIONES = ["Belisario", "Carapungo", "Cotocollao",
              "ElCamal", "LosChillos", "Tumbaco"]

COL_TIEMPO = "fechaHora"
COL_ESTACION = "estacion"

# Splits temporales (por año)
SPLIT_TRAIN = (2008, 2021)
SPLIT_VAL = (2022, 2023)
SPLIT_TEST = (2024, 2026)

# ---------------------------------------------------------------------------
# COORDENADAS DE ESTACIONES  (grados decimales)
# ---------------------------------------------------------------------------
# IMPORTANTE: estas coordenadas son APROXIMADAS. Para reproducir exactamente
# tu grafo previo (sigma = 12.8 km), coloca un CSV 'estaciones_coords.csv'
# (columnas: estacion,lat,lon) junto al parquet o pásalo con --coords.
# Si no se encuentra, se usan estas y se emite una ADVERTENCIA.
COORDS_APROX = {
    "Belisario":  (-0.185170, -78.495677),
    "Carapungo":  (-0.095393, -78.449755),
    "Cotocollao": (-0.107716, -78.497268),
    "ElCamal":    (-0.249970, -78.510058),
    "LosChillos": (-0.297100, -78.455270),
    "Tumbaco":    (-0.215015, -78.403442),
}


def log(m=""):
    print(m, flush=True)


# ---------------------------------------------------------------------------
# 1-3. TENSOR (T, N, F)
# ---------------------------------------------------------------------------
def construir_tensor(df):
    faltan = [c for c in FEATURE_ORDER + [COL_TIEMPO, COL_ESTACION]
              if c not in df.columns]
    if faltan:
        raise ValueError(f"Faltan columnas requeridas en el parquet: {faltan}")

    df = df.copy()
    df[COL_TIEMPO] = pd.to_datetime(df[COL_TIEMPO])
    df[COL_ESTACION] = df[COL_ESTACION].astype(str)

    est_presentes = sorted(df[COL_ESTACION].unique())
    if est_presentes != sorted(ESTACIONES):
        log(f"[ADVERTENCIA] Estaciones en base {est_presentes} != {sorted(ESTACIONES)}")

    tiempos = np.sort(df[COL_TIEMPO].unique())
    T, N, F = len(tiempos), len(ESTACIONES), len(FEATURE_ORDER)
    log(f"Construyendo tensor (T={T:,}, N={N}, F={F}) ...")

    X = np.empty((T, N, F), dtype=np.float32)
    for fi, feat in enumerate(FEATURE_ORDER):
        piv = (df.pivot(index=COL_TIEMPO, columns=COL_ESTACION, values=feat)
                 .reindex(index=tiempos, columns=ESTACIONES))
        if piv.isna().any().any():
            raise ValueError(f"NaN tras pivotar la feature '{feat}' "
                             f"(¿faltan pares tiempo-estación?)")
        X[:, :, fi] = piv.values.astype(np.float32)

    return X, tiempos


# ---------------------------------------------------------------------------
# 4. SPLITS
# ---------------------------------------------------------------------------
def calcular_splits(tiempos):
    years = pd.DatetimeIndex(tiempos).year.values
    def rango(a, b):
        idx = np.where((years >= a) & (years <= b))[0]
        return int(idx.min()), int(idx.max()) + 1        # [a, b) exclusivo
    splits = {
        "train": list(rango(*SPLIT_TRAIN)),
        "val":   list(rango(*SPLIT_VAL)),
        "test":  list(rango(*SPLIT_TEST)),
    }
    # verificación de contigüidad y ausencia de solapamiento
    assert splits["train"][1] == splits["val"][0], "splits no contiguos train/val"
    assert splits["val"][1] == splits["test"][0], "splits no contiguos val/test"
    return splits


# ---------------------------------------------------------------------------
# 5. STANDARD SCALER (solo train, solo climáticas)
# ---------------------------------------------------------------------------
def escalar(X, splits):
    a, b = splits["train"]
    Xtr = X[a:b, :, :N_CLIM]                              # (Ttr, N, 8)
    mean = Xtr.reshape(-1, N_CLIM).mean(axis=0)
    std = Xtr.reshape(-1, N_CLIM).std(axis=0)
    std[std == 0] = 1.0

    Xs = X.copy()
    Xs[:, :, :N_CLIM] = (X[:, :, :N_CLIM] - mean) / std   # climáticas escaladas
    # cíclicas (idx 8..13) quedan intactas

    scaler = {
        "features": CLIMATICAS,
        "mean": mean.tolist(),
        "scale": std.tolist(),
    }
    target_scaler = {
        "idx": FEATURE_ORDER.index(TARGET),               # 0
        "mean": float(mean[0]),
        "scale": float(std[0]),
    }
    return Xs.astype(np.float32), scaler, target_scaler


# ---------------------------------------------------------------------------
# 6. ADYACENCIA  (Haversine + kernel gaussiano)
# ---------------------------------------------------------------------------
def cargar_coords(ruta_coords):
    if ruta_coords and os.path.exists(ruta_coords):
        dfc = pd.read_csv(ruta_coords)
        dfc[COL_ESTACION if COL_ESTACION in dfc.columns else "estacion"]
        dfc = dfc.rename(columns={c: c.lower() for c in dfc.columns})
        m = {str(r["estacion"]): (float(r["lat"]), float(r["lon"]))
             for _, r in dfc.iterrows()}
        faltan = [e for e in ESTACIONES if e not in m]
        if faltan:
            raise ValueError(f"Coords faltantes para {faltan} en {ruta_coords}")
        log(f"Coordenadas leídas de: {ruta_coords}")
        return {e: m[e] for e in ESTACIONES}, True
    log("[ADVERTENCIA] No se encontró CSV de coordenadas; se usan APROXIMADAS.")
    log("               El grafo NO coincidirá con tu adyacencia previa hasta")
    log("               proveer coordenadas oficiales REMMAQ (--coords).")
    return {e: COORDS_APROX[e] for e in ESTACIONES}, False


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def construir_adyacencia(coords):
    N = len(ESTACIONES)
    lat = np.array([coords[e][0] for e in ESTACIONES])
    lon = np.array([coords[e][1] for e in ESTACIONES])

    D = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            if i != j:
                D[i, j] = haversine_km(lat[i], lon[i], lat[j], lon[j])

    off = D[~np.eye(N, dtype=bool)]
    sigma = float(off.mean())                             # media de distancias
    log(f"Sigma (media de distancias) = {sigma:.3f} km "
        f"(referencia proyecto previo: ~12.8 km)")

    # kernel gaussiano; diagonal = 1 (auto-similitud). Threshold 0 => conectado.
    W = np.exp(-(D ** 2) / (sigma ** 2))
    np.fill_diagonal(W, 1.0)

    # edge_index / edge_weight SIN self-loops (las capas GCN los añaden solas)
    ei, ew = [], []
    for i in range(N):
        for j in range(N):
            if i != j:
                ei.append((i, j))
                ew.append(W[i, j])
    edge_index = np.array(ei, dtype=np.int64).T           # (2, E)
    edge_weight = np.array(ew, dtype=np.float32)          # (E,)
    return W.astype(np.float32), edge_index, edge_weight, D.astype(np.float32), sigma


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Preprocesamiento canónico STGNN-DMQ")
    ap.add_argument("--parquet", required=True, help="Ruta al parquet limpio")
    ap.add_argument("--salida", default="artefactos", help="Directorio de salida")
    ap.add_argument("--coords", default=None,
                    help="CSV opcional estacion,lat,lon (coords oficiales REMMAQ)")
    args = ap.parse_args()

    os.makedirs(args.salida, exist_ok=True)
    log("=" * 72)
    log(" PREPROCESAMIENTO CANÓNICO")
    log("=" * 72)

    df = pd.read_parquet(args.parquet)
    log(f"Parquet leído: {df.shape[0]:,} filas, {df.shape[1]} columnas")
    log(f"Features conservadas ({len(FEATURE_ORDER)}): {FEATURE_ORDER}")

    X, tiempos = construir_tensor(df)
    splits = calcular_splits(tiempos)
    log(f"Splits (índices sobre T): {splits}")
    for k, (a, b) in splits.items():
        log(f"  {k:<6} {pd.Timestamp(tiempos[a])} -> "
            f"{pd.Timestamp(tiempos[b-1])}  ({b-a:,} pasos)")

    Xs, scaler, target_scaler = escalar(X, splits)
    log(f"StandardScaler ajustado en train sobre {N_CLIM} climáticas.")
    log(f"  target mean={target_scaler['mean']:.4f}  "
        f"scale={target_scaler['scale']:.4f}")

    # sanity: media/std ~0/1 de climáticas en train tras escalar
    a, b = splits["train"]
    chk = Xs[a:b, :, :N_CLIM].reshape(-1, N_CLIM)
    log(f"  chequeo train escalado -> mean~{chk.mean(axis=0).round(3).tolist()}")

    coords, coords_ok = cargar_coords(args.coords or
                                      os.path.join(os.path.dirname(args.parquet),
                                                   "estaciones_coords.csv"))
    W, edge_index, edge_weight, D, sigma = construir_adyacencia(coords)
    log("Matriz de adyacencia W (redondeada):")
    log(np.array2string(W, precision=3, suppress_small=True))

    # -------------------- guardar artefactos --------------------
    np.save(os.path.join(args.salida, "X.npy"), Xs)
    np.save(os.path.join(args.salida, "tiempos.npy"), tiempos)
    np.save(os.path.join(args.salida, "adyacencia.npy"), W)
    np.save(os.path.join(args.salida, "edge_index.npy"), edge_index)
    np.save(os.path.join(args.salida, "edge_weight.npy"), edge_weight)
    np.save(os.path.join(args.salida, "distancias_km.npy"), D)

    def _dump(nombre, obj):
        with open(os.path.join(args.salida, nombre), "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    _dump("feature_order.json", FEATURE_ORDER)
    _dump("station_order.json", ESTACIONES)
    _dump("scaler.json", scaler)
    _dump("target_scaler.json", target_scaler)
    _dump("splits.json", splits)

    pd.DataFrame(
        [{"estacion": e, "lat": coords[e][0], "lon": coords[e][1]}
         for e in ESTACIONES]
    ).to_csv(os.path.join(args.salida, "estaciones_coords_usadas.csv"), index=False)

    _dump("resumen_preprocesamiento.json", {
        "n_filas_parquet": int(df.shape[0]),
        "tensor_shape": list(Xs.shape),
        "feature_order": FEATURE_ORDER,
        "target": TARGET,
        "target_idx": target_scaler["idx"],
        "station_order": ESTACIONES,
        "splits_idx": splits,
        "sigma_km": sigma,
        "coords_oficiales": coords_ok,
        "descartadas": ["hora", "mes", "dia_semana", "anio",
                        "precipitacion", "radiacionSolar_lag2", "fecha"],
    })

    log("\n[OK] Artefactos guardados en: " + os.path.abspath(args.salida))
    log("     -> ahora ejecuta entrenar_a3tgcn.py y entrenar_dcrnn.py")


if __name__ == "__main__":
    main()