#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_data.py
----------------------------------------------------------------------------
Script local (no se corre dentro de Docker) que materializa en app/model_data/
el subconjunto mínimo de artefactos + checkpoints necesarios para servir el
modelo DCRNN en el dashboard, en sus 3 escenarios (corto h=3h, medio h=48h,
largo h=72h).

Recorta artefactos/X.npy y artefactos/tiempos.npy al split de test (para no
arrastrar 18 años de historia ni el checkpoint completo del repo del TFM) y
copia el resto de artefactos livianos tal cual.

Uso:
  python app/prepare_data.py
"""

import json
import os
import shutil

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTEFACTOS = os.path.join(RAIZ, "artefactos")
RESULTADOS_DCRNN = os.path.join(RAIZ, "resultados", "dcrnn")
SALIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_data")


def main():
    os.makedirs(SALIDA, exist_ok=True)

    with open(os.path.join(ARTEFACTOS, "splits.json"), "r", encoding="utf-8") as f:
        splits = json.load(f)
    a, b = splits["test"]

    X = np.load(os.path.join(ARTEFACTOS, "X.npy"))
    tiempos = np.load(os.path.join(ARTEFACTOS, "tiempos.npy"), allow_pickle=True)

    X_test = X[a:b].astype(np.float32)
    tiempos_test = tiempos[a:b]

    np.save(os.path.join(SALIDA, "X_test.npy"), X_test)
    np.save(os.path.join(SALIDA, "tiempos_test.npy"), tiempos_test)
    print(f"X_test: {X_test.shape} | tiempos_test: {tiempos_test.shape} "
          f"({tiempos_test[0]} -> {tiempos_test[-1]})")

    for nombre in ("edge_index.npy", "edge_weight.npy"):
        shutil.copy(os.path.join(ARTEFACTOS, nombre), os.path.join(SALIDA, nombre))

    for nombre in ("target_scaler.json", "feature_order.json", "station_order.json"):
        shutil.copy(os.path.join(ARTEFACTOS, nombre), os.path.join(SALIDA, nombre))

    shutil.copy(
        os.path.join(ARTEFACTOS, "estaciones_coords_usadas.csv"),
        os.path.join(SALIDA, "estaciones_coords.csv"),
    )

    for nombre in (
        "dcrnn_corto_h3_best.pt",
        "dcrnn_medio_h48_best.pt",
        "dcrnn_largo_h72_best.pt",
    ):
        shutil.copy(
            os.path.join(RESULTADOS_DCRNN, nombre),
            os.path.join(SALIDA, nombre),
        )

    shutil.copy(
        os.path.join(RESULTADOS_DCRNN, "metricas_dcrnn.json"),
        os.path.join(SALIDA, "metricas_dcrnn.json"),
    )

    print(f"Artefactos escritos en {SALIDA}")
    for nombre in sorted(os.listdir(SALIDA)):
        ruta = os.path.join(SALIDA, nombre)
        print(f"  {nombre:<28} {os.path.getsize(ruta) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
