#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
 entrenar_dcrnn.py
----------------------------------------------------------------------------
 Entrena y evalúa DCRNN sobre la MISMA base canónica, en los 3 escenarios:
   corto (h=3, seq=24) | medio (h=48, seq=96) | largo (h=72, seq=168)

 DCRNN (torch_geometric_temporal) procesa UN paso temporal por llamada y
 arrastra el estado oculto H. Aquí se usa como encoder recurrente: se
 recorre la secuencia (seq_len pasos) y el H final se proyecta al target
 del horizonte (predicción directa de un valor por nodo). El grafo se arma
 por lote una sola vez y se reutiliza en todos los pasos temporales.

 Métricas por escenario y por estación: MAE, RMSE, R2, MAPE (en °C).

 Uso (desde la raiz del repo):
   python src/models/dcrnn/entrenar_dcrnn.py --artefactos artefactos --salida resultados/dcrnn
   python src/models/dcrnn/entrenar_dcrnn.py --escenarios largo --epochs 60 --seed 42
============================================================================
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch_geometric_temporal.nn.recurrent import DCRNN

# common.py vive en el directorio padre (src/models/), compartido con
# entrenar_a3tgcn.py -- se agrega explicitamente al path porque ya no es
# un sibling directo de este script (src/models/dcrnn/).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as C


# ---------------------------------------------------------------------------
# MODELO
# ---------------------------------------------------------------------------
class ModeloDCRNN(nn.Module):
    """
    forward(xb) con xb=(B, N, L, F) -> (B, N) [target escalado].
    Recorre los L pasos con DCRNN arrastrando H y proyecta el H final.
    """

    def __init__(self, in_channels, hidden, K, edge_index, edge_weight):
        super().__init__()
        self.recurrent = DCRNN(in_channels=in_channels,
                               out_channels=hidden, K=K)
        self.head = nn.Sequential(nn.ReLU(), nn.Linear(hidden, 1))
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_weight", edge_weight)

    def forward(self, xb):
        B, N, L, Fd = xb.shape
        x = xb.reshape(B * N, L, Fd)                 # (B*N, L, F)
        ei, ew = C.construir_grafo_batch(self.edge_index, self.edge_weight, B, N)
        H = None
        for t in range(L):
            H = self.recurrent(x[:, t, :], ei, ew, H)  # (B*N, hidden)
        out = self.head(H).view(B, N)                # (B, N)
        return out


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Entrenamiento DCRNN")
    ap.add_argument("--artefactos", default="artefactos")
    ap.add_argument("--salida", default="resultados/dcrnn")
    ap.add_argument("--escenarios", nargs="+",
                    default=["corto", "medio", "largo"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--K", type=int, default=2, help="Orden de difusión (Chebyshev)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    C.set_seed(args.seed)
    os.makedirs(args.salida, exist_ok=True)
    device = torch.device(args.device)
    print(f"Dispositivo: {device} | seed: {args.seed}")

    art = C.cargar_artefactos(args.artefactos)
    estaciones = art["station_order"]
    edge_index = torch.as_tensor(art["edge_index"], dtype=torch.long)
    edge_weight = torch.as_tensor(art["edge_weight"], dtype=torch.float32)
    print(f"Tensor X: {art['X'].shape} | N={art['N']} | F={art['F']} "
          f"| target_idx={art['target_idx']}")

    cfg = {"lr": args.lr, "weight_decay": args.weight_decay,
           "epochs": args.epochs, "patience": args.patience,
           "grad_clip": args.grad_clip}

    resultados = {}
    for esc in args.escenarios:
        h = C.HORIZONTES[esc]
        print("\n" + "=" * 72)
        print(f" DCRNN | escenario '{esc}' | horizon={h['horizon']}h "
              f"seq_len={h['seq_len']} batch={h['batch']}")
        print("=" * 72)

        loaders, tam = C.construir_loaders(art, h["seq_len"], h["horizon"],
                                           h["batch"])
        print(f"Ventanas -> train {tam['train']:,} | val {tam['val']:,} "
              f"| test {tam['test']:,}")

        modelo = ModeloDCRNN(in_channels=art["F"], hidden=args.hidden, K=args.K,
                             edge_index=edge_index, edge_weight=edge_weight)

        etiqueta = f"dcrnn_{esc}_h{h['horizon']}"
        met = C.entrenar_y_evaluar(modelo, loaders, estaciones,
                                   art["target_scaler"], cfg, device,
                                   args.salida, etiqueta)
        met["config"] = {"modelo": "DCRNN", "escenario": esc, **h,
                         "hidden": args.hidden, "K": args.K, "seed": args.seed}
        resultados[esc] = met

    with open(os.path.join(args.salida, "metricas_dcrnn.json"), "w",
              encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 72)
    print(" RESUMEN DCRNN (TEST, °C)")
    print("=" * 72)
    print(f"{'escenario':<10}{'MAE':>10}{'RMSE':>10}{'R2':>10}{'MAPE%':>10}")
    for esc, met in resultados.items():
        g = met["global"]
        print(f"{esc:<10}{g['MAE']:>10.4f}{g['RMSE']:>10.4f}"
              f"{g['R2']:>10.4f}{g['MAPE']:>10.2f}")
    print(f"\n[OK] Resultados en: {os.path.abspath(args.salida)}")


if __name__ == "__main__":
    main()
