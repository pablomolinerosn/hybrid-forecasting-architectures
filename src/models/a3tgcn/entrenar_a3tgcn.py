#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
 entrenar_a3tgcn.py
----------------------------------------------------------------------------
 Entrena y evalúa A3T-GCN sobre la base canónica, en los 3 escenarios:
   corto (h=3, seq=24) | medio (h=48, seq=96) | largo (h=72, seq=168)

 A3T-GCN (torch_geometric_temporal) recibe por muestra un tensor
 (num_nodos, in_channels, periods) y aplica atención temporal sobre los
 'periods' (= seq_len). Se entrena con batching a nivel de grafo (grafo
 disjunto de B copias) para acelerar, replicando tu optimización previa.

 Métricas por escenario y por estación: MAE, RMSE, R2, MAPE (en °C).

 Uso (desde la raiz del repo):
   python src/models/a3tgcn/entrenar_a3tgcn.py --artefactos artefactos --salida resultados/a3tgcn
   python src/models/a3tgcn/entrenar_a3tgcn.py --escenarios corto medio --epochs 60 --seed 42
============================================================================
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch_geometric_temporal.nn.recurrent import A3TGCN

# common.py vive en el directorio padre (src/models/), compartido con
# entrenar_dcrnn.py -- se agrega explicitamente al path porque ya no es
# un sibling directo de este script (src/models/a3tgcn/).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common as C


# ---------------------------------------------------------------------------
# MODELO
# ---------------------------------------------------------------------------
class ModeloA3TGCN(nn.Module):
    """
    forward(xb) con xb=(B, N, L, F) -> (B, N) [target escalado].
    Internamente arma el grafo por lote y aplica A3TGCN + capa lineal.
    """

    def __init__(self, in_channels, hidden, periods, edge_index, edge_weight):
        super().__init__()
        self.recurrent = A3TGCN(in_channels=in_channels,
                                out_channels=hidden,
                                periods=periods)
        self.head = nn.Sequential(nn.ReLU(), nn.Linear(hidden, 1))
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_weight", edge_weight)

    def forward(self, xb):
        B, N, L, Fd = xb.shape
        # A3TGCN espera (nodos, in_channels, periods) = (B*N, F, L)
        x = xb.reshape(B * N, L, Fd).permute(0, 2, 1).contiguous()
        ei, ew = C.construir_grafo_batch(self.edge_index, self.edge_weight, B, N)
        h = self.recurrent(x, ei, ew)          # (B*N, hidden)
        out = self.head(h).view(B, N)          # (B, N)
        return out


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Entrenamiento A3T-GCN")
    ap.add_argument("--artefactos", default="artefactos")
    ap.add_argument("--salida", default="resultados/a3tgcn")
    ap.add_argument("--escenarios", nargs="+",
                    default=["corto", "medio", "largo"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--hidden", type=int, default=64)
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
        print(f" A3T-GCN | escenario '{esc}' | horizon={h['horizon']}h "
              f"seq_len={h['seq_len']} batch={h['batch']}")
        print("=" * 72)

        loaders, tam = C.construir_loaders(art, h["seq_len"], h["horizon"],
                                           h["batch"])
        print(f"Ventanas -> train {tam['train']:,} | val {tam['val']:,} "
              f"| test {tam['test']:,}")

        modelo = ModeloA3TGCN(in_channels=art["F"], hidden=args.hidden,
                              periods=h["seq_len"],
                              edge_index=edge_index, edge_weight=edge_weight)

        etiqueta = f"a3tgcn_{esc}_h{h['horizon']}"
        met = C.entrenar_y_evaluar(modelo, loaders, estaciones,
                                   art["target_scaler"], cfg, device,
                                   args.salida, etiqueta)
        met["config"] = {"modelo": "A3T-GCN", "escenario": esc, **h,
                         "hidden": args.hidden, "seed": args.seed}
        resultados[esc] = met

    with open(os.path.join(args.salida, "metricas_a3tgcn.json"), "w",
              encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    # tabla resumen
    print("\n" + "=" * 72)
    print(" RESUMEN A3T-GCN (TEST, °C)")
    print("=" * 72)
    print(f"{'escenario':<10}{'MAE':>10}{'RMSE':>10}{'R2':>10}{'MAPE%':>10}")
    for esc, met in resultados.items():
        g = met["global"]
        print(f"{esc:<10}{g['MAE']:>10.4f}{g['RMSE']:>10.4f}"
              f"{g['R2']:>10.4f}{g['MAPE']:>10.2f}")
    print(f"\n[OK] Resultados en: {os.path.abspath(args.salida)}")


if __name__ == "__main__":
    main()
