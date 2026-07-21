"""
Carga de artefactos, ventaneo e inferencia para el dashboard de alerta
temprana DCRNN.

Replica la convención de ventaneo de src/models/common.py::VentanaDataset
(x = X[i:i+seq_len], y = X[i+seq_len+horizon-1, :, target_idx]) pero indexada
por la "hora actual" j = último paso observado (j = i + seq_len - 1), que es
el concepto que se le muestra al usuario en el selector de la UI.

Sirve los 3 escenarios entrenados (mismos que src/models/common.py::HORIZONTES):
corto (h=3h, ventana 24h), medio (h=48h, ventana 96h) y largo (h=72h,
ventana 168h). Los 3 comparten arquitectura (hidden=64, K=2), grafo y
scaler — solo cambian seq_len/horizon y el checkpoint entrenado.
"""

import json
import os

import numpy as np
import pandas as pd
import torch

from model import ModeloDCRNN

ESCENARIOS = {
    "corto": {"horizon": 3, "seq_len": 24},
    "medio": {"horizon": 48, "seq_len": 96},
    "largo": {"horizon": 72, "seq_len": 168},
}
HIDDEN = 64
K = 2

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_data")

# El modelo es minúsculo (hidden=64, 6 nodos): en CPU, paralelizar sus
# operaciones entre varios threads pesa más que el propio cómputo. Medido:
# ~15 ms/predicción con 1 thread vs. ~20 ms con el default (8 threads).
torch.set_num_threads(1)


def _load_json(nombre):
    with open(os.path.join(MODEL_DIR, nombre), "r", encoding="utf-8") as f:
        return json.load(f)


def load_artifacts():
    X_test = np.load(os.path.join(MODEL_DIR, "X_test.npy")).astype(np.float32)
    tiempos_test = np.load(os.path.join(MODEL_DIR, "tiempos_test.npy"), allow_pickle=True)
    edge_index = np.load(os.path.join(MODEL_DIR, "edge_index.npy")).astype(np.int64)
    edge_weight = np.load(os.path.join(MODEL_DIR, "edge_weight.npy")).astype(np.float32)

    target_scaler = _load_json("target_scaler.json")
    feature_order = _load_json("feature_order.json")
    station_order = _load_json("station_order.json")
    metricas = _load_json("metricas_dcrnn.json")
    coords = pd.read_csv(os.path.join(MODEL_DIR, "estaciones_coords.csv"))

    edge_index_t = torch.as_tensor(edge_index, dtype=torch.long)
    edge_weight_t = torch.as_tensor(edge_weight, dtype=torch.float32)

    modelos = {}
    for esc in ESCENARIOS:
        h = ESCENARIOS[esc]["horizon"]
        modelo = ModeloDCRNN(
            in_channels=len(feature_order), hidden=HIDDEN, K=K,
            edge_index=edge_index_t, edge_weight=edge_weight_t,
        )
        state = torch.load(
            os.path.join(MODEL_DIR, f"dcrnn_{esc}_h{h}_best.pt"), map_location="cpu"
        )
        modelo.load_state_dict(state)
        modelo.eval()
        modelos[esc] = modelo

    return {
        "X_test": X_test,
        "tiempos_test": tiempos_test,
        "target_scaler": target_scaler,
        "feature_order": feature_order,
        "station_order": station_order,
        "coords": coords,
        "metricas": metricas,
        "models": modelos,
    }


def valid_index_range(artifacts, escenario):
    """Rango de índices 'j' (hora actual) con seq_len de historia y horizon
    de horizonte disponibles dentro del split de test, para el escenario dado."""
    seq_len = ESCENARIOS[escenario]["seq_len"]
    horizon = ESCENARIOS[escenario]["horizon"]
    n = artifacts["X_test"].shape[0]
    return seq_len - 1, n - 1 - horizon


def desnormalizar(y_scaled, target_scaler):
    return y_scaled * float(target_scaler["scale"]) + float(target_scaler["mean"])


def predict_at(artifacts, j, escenario):
    """Corre inferencia real del DCRNN del escenario dado para la 'hora
    actual' j (índice en X_test). Devuelve predicción a t+horizon, valor
    real (ya que es histórico) y las últimas seq_len horas observadas, todo
    en °C."""
    seq_len = ESCENARIOS[escenario]["seq_len"]
    horizon = ESCENARIOS[escenario]["horizon"]
    X = artifacts["X_test"]
    ti = int(artifacts["target_scaler"]["idx"])

    window = X[j - seq_len + 1: j + 1]                        # (L, N, F)
    x = torch.as_tensor(window, dtype=torch.float32)
    x = x.permute(1, 0, 2).unsqueeze(0)                       # (1, N, L, F)

    with torch.no_grad():
        out = artifacts["models"][escenario](x).squeeze(0).numpy()  # (N,) escalado

    pred_c = desnormalizar(out, artifacts["target_scaler"])
    actual_c = desnormalizar(X[j + horizon, :, ti], artifacts["target_scaler"])
    history_c = desnormalizar(X[j - seq_len + 1: j + 1, :, ti], artifacts["target_scaler"])

    return {
        "pred": pred_c,                # (N,) predicción t+horizon
        "actual": actual_c,            # (N,) valor real t+horizon
        "history": history_c,          # (seq_len, N) historia observada
        "t_actual": artifacts["tiempos_test"][j],
        "t_pred": artifacts["tiempos_test"][j + horizon],
        "t_history": artifacts["tiempos_test"][j - seq_len + 1: j + 1],
    }


def default_thresholds(artifacts):
    """Percentiles p05/p95 de la temperatura real observada en el split de
    test (todas las estaciones), usados como umbrales de alerta por defecto."""
    X = artifacts["X_test"]
    ti = int(artifacts["target_scaler"]["idx"])
    valores_c = desnormalizar(X[:, :, ti], artifacts["target_scaler"])
    return float(np.percentile(valores_c, 5)), float(np.percentile(valores_c, 95))
