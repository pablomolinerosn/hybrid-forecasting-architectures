#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
 common.py  -  Utilidades compartidas por A3T-GCN y DCRNN
----------------------------------------------------------------------------
 Garantiza que AMBOS modelos usen exactamente:
   - los mismos artefactos (tensor, splits, scaler, adyacencia),
   - el mismo ventaneo por horizonte,
   - las mismas métricas (MAE, RMSE, R2, MAPE),
   - el mismo bucle de entrenamiento/evaluación.
 Así la comparación entre modelos es justa (misma base, mismo protocolo).

 Escenarios de predicción (input window elegido para capturar cada horizonte):
   corto  : horizon = 3h   | seq_len = 24  (1 día  -> ciclo diurno completo)
   medio  : horizon = 48h  | seq_len = 96  (4 días -> persistencia multi-día)
   largo  : horizon = 72h  | seq_len = 168 (7 días -> variación sinóptica/semanal)
============================================================================
"""

import json
import os
import time
import copy

import numpy as np

# torch se importa de forma perezosa: las funciones de datos/métricas
# (numpy puro) no lo requieren; solo el Dataset y el bucle de entrenamiento.
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    _TORCH_OK = True
except Exception:                      # pragma: no cover
    _TORCH_OK = False
    Dataset = object                   # placeholder para que el import no falle


# ---------------------------------------------------------------------------
# CONFIGURACIÓN DE HORIZONTES / VENTANEO
# ---------------------------------------------------------------------------
HORIZONTES = {
    "corto": {"horizon": 3,  "seq_len": 24,  "batch": 128},
    "medio": {"horizon": 48, "seq_len": 96,  "batch": 64},
    "largo": {"horizon": 72, "seq_len": 168, "batch": 32},
}


# ---------------------------------------------------------------------------
# REPRODUCIBILIDAD
# ---------------------------------------------------------------------------
def set_seed(seed=42):
    np.random.seed(seed)
    if _TORCH_OK:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # determinismo razonable sin sacrificar demasiado rendimiento
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# CARGA DE ARTEFACTOS CANÓNICOS
# ---------------------------------------------------------------------------
def cargar_artefactos(art_dir):
    """Lee todos los artefactos producidos por preprocesamiento_canonico.py."""
    def _j(p):
        with open(os.path.join(art_dir, p), "r", encoding="utf-8") as f:
            return json.load(f)

    X = np.load(os.path.join(art_dir, "X.npy"))                 # (T, N, F) escalado
    feature_order = _j("feature_order.json")
    station_order = _j("station_order.json")
    target_scaler = _j("target_scaler.json")                    # {idx, mean, scale}
    splits = _j("splits.json")                                  # {train:[a,b], ...}
    edge_index = np.load(os.path.join(art_dir, "edge_index.npy"))   # (2, E) int64
    edge_weight = np.load(os.path.join(art_dir, "edge_weight.npy")) # (E,)   float32

    return {
        "X": X.astype(np.float32),
        "feature_order": feature_order,
        "station_order": station_order,
        "target_scaler": target_scaler,
        "target_idx": int(target_scaler["idx"]),
        "splits": splits,
        "edge_index": edge_index.astype(np.int64),
        "edge_weight": edge_weight.astype(np.float32),
        "N": len(station_order),
        "F": len(feature_order),
    }


def slice_split(X, splits, nombre):
    a, b = splits[nombre]
    return X[a:b]


# ---------------------------------------------------------------------------
# VENTANEO
# ---------------------------------------------------------------------------
def n_ventanas(T_split, seq_len, horizon):
    """Nº de ventanas válidas dentro de un split (sin cruzar fronteras)."""
    return max(0, T_split - seq_len - horizon + 1)


class VentanaDataset(Dataset):
    """
    Convierte un bloque temporal (Ts, N, F) en pares (x, y).

    Convención de horizonte:
      - x = X[i : i+seq_len]                       -> (N, seq_len, F)
      - y = X[i+seq_len+horizon-1, :, target_idx]  -> (N,)
      es decir, se predice la temperatura 'horizon' horas DESPUÉS del último
      paso observado. Para 'corto' (horizon=3) => 3 horas hacia adelante.

    Las ventanas se construyen DENTRO de cada split, por lo que ninguna
    ventana cruza la frontera train/val/test (sin fuga temporal).
    """

    def __init__(self, X_split, seq_len, horizon, target_idx):
        if not _TORCH_OK:
            raise RuntimeError("torch no disponible: no se puede crear el Dataset.")
        self.X = torch.as_tensor(np.ascontiguousarray(X_split), dtype=torch.float32)
        self.L = int(seq_len)
        self.h = int(horizon)
        self.ti = int(target_idx)
        self.n = n_ventanas(self.X.shape[0], self.L, self.h)
        if self.n <= 0:
            raise ValueError(
                f"Split demasiado corto: T={self.X.shape[0]}, "
                f"seq_len={self.L}, horizon={self.h}")

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        x = self.X[i:i + self.L]                          # (L, N, F)
        y = self.X[i + self.L + self.h - 1, :, self.ti]   # (N,)
        return x.permute(1, 0, 2).contiguous(), y         # (N, L, F), (N,)


def construir_loaders(art, seq_len, horizon, batch, num_workers=2):
    """Crea DataLoaders train/val/test para un escenario dado."""
    X, splits, ti = art["X"], art["splits"], art["target_idx"]
    ds = {}
    for nombre in ("train", "val", "test"):
        ds[nombre] = VentanaDataset(slice_split(X, splits, nombre),
                                    seq_len, horizon, ti)
    loaders = {
        "train": DataLoader(ds["train"], batch_size=batch, shuffle=True,
                            num_workers=num_workers, drop_last=False),
        "val":   DataLoader(ds["val"],   batch_size=batch, shuffle=False,
                            num_workers=num_workers),
        "test":  DataLoader(ds["test"],  batch_size=batch, shuffle=False,
                            num_workers=num_workers),
    }
    tam = {k: len(v) for k, v in ds.items()}
    return loaders, tam


# ---------------------------------------------------------------------------
# GRAFO POR LOTE (batching a nivel de grafo, disjoint union)
# ---------------------------------------------------------------------------
def construir_grafo_batch(edge_index, edge_weight, B, N):
    """
    Replica el grafo estático B veces como un único grafo disjunto:
    los nodos del lote k se desplazan +k*N. Devuelve (edge_index_B, edge_weight_B)
    en el mismo device que los tensores de entrada.
    """
    ei = torch.cat([edge_index + k * N for k in range(B)], dim=1)
    ew = edge_weight.repeat(B)
    return ei, ew


# ---------------------------------------------------------------------------
# MÉTRICAS  (numpy puro -> testeable sin torch). Se calculan en °C.
# ---------------------------------------------------------------------------
def desnormalizar(y_scaled, target_scaler):
    """Invierte StandardScaler del target -> grados Celsius."""
    return y_scaled * float(target_scaler["scale"]) + float(target_scaler["mean"])


def metricas(y_true, y_pred, eps=1e-2):
    """
    MAE, RMSE, R2 y MAPE sobre valores en °C (arrays 1-D o 2-D aplanables).

    NOTA sobre MAPE: la temperatura mínima del dataset es ~0.65 °C, por lo que
    valores cercanos a 0 °C inflan el MAPE (división por casi cero). Se aplica
    un piso 'eps' para evitar divisiones por cero, pero interpreta el MAPE con
    cautela para esta variable; MAE/RMSE en °C son más fiables aquí.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    mape = float(np.mean(np.abs(err) / np.maximum(np.abs(y_true), eps)) * 100.0)
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


def metricas_por_estacion(y_true, y_pred, estaciones):
    """y_true, y_pred: (S, N) en °C. Devuelve dict estacion -> métricas."""
    out = {}
    for j, est in enumerate(estaciones):
        out[est] = metricas(y_true[:, j], y_pred[:, j])
    return out


# ---------------------------------------------------------------------------
# ENTRENAMIENTO / EVALUACIÓN  (agnóstico al modelo)
# ---------------------------------------------------------------------------
def _pred_en_grados(model, loader, target_scaler, device):
    """Recorre un loader y devuelve (y_true_C, y_pred_C) como (S, N)."""
    model.eval()
    yt, yp = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb).cpu().numpy()          # (B, N) escalado
            yt.append(yb.numpy())
            yp.append(out)
    yt = np.concatenate(yt, axis=0)
    yp = np.concatenate(yp, axis=0)
    return desnormalizar(yt, target_scaler), desnormalizar(yp, target_scaler)


def entrenar_y_evaluar(model, loaders, estaciones, target_scaler, cfg,
                       device, salida_dir, etiqueta):
    """
    Bucle común. `model(xb)` debe aceptar xb=(B,N,L,F) y devolver (B,N) escalado.

    cfg: dict con lr, weight_decay, epochs, patience, grad_clip.
    Guarda: mejor checkpoint, curva de convergencia y predicciones de test.
    Devuelve dict de métricas (global + por estación) sobre TEST.
    """
    os.makedirs(salida_dir, exist_ok=True)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                           weight_decay=cfg["weight_decay"])
    lossf = torch.nn.MSELoss()

    mejor_val = float("inf")
    mejor_estado = copy.deepcopy(model.state_dict())
    sin_mejora = 0
    curva = []

    for ep in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        model.train()
        run = 0.0
        for xb, yb in loaders["train"]:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)                      # (B, N) escalado
            loss = lossf(out, yb)
            loss.backward()
            if cfg.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            run += loss.item() * xb.size(0)
        train_loss = run / len(loaders["train"].dataset)

        # validación en °C (MAE)
        yv_t, yv_p = _pred_en_grados(model, loaders["val"], target_scaler, device)
        val_mae = float(np.mean(np.abs(yv_p - yv_t)))
        curva.append({"epoch": ep, "train_loss": train_loss,
                      "val_mae_C": val_mae, "seg": round(time.time() - t0, 1)})
        print(f"  [{etiqueta}] epoch {ep:03d} | train_loss {train_loss:.4f} "
              f"| val_MAE {val_mae:.4f} °C | {curva[-1]['seg']}s", flush=True)

        if val_mae < mejor_val - 1e-5:
            mejor_val = val_mae
            mejor_estado = copy.deepcopy(model.state_dict())
            sin_mejora = 0
        else:
            sin_mejora += 1
            if sin_mejora >= cfg["patience"]:
                print(f"  [{etiqueta}] early stopping en epoch {ep} "
                      f"(mejor val_MAE {mejor_val:.4f} °C)", flush=True)
                break

    # restaurar mejor y evaluar en test
    model.load_state_dict(mejor_estado)
    torch.save(mejor_estado, os.path.join(salida_dir, f"{etiqueta}_best.pt"))

    yt_t, yt_p = _pred_en_grados(model, loaders["test"], target_scaler, device)
    m_global = metricas(yt_t, yt_p)
    m_est = metricas_por_estacion(yt_t, yt_p, estaciones)

    # persistir artefactos de resultados
    np.save(os.path.join(salida_dir, f"{etiqueta}_pred_test.npy"), yt_p)
    np.save(os.path.join(salida_dir, f"{etiqueta}_true_test.npy"), yt_t)
    _guardar_csv(os.path.join(salida_dir, f"{etiqueta}_convergencia.csv"), curva)

    print(f"  [{etiqueta}] TEST -> MAE {m_global['MAE']:.4f} | "
          f"RMSE {m_global['RMSE']:.4f} | R2 {m_global['R2']:.4f} | "
          f"MAPE {m_global['MAPE']:.2f}%", flush=True)

    return {"global": m_global, "por_estacion": m_est,
            "mejor_val_mae_C": mejor_val, "epochs_corridos": len(curva)}


def _guardar_csv(path, filas):
    if not filas:
        return
    cols = list(filas[0].keys())
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in filas:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
