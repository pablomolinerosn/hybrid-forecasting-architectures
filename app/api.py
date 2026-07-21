"""
API FastAPI + frontend estático — dashboard de alerta temprana DCRNN.

Reemplaza la versión Streamlit: el modelo y los artefactos (model.py,
data_utils.py) no cambian, pero ahora se sirven como una API JSON consumida
por un front propio (static/) en vez de reejecutar un script completo en
cada interacción. Los umbrales de alerta se comparan en el cliente (no
requieren red), así que moverlos ya no toca el backend ni recalcula nada.
"""

from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import data_utils as du

app = FastAPI(title="DCRNN Alerta Temprana — API")

ARTIFACTS = du.load_artifacts()
ESCENARIOS = list(du.ESCENARIOS)  # ["corto", "medio", "largo"]
RANGOS_J = {esc: du.valid_index_range(ARTIFACTS, esc) for esc in ESCENARIOS}
TS0 = pd.Timestamp(ARTIFACTS["tiempos_test"][0])
DEFAULT_FRIO, DEFAULT_CALOR = du.default_thresholds(ARTIFACTS)


def _ts_to_j(ts: pd.Timestamp, escenario: str) -> int:
    lo, hi = RANGOS_J[escenario]
    j = int((ts - TS0) / pd.Timedelta(hours=1))
    return max(lo, min(hi, j))


@lru_cache(maxsize=512)
def _predict_cached(j: int, escenario: str):
    return du.predict_at(ARTIFACTS, j, escenario)


def _check_escenario(escenario: str) -> str:
    if escenario not in ESCENARIOS:
        raise HTTPException(400, f"Escenario inválido: {escenario!r}. Use uno de {ESCENARIOS}.")
    return escenario


class EstacionCoord(BaseModel):
    estacion: str
    lat: float
    lon: float


class RangoValido(BaseModel):
    min: str
    max: str


class Umbrales(BaseModel):
    frio: float
    calor: float


class EscenarioMeta(BaseModel):
    metrics_global: dict
    metrics_por_estacion: dict
    config: dict
    valid_range: RangoValido


class Meta(BaseModel):
    stations: list[str]
    coords: list[EstacionCoord]
    escenarios: dict[str, EscenarioMeta]
    default_thresholds: Umbrales


@app.get("/api/meta", response_model=Meta)
def get_meta():
    coords = [EstacionCoord(**row) for row in ARTIFACTS["coords"].to_dict("records")]
    escenarios = {}
    for esc in ESCENARIOS:
        met = ARTIFACTS["metricas"][esc]
        config = dict(met["config"])
        config["epochs_corridos"] = met["epochs_corridos"]
        lo, hi = RANGOS_J[esc]
        escenarios[esc] = EscenarioMeta(
            metrics_global=met["global"],
            metrics_por_estacion=met["por_estacion"],
            config=config,
            valid_range=RangoValido(
                min=pd.Timestamp(ARTIFACTS["tiempos_test"][lo]).isoformat(),
                max=pd.Timestamp(ARTIFACTS["tiempos_test"][hi]).isoformat(),
            ),
        )
    return Meta(
        stations=ARTIFACTS["station_order"],
        coords=coords,
        escenarios=escenarios,
        default_thresholds=Umbrales(frio=DEFAULT_FRIO, calor=DEFAULT_CALOR),
    )


class HistoryPoint(BaseModel):
    t: str
    v: float


class EstacionPred(BaseModel):
    estacion: str
    pred: float
    actual: float
    history: list[HistoryPoint]


class Prediccion(BaseModel):
    t_actual: str
    t_pred: str
    stations: list[EstacionPred]


@app.get("/api/predict", response_model=Prediccion)
def predict(ts: str, escenario: str = "corto"):
    """Corre inferencia real del DCRNN para la hora 'actual' simulada `ts`
    (ISO, p.ej. 2026-03-31T20:00:00) en el escenario dado (corto/medio/largo).
    Se recorta al rango de test disponible para ese escenario."""
    _check_escenario(escenario)
    try:
        ts_parsed = pd.Timestamp(ts)
    except ValueError:
        raise HTTPException(400, "Timestamp inválido")

    j = _ts_to_j(ts_parsed, escenario)
    r = _predict_cached(j, escenario)

    stations = []
    for i, est in enumerate(ARTIFACTS["station_order"]):
        history = [
            HistoryPoint(t=pd.Timestamp(t).isoformat(), v=float(v))
            for t, v in zip(r["t_history"], r["history"][:, i])
        ]
        stations.append(EstacionPred(
            estacion=est, pred=float(r["pred"][i]), actual=float(r["actual"][i]),
            history=history,
        ))

    return Prediccion(
        t_actual=pd.Timestamp(r["t_actual"]).isoformat(),
        t_pred=pd.Timestamp(r["t_pred"]).isoformat(),
        stations=stations,
    )


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
