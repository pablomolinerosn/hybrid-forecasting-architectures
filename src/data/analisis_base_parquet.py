"""
============================================================================
 analisis_base_parquet.py
----------------------------------------------------------------------------
 EDA / QA-QC de la nueva base limpia (.parquet) para el proyecto STGNN
 de predicción de temperatura media horaria en el DMQ.

 Objetivo: verificar que la base es apta para reconstruir el tensor
 (T, N=6 estaciones, F=14 features) y alimentar de forma IDÉNTICA a todos
 los modelos (A3T-GCN, DCRNN, y baselines LSTM/GRU/ARIMA).

 El script:
   1. Detecta el esquema automáticamente (formato largo vs. ancho, columna
      de tiempo, columna de estación).
   2. Analiza estructura, temporalidad, variables, nulos, inconsistencias
      y hace chequeos físicos suaves + detección de si ya está escalada.
   3. Imprime un resumen legible en consola.
   4. Guarda dos artefactos:
        - reporte_eda_base.txt   (reporte completo legible)
        - contexto_eda_base.json (resumen compacto para pegar como contexto)

 Uso:
   python analisis_base_parquet.py --parquet ruta/a/base.parquet
   (si no se pasa --parquet, usa RUTA_PARQUET_DEFECTO de abajo)
============================================================================
"""

import argparse
import json
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIGURACIÓN ESPERADA DEL PROYECTO  (ajusta si cambian los nombres)
# ---------------------------------------------------------------------------
RUTA_PARQUET_DEFECTO = "base_limpia.parquet"

ESTACIONES_ESPERADAS = [
    "Belisario", "Carapungo", "Cotocollao",
    "El Camal", "Los Chillos", "Tumbaco",
]
TARGET_ESPERADO = "temperaturaMedia"
FREQ_ESPERADA = "h"          # horaria
N_FEATURES_ESPERADAS = 14    # 8 climáticas + 6 cíclicas

# Palabras clave -> rango físico plausible (para chequeos suaves, no asserts).
# Se aplican SOLO si el nombre de la columna contiene la palabra clave.
RANGOS_FISICOS = {
    "temperatura": (-5.0, 35.0),    # Quito ~2850 m; media horaria realista
    "humedad":     (0.0, 100.0),
    "presion":     (68.0, 76.0),    # kPa aprox a la altitud de Quito
    "viento":      (0.0, 40.0),     # m/s (velocidad; dirección se filtra aparte)
    "velocidad":   (0.0, 40.0),
    "precipitacion": (0.0, 200.0),  # mm/h
    "lluvia":      (0.0, 200.0),
    "radiacion":   (0.0, 1500.0),   # W/m2
    "ozono":       (0.0, 400.0),    # ug/m3
}

SEP = "=" * 76
SUB = "-" * 76


# ---------------------------------------------------------------------------
# UTILIDADES
# ---------------------------------------------------------------------------
def log(msg=""):
    print(msg, flush=True)


def seccion(titulo):
    log("\n" + SEP)
    log(f" {titulo}")
    log(SEP)


# ---------------------------------------------------------------------------
# 1. CARGA Y DETECCIÓN DE ESQUEMA
# ---------------------------------------------------------------------------
def cargar(path):
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        log(f"[ERROR] No se pudo leer el parquet: {e}")
        log("        Verifica que 'pyarrow' o 'fastparquet' estén instalados.")
        sys.exit(1)
    return df


def detectar_columna_tiempo(df):
    # 1) por dtype datetime, 2) por nombre
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    candidatos = ["fecha", "date", "datetime", "timestamp", "time", "tiempo"]
    for c in df.columns:
        if any(k in c.lower() for k in candidatos):
            return c
    # También revisa si está en el índice
    if isinstance(df.index, pd.DatetimeIndex):
        return "__index__"
    return None


def detectar_columna_estacion(df):
    candidatos = ["estacion", "station", "nombre", "site", "sensor"]
    for c in df.columns:
        if any(k in c.lower() for k in candidatos):
            # que tenga pocas categorías (~6)
            if df[c].nunique(dropna=True) <= 20:
                return c
    return None


def preparar(df):
    """Normaliza: columna de tiempo como datetime + detecta formato."""
    dt_col = detectar_columna_tiempo(df)
    if dt_col == "__index__":
        df = df.reset_index().rename(columns={df.index.name or "index": "fecha"})
        dt_col = "fecha"
    if dt_col is not None and not pd.api.types.is_datetime64_any_dtype(df[dt_col]):
        df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")

    st_col = detectar_columna_estacion(df)
    formato = "largo (tidy: fila por tiempo-estacion)" if st_col else \
              "ancho (columnas por estacion/feature) o single-station"
    return df, dt_col, st_col, formato


# ---------------------------------------------------------------------------
# 2. ESTRUCTURA
# ---------------------------------------------------------------------------
def resumen_estructura(df, dt_col, st_col, formato, ctx):
    seccion("1. ESTRUCTURA GENERAL")
    n_rows, n_cols = df.shape
    mem_mb = df.memory_usage(deep=True).sum() / 1e6
    log(f"Filas:      {n_rows:,}")
    log(f"Columnas:   {n_cols}")
    log(f"Memoria:    {mem_mb:.1f} MB")
    log(f"Formato:    {formato}")
    log(f"Col. tiempo:    {dt_col}")
    log(f"Col. estacion:  {st_col}")

    log("\nColumnas y dtypes:")
    for c in df.columns:
        log(f"  - {c:<28} {str(df[c].dtype)}")

    ctx["n_filas"] = int(n_rows)
    ctx["n_columnas"] = int(n_cols)
    ctx["memoria_mb"] = round(float(mem_mb), 1)
    ctx["formato"] = formato
    ctx["col_tiempo"] = dt_col
    ctx["col_estacion"] = st_col
    ctx["columnas"] = {c: str(df[c].dtype) for c in df.columns}


# ---------------------------------------------------------------------------
# 3. TEMPORALIDAD
# ---------------------------------------------------------------------------
def analisis_temporal(df, dt_col, st_col, ctx):
    seccion("2. TEMPORALIDAD")
    if dt_col is None:
        log("[AVISO] No se detectó columna de tiempo. Se omite el análisis temporal.")
        ctx["temporal"] = {"error": "sin columna de tiempo"}
        return

    t = df[dt_col]
    n_nat = int(t.isna().sum())
    tmin, tmax = t.min(), t.max()
    log(f"Rango:        {tmin}  ->  {tmax}")
    log(f"Fechas NaT:   {n_nat:,}")
    if pd.notna(tmin) and pd.notna(tmax):
        span_h = (tmax - tmin).total_seconds() / 3600 + 1
        log(f"Span teórico horario: {int(span_h):,} timesteps")

    ctx_temp = {
        "rango": [str(tmin), str(tmax)],
        "fechas_nat": n_nat,
    }

    # Análisis por estación (o global si no hay estación)
    grupos = df.groupby(st_col) if st_col else [("__global__", df)]
    log("\nCobertura y regularidad por estación:")
    log(f"  {'estacion':<14}{'n':>10}{'inicio':>21}{'fin':>21}{'dups':>7}{'gaps':>8}")
    cobertura = {}
    for nombre, g in grupos:
        ts = g[dt_col].dropna().sort_values()
        n = len(ts)
        dups = int(ts.duplicated().sum())
        # gaps: diferencias distintas al paso horario esperado
        if n > 1:
            diffs = ts.diff().dropna()
            paso_mediano = diffs.median()
            gaps = int((diffs > pd.Timedelta(hours=1)).sum())
        else:
            paso_mediano, gaps = pd.NaT, 0
        log(f"  {str(nombre):<14}{n:>10,}{str(ts.min()):>21}"
            f"{str(ts.max()):>21}{dups:>7}{gaps:>8}")
        cobertura[str(nombre)] = {
            "n": int(n), "inicio": str(ts.min()), "fin": str(ts.max()),
            "duplicados": dups, "gaps_mayores_1h": gaps,
            "paso_mediano": str(paso_mediano),
        }
    ctx_temp["por_estacion"] = cobertura

    # ¿Todas las estaciones comparten exactamente la misma rejilla temporal?
    if st_col:
        conteos = {k: v["n"] for k, v in cobertura.items()}
        alineadas = len(set(conteos.values())) == 1
        log(f"\n¿Mismo nº de timesteps en todas las estaciones? {alineadas}")
        if not alineadas:
            log("  [AVISO] Las estaciones NO están alineadas -> "
                "requiere reindexar a rejilla horaria común antes de armar (T,N,F).")
        ctx_temp["estaciones_alineadas"] = bool(alineadas)

    ctx["temporal"] = ctx_temp


# ---------------------------------------------------------------------------
# 4. VARIABLES / FEATURES
# ---------------------------------------------------------------------------
def columnas_numericas(df, dt_col, st_col):
    excl = {dt_col, st_col}
    return [c for c in df.columns
            if c not in excl and pd.api.types.is_numeric_dtype(df[c])]


def analisis_variables(df, dt_col, st_col, ctx):
    seccion("3. VARIABLES (estadísticos descriptivos)")
    num_cols = columnas_numericas(df, dt_col, st_col)
    log(f"Columnas numéricas: {len(num_cols)}")

    # Detectar target y features cíclicas
    target_ok = TARGET_ESPERADO in df.columns
    ciclicas = [c for c in num_cols
                if any(k in c.lower() for k in ["sin", "cos"])]
    climaticas = [c for c in num_cols if c not in ciclicas]
    log(f"Target '{TARGET_ESPERADO}' presente: {target_ok}")
    log(f"Features cíclicas detectadas ({len(ciclicas)}): {ciclicas}")
    log(f"Features climáticas/otras ({len(climaticas)}): {climaticas}")

    desc = df[num_cols].describe().T[["mean", "std", "min", "25%",
                                      "50%", "75%", "max"]]
    log("\nDescriptivos:")
    with pd.option_context("display.max_columns", None,
                           "display.width", 200,
                           "display.float_format", lambda x: f"{x:,.4f}"):
        log(desc.to_string())

    # ¿Cíclicas dentro de [-1, 1]?
    fuera_ciclicas = {}
    for c in ciclicas:
        mn, mx = df[c].min(), df[c].max()
        if mn < -1.01 or mx > 1.01:
            fuera_ciclicas[c] = [float(mn), float(mx)]
    if fuera_ciclicas:
        log(f"\n[AVISO] Cíclicas fuera de [-1,1]: {fuera_ciclicas}")

    # ¿La base ya viene escalada (StandardScaler)? -> media~0, std~1 en climáticas
    escaladas = []
    for c in climaticas:
        m, s = df[c].mean(), df[c].std()
        if abs(m) < 0.15 and 0.7 < s < 1.3:
            escaladas.append(c)
    ya_escalada = len(escaladas) >= max(1, len(climaticas) // 2)
    log(f"\n¿La base parece YA normalizada (StandardScaler)? {ya_escalada}")
    if ya_escalada:
        log("  [IMPORTANTE] Si ya está escalada, NO vuelvas a aplicar scaler, "
            "o re-ajusta SOLO en train para evitar fuga de información.")
    else:
        log("  La base parece estar en unidades físicas -> escalar en el pipeline "
            "(ajustar StandardScaler solo en TRAIN).")

    ctx["variables"] = {
        "n_numericas": len(num_cols),
        "target_presente": bool(target_ok),
        "ciclicas": ciclicas,
        "climaticas": climaticas,
        "ciclicas_fuera_rango": fuera_ciclicas,
        "parece_normalizada": bool(ya_escalada),
        "descriptivos": json.loads(desc.round(4).to_json(orient="index")),
    }
    return num_cols, climaticas, ciclicas


# ---------------------------------------------------------------------------
# 5. NULOS
# ---------------------------------------------------------------------------
def analisis_nulos(df, dt_col, st_col, ctx):
    seccion("4. NULOS / VALORES FALTANTES")
    total = len(df)
    nulos = df.isna().sum()
    nulos = nulos[nulos > 0].sort_values(ascending=False)
    if nulos.empty:
        log("Sin valores nulos en ninguna columna.")
    else:
        log(f"{'columna':<28}{'n_nulos':>12}{'%':>10}")
        for c, n in nulos.items():
            log(f"{c:<28}{int(n):>12,}{100*n/total:>9.2f}%")

    # Nulos por estación (útil para confirmar el caso El Camal)
    por_est = {}
    if st_col:
        log("\n% de nulos por estación (promedio sobre columnas numéricas):")
        num_cols = columnas_numericas(df, dt_col, st_col)
        for nombre, g in df.groupby(st_col):
            pct = 100 * g[num_cols].isna().mean().mean()
            por_est[str(nombre)] = round(float(pct), 3)
            log(f"  {str(nombre):<14}{pct:>7.3f}%")

    # ¿Existe columna de máscara?
    mask_cols = [c for c in df.columns
                 if any(k in c.lower() for k in ["mask", "mascara", "flag", "imputad"])]
    if mask_cols:
        log(f"\nColumnas tipo máscara/flag detectadas: {mask_cols}")

    ctx["nulos"] = {
        "por_columna": {c: int(n) for c, n in nulos.items()},
        "pct_por_estacion": por_est,
        "columnas_mascara": mask_cols,
    }


# ---------------------------------------------------------------------------
# 6. INCONSISTENCIAS Y CHEQUEOS FÍSICOS
# ---------------------------------------------------------------------------
def analisis_inconsistencias(df, dt_col, st_col, num_cols, ctx):
    seccion("5. INCONSISTENCIAS Y CHEQUEOS FÍSICOS")

    # Duplicados de clave (tiempo + estacion)
    if dt_col and st_col:
        dup = int(df.duplicated(subset=[dt_col, st_col]).sum())
        log(f"Filas duplicadas por (tiempo, estacion): {dup:,}")
    elif dt_col:
        dup = int(df.duplicated(subset=[dt_col]).sum())
        log(f"Filas duplicadas por tiempo: {dup:,}")
    else:
        dup = int(df.duplicated().sum())
        log(f"Filas totalmente duplicadas: {dup:,}")

    # Columnas constantes / casi constantes
    constantes = [c for c in num_cols if df[c].nunique(dropna=True) <= 1]
    if constantes:
        log(f"[AVISO] Columnas constantes: {constantes}")

    # Chequeos físicos suaves
    log("\nChequeos físicos (conteo de valores fuera de rango plausible):")
    fisicos = {}
    for c in num_cols:
        clow = c.lower()
        # saltar cíclicas y direccion de viento
        if any(k in clow for k in ["sin", "cos", "direccion"]):
            continue
        for kw, (lo, hi) in RANGOS_FISICOS.items():
            if kw in clow:
                fuera = int(((df[c] < lo) | (df[c] > hi)).sum())
                if fuera > 0:
                    log(f"  {c:<28} fuera de [{lo}, {hi}]: {fuera:,}")
                fisicos[c] = {"rango": [lo, hi], "fuera": fuera}
                break
    if not any(v["fuera"] for v in fisicos.values()):
        log("  Sin valores fuera de rango físico plausible.")

    # Outliers estadísticos del target (IQR) como referencia
    if TARGET_ESPERADO in df.columns:
        s = df[TARGET_ESPERADO].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
        out = int(((s < lo) | (s > hi)).sum())
        log(f"\nOutliers 'extremos' en target (3·IQR): {out:,} "
            f"(límites [{lo:.2f}, {hi:.2f}])")

    ctx["inconsistencias"] = {
        "duplicados_clave": dup,
        "columnas_constantes": constantes,
        "chequeos_fisicos": fisicos,
    }


# ---------------------------------------------------------------------------
# 7. APTITUD PARA MODELOS STGNN  (¿se puede armar (T, N, F)?)
# ---------------------------------------------------------------------------
def aptitud_modelo(df, dt_col, st_col, num_cols, ctx):
    seccion("6. APTITUD PARA RECONSTRUIR EL TENSOR (T, N, F)")
    est_presentes = []
    if st_col:
        vals = [str(v) for v in df[st_col].dropna().unique()]
        est_presentes = vals
        n_est = len(vals)
        log(f"Estaciones en la base ({n_est}): {vals}")
        faltan = [e for e in ESTACIONES_ESPERADAS if e not in vals]
        extra = [e for e in vals if e not in ESTACIONES_ESPERADAS]
        if faltan:
            log(f"  [AVISO] Faltan estaciones esperadas: {faltan}")
        if extra:
            log(f"  [INFO] Estaciones no esperadas / nombres distintos: {extra}")
    else:
        log("Sin columna de estación: la base es single-station o formato ancho.")
        n_est = None

    n_feat = len(num_cols)
    log(f"\nFeatures numéricas: {n_feat} (esperadas {N_FEATURES_ESPERADAS})")
    if n_feat != N_FEATURES_ESPERADAS:
        log("  [AVISO] Nº de features difiere del esperado -> revisar mapeo "
            "de columnas antes de indexar el target (idx 6 en el pipeline previo).")

    # Índice del target en el orden actual de columnas numéricas
    if TARGET_ESPERADO in num_cols:
        idx = num_cols.index(TARGET_ESPERADO)
        log(f"Índice del target '{TARGET_ESPERADO}' en columnas numéricas: {idx} "
            f"(el pipeline previo asumía idx 6 -> confirmar o reordenar)")

    # T teórico
    if dt_col and st_col and est_presentes:
        n_ts_por_est = df.groupby(st_col)[dt_col].nunique()
        T = int(n_ts_por_est.max())
        log(f"\nTimesteps (máx por estación): {T:,}")
        log(f"Tensor objetivo aprox.: (T={T:,}, N={n_est}, F={n_feat})")

    log("\nRecomendación de partición temporal (coherente con la tesis):")
    log("  TRAIN 2008-2021 | VAL 2022-2023 | TEST 2024-2026")
    log("  -> Ajustar StandardScaler SOLO en TRAIN.")

    ctx["aptitud"] = {
        "estaciones_presentes": est_presentes,
        "n_features": n_feat,
        "target_idx_actual": num_cols.index(TARGET_ESPERADO)
        if TARGET_ESPERADO in num_cols else None,
    }


# ---------------------------------------------------------------------------
# GUARDAR ARTEFACTOS
# ---------------------------------------------------------------------------
def guardar(ctx, ruta_txt, ruta_json):
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(ctx, f, ensure_ascii=False, indent=2, default=str)
    log(f"\n[OK] Contexto compacto guardado en: {ruta_json}")
    log(f"[OK] Reporte completo (stdout) redirigible a: {ruta_txt}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="EDA base parquet STGNN-DMQ")
    parser.add_argument("--parquet", default=RUTA_PARQUET_DEFECTO,
                        help="Ruta al archivo .parquet")
    parser.add_argument("--json", default="contexto_eda_base.json")
    parser.add_argument("--txt", default="reporte_eda_base.txt")
    args = parser.parse_args()

    log(SEP)
    log(" EDA BASE PARQUET - PROYECTO STGNN TEMPERATURA DMQ")
    log(f" Ejecutado: {datetime.now():%Y-%m-%d %H:%M:%S}")
    log(f" Archivo:   {args.parquet}")
    log(SEP)

    ctx = {"archivo": args.parquet, "timestamp": str(datetime.now())}

    df = cargar(args.parquet)
    df, dt_col, st_col, formato = preparar(df)

    resumen_estructura(df, dt_col, st_col, formato, ctx)
    analisis_temporal(df, dt_col, st_col, ctx)
    num_cols, climaticas, ciclicas = analisis_variables(df, dt_col, st_col, ctx)
    analisis_nulos(df, dt_col, st_col, ctx)
    analisis_inconsistencias(df, dt_col, st_col, num_cols, ctx)
    aptitud_modelo(df, dt_col, st_col, num_cols, ctx)

    guardar(ctx, args.txt, args.json)

    seccion("FIN DEL ANÁLISIS")
    log("Pega el contenido de 'contexto_eda_base.json' o del bloque RESUMEN "
        "para continuar con la estrategia de modelado.")


if __name__ == "__main__":
    main()