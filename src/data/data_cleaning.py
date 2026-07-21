import pandas as pd
from pathlib import Path

ARCHIVO_ENTRADA = Path('data/processed/mdt_weather_forecasting.parquet')
ARCHIVO_SALIDA  = Path('data/processed/mdt_weather_forecasting_eje.parquet')

# Cargar datos
df = pd.read_parquet(ARCHIVO_ENTRADA)
df['fecha'] = pd.to_datetime(df['fecha'])

print(f"Shape original: {df.shape}")
print(f"Estaciones originales: {sorted(df['estacion'].unique())}")

# Eliminar estaciones con alto % de nulos en variables 
ESTACIONES_EXCLUIR = ['Condado', 'Jipijapa', 'Centro', 'Guamani', 'SanAntonio']

df = df[~df['estacion'].isin(ESTACIONES_EXCLUIR)].copy()

print(f"\nEstaciones eliminadas: {ESTACIONES_EXCLUIR}")
print(f"Estaciones restantes:  {sorted(df['estacion'].unique())}")

# Variables con poco aporte en la estimación de temperatura y precipitación
VARIABLES_EXCLUIR = [
    'dioxidoAzufre',
    'dioxidoNitrogeno',
    'monoxidoCarbono',
    'particulasMenores10',
    'particulasMenores2p5',
    'radiacionUltravioleta'
]

VARIABLES_EXCLUIR_PRESENTES = [v for v in VARIABLES_EXCLUIR if v in df.columns]
df = df.drop(columns=VARIABLES_EXCLUIR_PRESENTES)

print(f"\nColumnas eliminadas: {VARIABLES_EXCLUIR_PRESENTES}")
print(f"Columnas restantes:  {[c for c in df.columns]}")

# Resumen de ejecución
VARS_CLEAN = [c for c in df.columns if c not in ['fecha', 'fechaHora', 'hora', 'estacion']]

print(f"\nShape tras limpieza: {df.shape}")
print(f"\nDatos faltantes tras limpieza:")
faltantes = pd.DataFrame({
    'nulos':   df[VARS_CLEAN].isnull().sum(),
    'pct (%)': (df[VARS_CLEAN].isnull().mean() * 100).round(2)
}).sort_values('pct (%)', ascending=False)
print(faltantes.to_string())

# Guardar el dataframe limpio
df.to_parquet(ARCHIVO_SALIDA, index=False, engine='pyarrow')

print(f"\nArchivo guardado: {ARCHIVO_SALIDA.resolve()}")