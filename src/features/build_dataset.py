import pandas as pd
from pathlib import Path

ARCHIVO_MDT    = Path('data/processed/mdt_remmaq.parquet')
ARCHIVO_SALIDA = Path('data/processed/mdt_weather_forecasting.parquet')

# Diccionario de siglas -> nombre descriptivo en lowerCamelCase
MAPA_VARIABLES = {
    'CO':    'monoxidoCarbono',
    'NO2':   'dioxidoNitrogeno',
    'O3':    'ozono',
    'PM2.5': 'particulasMenores2p5',
    'PM10':  'particulasMenores10',
    'SO2':   'dioxidoAzufre',
    'DIR':   'direccionViento',
    'HUM':   'humedadRelativa',
    'IUV':   'radiacionUltravioleta',
    'LLU':   'precipitacion',
    'PRE':   'presionBarometrica',
    'RS':    'radiacionSolar',
    'TMP':   'temperaturaMedia',
    'VEL':   'velocidadViento',
}


def main():
    print(f"Leyendo MDT: {ARCHIVO_MDT}")
    mdt = pd.read_parquet(ARCHIVO_MDT)

    # Asegurar que fecha es datetime
    mdt['fechaHora'] = pd.to_datetime(mdt['fecha'], errors='coerce')

    # Extraer columnas de fecha y hora
    mdt['fecha'] = mdt['fechaHora'].dt.date
    mdt['hora']      = mdt['fechaHora'].dt.time

    # Renombrar siglas de variable a nombres descriptivos
    mdt['variable'] = mdt['variable'].map(MAPA_VARIABLES).fillna(mdt['variable'])

    print(f"Variables encontradas: {sorted(mdt['variable'].unique())}\n")

    # Pivot: agrupar por (fechaHora, fecha, hora, estacion) y cada variable como columna
    print("Generando tabla pivote...")
    serie = mdt.pivot_table(
        index=['fechaHora', 'fecha', 'hora', 'estacion'],
        columns='variable',
        values='valor',
        aggfunc='mean'
    ).reset_index()

    # Eliminar el nombre del indice de columnas generado por pivot
    serie.columns.name = None

    # Ordenar columnas
    cols_fijas = ['fechaHora', 'fecha', 'hora', 'estacion']
    cols_vars  = sorted([c for c in serie.columns if c not in cols_fijas])
    serie = serie[cols_fijas + cols_vars]

    # Ordenar filas
    serie = serie.sort_values(['estacion', 'fechaHora']).reset_index(drop=True)

    # Guardar
    print(f"Guardando en: {ARCHIVO_SALIDA}")
    serie.to_parquet(ARCHIVO_SALIDA, index=False, engine='pyarrow')

    separador = '=' * 60
    print(f"\n{separador}")
    print(f"  SERIE DE TIEMPO - RESUMEN")
    print(f"{separador}")
    print(f"  Total filas:       {len(serie):,}")
    print(f"  Total columnas:    {len(serie.columns)}")
    print(f"  Rango de fechas:   {serie['fechaHora'].min()} / {serie['fechaHora'].max()}")
    print(f"  Estaciones ({serie['estacion'].nunique()}): {sorted(serie['estacion'].unique())}")
    print(f"\n  Columnas generadas:")
    for col in serie.columns:
        nulos = serie[col].isna().sum()
        pct   = nulos / len(serie) * 100
        print(f"    - {col:<28} nulos: {nulos:>8,} ({pct:5.1f}%)")
    print(f"\n  Archivo guardado: {ARCHIVO_SALIDA.resolve()}")
    print(f"{separador}\n")


if __name__ == '__main__':
    main()