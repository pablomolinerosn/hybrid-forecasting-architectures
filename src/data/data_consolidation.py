import pandas as pd
from pathlib import Path
from typing import Optional

CARPETA_DATOS = Path('data/raw')
ARCHIVO_SALIDA = Path('data/processed/mdt_remmaq.parquet')

# Mapa de homologacion: variante original -> nombre canonico
MAPA_ESTACIONES = {
    'BELISARIO':   'Belisario',
    'Belisario':   'Belisario',
    'CARAPUNGO':   'Carapungo',
    'Carapungo':   'Carapungo',
    'CENTRO':      'Centro',
    'Centro':      'Centro',
    'COTOCOLLAO':  'Cotocollao',
    'Cotocollao':  'Cotocollao',
    'EL CAMAL':    'ElCamal',
    'ElCamal':     'ElCamal',
    'GUAMANI':     'Guamani',
    'Guamaní':     'Guamani',
    'Guamani':     'Guamani',
    'LOS CHILLOS': 'LosChillos',
    'LosChillos':  'LosChillos',
    'Los Chillos': 'LosChillos',
    'SAN ANTONIO': 'SanAntonio',
    'SanAntonio':  'SanAntonio',
    'San Antonio': 'SanAntonio',
    'TUMBACO':     'Tumbaco',
    'Tumbaco':     'Tumbaco',
    'CONDADO':     'Condado',
    'Condado':     'Condado',
    'Jipijapa':    'Jipijapa',
    'fecha':       'fecha',
    'Fecha':       'fecha',
}

EXTENSIONES = {'.xlsx', '.xls', '.csv', '.txt', '.tsv', '.dat'}


def leer_archivo(archivo: Path) -> Optional[pd.DataFrame]:
    ext = archivo.suffix.lower()
    try:
        if ext == '.csv':
            return pd.read_csv(archivo)
        elif ext in {'.tsv', '.dat'}:
            return pd.read_csv(archivo, sep='\t')
        elif ext in {'.xlsx', '.xls'}:
            engine = 'xlrd' if ext == '.xls' else 'openpyxl'
            return pd.read_excel(archivo, engine=engine)
        elif ext == '.txt':
            for sep in [',', ';', '\t']:
                df = pd.read_csv(archivo, sep=sep)
                if len(df.columns) > 1:
                    return df
    except Exception as e:
        print(f"  ERROR leyendo {archivo.name}: {e}")
    return None


def homologar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={col: MAPA_ESTACIONES.get(col, col) for col in df.columns})


def extraer_variable(nombre_archivo: str) -> str:
    """Extrae la sigla de la variable desde el nombre del archivo (antes del '_')."""
    return nombre_archivo.split('_')[0]


def main():
    archivos = sorted([
        f for f in CARPETA_DATOS.iterdir()
        if f.is_file()
        and f.suffix.lower() in EXTENSIONES
        and f.stem != 'MDT_unificado'
    ])

    if not archivos:
        print(f"No se encontraron archivos en '{CARPETA_DATOS}'")
        return

    print(f"Archivos a procesar: {len(archivos)}\n")

    frames = []

    for archivo in archivos:
        variable = extraer_variable(archivo.stem)
        print(f"[{archivo.name}]")
        print(f"  Variable: {variable}")

        df = leer_archivo(archivo)
        if df is None:
            print(f"  Omitido por error de lectura.\n")
            continue

        print(f"  Columnas originales: {list(df.columns)}")

        # Homologar nombres de columnas
        df = homologar_columnas(df)

        if 'fecha' not in df.columns:
            print(f"  ADVERTENCIA: no se encontro columna Fecha, omitiendo archivo.\n")
            continue

        # Convertir a formato largo: una fila por (Fecha, Estacion)
        cols_estacion = [c for c in df.columns if c != 'fecha']
        df_largo = df.melt(
            id_vars='fecha',
            value_vars=cols_estacion,
            var_name='estacion',
            value_name='valor'
        )

        # Columna identificadora de la variable
        df_largo.insert(2, 'variable', variable)

        print(f"  Columnas homologadas: {['fecha'] + cols_estacion}")
        print(f"  Filas generadas: {len(df_largo):,}\n")

        frames.append(df_largo)

    if not frames:
        print("No se pudo procesar ningun archivo.")
        return

    # Union de todos los dataframes
    print("Unificando todos los archivos...")
    mdt = pd.concat(frames, ignore_index=True)

    # Normalizar fechas
    mdt['fecha'] = pd.to_datetime(mdt['fecha'], errors='coerce')

    # Ordenar
    mdt = mdt.sort_values(['variable', 'estacion', 'fecha']).reset_index(drop=True)

    # Estructura final: Fecha | Estacion | Variable | Valor
    mdt = mdt[['fecha', 'estacion', 'variable', 'valor']]

    # Guardar
    print(f"Guardando en: {ARCHIVO_SALIDA}")
    mdt.to_parquet(ARCHIVO_SALIDA, index=False, engine='pyarrow')

    separador = '=' * 55
    print(f"\n{separador}")
    print(f"  MDT UNIFICADO - RESUMEN")
    print(f"{separador}")
    print(f"  Total filas:       {len(mdt):,}")
    print(f"  Rango de fechas:   {mdt['fecha'].min().date()} / {mdt['fecha'].max().date()}")
    print(f"  Variables ({mdt['variable'].nunique()}):  {sorted(mdt['variable'].unique())}")
    print(f"  Estaciones ({mdt['estacion'].nunique()}): {sorted(mdt['estacion'].unique())}")
    print(f"  Archivo guardado:  {ARCHIVO_SALIDA.resolve()}")
    print(f"{separador}\n")


if __name__ == '__main__':
    main()