import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Optional

CARPETA_DATOS = Path('data/raw')
EXTENSIONES_SOPORTADAS = {'.csv', '.xlsx', '.xls', '.txt', '.tsv', '.dat'}


def leer_cabeceras(archivo: Path) -> Optional[list]:
    """Lee solo las cabeceras de un archivo segun su extension."""
    ext = archivo.suffix.lower()
    try:
        if ext == '.csv':
            df = pd.read_csv(archivo, nrows=0)
        elif ext == '.tsv' or ext == '.dat':
            df = pd.read_csv(archivo, sep='\t', nrows=0)
        elif ext in {'.xlsx', '.xls'}:
            engine = 'xlrd' if ext == '.xls' else 'openpyxl'
            df = pd.read_excel(archivo, nrows=0, engine=engine)
        elif ext == '.txt':
            # Intentar con coma, luego con punto y coma, luego con tabulacion
            for sep in [',', ';', '\t']:
                try:
                    df = pd.read_csv(archivo, sep=sep, nrows=0)
                    if len(df.columns) > 1:
                        break
                except Exception:
                    continue
        else:
            return None
        return list(df.columns)
    except Exception as e:
        print(f"    ERROR leyendo {archivo.name}: {e}")
        return None


def main():
    if not CARPETA_DATOS.exists():
        print(f"ERROR: No se encontro la carpeta '{CARPETA_DATOS}'")
        return

    archivos = [
        f for f in CARPETA_DATOS.iterdir()
        if f.is_file() and f.suffix.lower() in EXTENSIONES_SOPORTADAS
    ]

    if not archivos:
        print(f"No se encontraron archivos de datos en '{CARPETA_DATOS}'")
        return

    archivos = sorted(archivos)
    print(f"Archivos encontrados: {len(archivos)}\n")

    # Leer cabeceras de cada archivo
    cabeceras_por_archivo = {}
    for archivo in archivos:
        cols = leer_cabeceras(archivo)
        if cols is not None:
            cabeceras_por_archivo[archivo.name] = cols
            print(f"  {archivo.name}: {len(cols)} columnas -> {cols}")
        else:
            print(f"  {archivo.name}: no se pudo leer")

    if not cabeceras_por_archivo:
        print("\nNo se pudieron leer cabeceras de ningun archivo.")
        return

    # Calcular campos comunes y diferentes
    todos_los_sets = [set(cols) for cols in cabeceras_por_archivo.values()]
    campos_comunes = set.intersection(*todos_los_sets)
    campos_totales = set.union(*todos_los_sets)
    campos_unicos = campos_totales - campos_comunes

    # Mapear cada campo diferente a que archivos lo contienen
    presencia = defaultdict(list)
    for nombre_archivo, cols in cabeceras_por_archivo.items():
        for col in cols:
            if col in campos_unicos:
                presencia[col].append(nombre_archivo)

    # Imprimir resumen
    separador = '=' * 60
    print(f"\n{separador}")
    print(f"  RESUMEN DE CABECERAS")
    print(f"{separador}")
    print(f"  Archivos analizados:   {len(cabeceras_por_archivo)}")
    print(f"  Columnas en comun:     {len(campos_comunes)}")
    print(f"  Columnas diferentes:   {len(campos_unicos)}")
    print(f"  Total columnas unicas: {len(campos_totales)}")

    print(f"\n{'─' * 60}")
    print(f"  COLUMNAS EN COMUN ({len(campos_comunes)})")
    print(f"{'─' * 60}")
    if campos_comunes:
        for col in sorted(campos_comunes):
            print(f"    - {col}")
    else:
        print("    Ningun campo es compartido por todos los archivos.")

    print(f"\n{'─' * 60}")
    print(f"  COLUMNAS EXCLUSIVAS O PARCIALES ({len(campos_unicos)})")
    print(f"{'─' * 60}")
    if campos_unicos:
        for col in sorted(campos_unicos):
            archivos_con_col = presencia[col]
            print(f"    - {col}")
            print(f"      Presente en ({len(archivos_con_col)}): {', '.join(archivos_con_col)}")
    else:
        print("    Todos los archivos comparten exactamente las mismas columnas.")

    print(f"\n{'─' * 60}")
    print(f"  DETALLE POR ARCHIVO")
    print(f"{'─' * 60}")
    for nombre, cols in sorted(cabeceras_por_archivo.items()):
        exclusivas = set(cols) - campos_comunes
        print(f"  {nombre} ({len(cols)} cols)")
        print(f"    Comunes:    {len(set(cols) & campos_comunes)}")
        print(f"    Exclusivas: {len(exclusivas)} -> {sorted(exclusivas) if exclusivas else 'ninguna'}")

    print(f"\n{separador}\n")


if __name__ == '__main__':
    main()