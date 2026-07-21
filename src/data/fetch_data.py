import os
import shutil
import subprocess
import requests
import rarfile
from pathlib import Path

# Ruta absoluta de unrar (WSL / Ubuntu)
UNRAR_PATH = '/usr/bin/unrar'
rarfile.UNRAR_TOOL = UNRAR_PATH

BASE_URL = 'https://datosambiente.quito.gob.ec/datos/'
ARCHIVOS = [
    'CO.rar',
    'NO2.rar',
    'O3.rar',
    'PM2.5.rar',
    'PM10.rar',
    'SO2.rar',
    'DIR.rar',
    'HUM.rar',
    'IUV.rar',
    'LLU.rar',
    'PRE.rar',
    'RS.rar',
    'TMP.rar',
    'VEL.rar'
]

EXTENSIONES_DATOS = {'.csv', '.xlsx', '.xls', '.txt', '.json', '.xml', '.tsv', '.dat', '.parquet'}

CARPETA_DESTINO = Path('data/raw')
MAX_INTENTOS = 2


def archivos_existentes(carpeta: Path) -> set:
    return {f.stem for f in carpeta.iterdir() if f.is_file()}


def descargar_archivo(url: str, destino: Path) -> bool:
    for intento in range(1, MAX_INTENTOS + 1):
        try:
            if intento > 1:
                print(f"  Reintentando descarga (intento {intento}/{MAX_INTENTOS})...")
            else:
                print(f"  Descargando: {destino.name} ...", end=' ', flush=True)

            response = requests.get(url, stream=True, timeout=120)
            response.raise_for_status()

            descargado = 0
            with open(destino, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        descargado += len(chunk)

            size_mb = descargado / (1024 * 1024)
            print(f"OK ({size_mb:.1f} MB)")
            return True

        except requests.exceptions.HTTPError as e:
            # Obtener el status code de forma segura: puede faltar `response` en la excepción
            resp = getattr(e, 'response', None)
            if resp is None:
                # fallback: usar el objeto `response` de la variable local si existe
                resp = locals().get('response')
            status = getattr(resp, 'status_code', 'N/A') if resp is not None else 'N/A'
            print(f"ERROR HTTP {status}")
        except requests.exceptions.ConnectionError:
            print("ERROR de conexion")
        except requests.exceptions.Timeout:
            print("ERROR tiempo de espera agotado")
        except Exception as e:
            print(f"ERROR inesperado: {e}")

        if destino.exists():
            destino.unlink()

    print(f"  Descarga fallida tras {MAX_INTENTOS} intentos: {destino.name}")
    return False


def extraer_con_subprocess(rar_path: Path, carpeta_temp: Path) -> None:
    """Extrae usando subprocess con la ruta completa de unrar."""
    if not UNRAR_PATH:
        raise RuntimeError("No se encontro el ejecutable unrar en el sistema.")
    resultado = subprocess.run(
        [UNRAR_PATH, 'x', '-y', str(rar_path), str(carpeta_temp) + os.sep],
        capture_output=True,
        text=True
    )
    if resultado.returncode != 0:
        raise RuntimeError(f"unrar fallo (codigo {resultado.returncode}): {resultado.stderr.strip()}")


def extraer_y_mover(rar_path: Path, carpeta_destino: Path) -> list:
    carpeta_temp = rar_path.with_suffix('')
    carpeta_temp.mkdir(exist_ok=True)
    guardados = []

    try:
        print(f"  Extrayendo: {rar_path.name} ...", end=' ', flush=True)

        try:
            with rarfile.RarFile(rar_path) as rf:
                rf.extractall(carpeta_temp)
        except (rarfile.RarCannotExec, rarfile.BadRarName, Exception):
            # Fallback directo a subprocess con ruta absoluta
            extraer_con_subprocess(rar_path, carpeta_temp)

        print("OK")

        datos_encontrados = [
            f for f in carpeta_temp.rglob('*')
            if f.is_file() and f.suffix.lower() in EXTENSIONES_DATOS
        ]

        if datos_encontrados:
            for archivo in datos_encontrados:
                destino_final = carpeta_destino / archivo.name
                if destino_final.exists():
                    destino_final = carpeta_destino / f"{rar_path.stem}_{archivo.name}"
                shutil.move(str(archivo), destino_final)
                guardados.append(destino_final.name)
                print(f"    Guardado: {destino_final.name}")
        else:
            print(f"    Sin archivos de datos reconocidos en {rar_path.name}")

    except rarfile.BadRarFile:
        print(f"ERROR: archivo .rar corrupto o invalido")
    except Exception as e:
        print(f"ERROR al extraer: {e}")
    finally:
        if carpeta_temp.exists():
            shutil.rmtree(carpeta_temp)
        if rar_path.exists():
            rar_path.unlink()
            print(f"    Eliminado: {rar_path.name}")

    return guardados


def main():
    CARPETA_DESTINO.mkdir(exist_ok=True)
    print(f"Carpeta de destino: {CARPETA_DESTINO.resolve()}\n")

    existentes = archivos_existentes(CARPETA_DESTINO)
    exitosos, fallidos, omitidos = [], [], []
    todos_los_datos = []

    for archivo in ARCHIVOS:
        nombre_base = Path(archivo).stem
        print(f"\n[{archivo}]")

        if nombre_base in existentes:
            print(f"  Ya existe '{nombre_base}' en la carpeta, omitiendo.")
            omitidos.append(archivo)
            continue

        url = BASE_URL + archivo
        rar_path = CARPETA_DESTINO / archivo

        ok = descargar_archivo(url, rar_path)
        if not ok:
            fallidos.append(archivo)
            continue

        datos = extraer_y_mover(rar_path, CARPETA_DESTINO)
        if datos:
            exitosos.append(archivo)
            todos_los_datos.extend(datos)
        else:
            fallidos.append(archivo)

    print(f"\n{'='*50}")
    print(f"  Procesados:              {len(exitosos)}/{len(ARCHIVOS)}")
    print(f"  Omitidos (ya existen):   {len(omitidos)}")
    print(f"  Fallidos:                {len(fallidos)}")
    print(f"  Archivos de datos total: {len(todos_los_datos)}")
    if fallidos:
        print(f"  Archivos fallidos: {', '.join(fallidos)}")
    print(f"  Carpeta final: {CARPETA_DESTINO.resolve()}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()