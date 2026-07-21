# Arquitecturas híbridas de pronóstico — Temperatura DMQ (REMMAQ)

Trabajo de Fin de Máster en Inteligencia Artificial Aplicada. Pronóstico
horario de temperatura media sobre la red de estaciones de calidad del
aire/meteorología del Distrito Metropolitano de Quito (REMMAQ), comparando
cuatro familias de modelos bajo condiciones lo más homologadas posible
(mismos splits temporales, mismas métricas, mismo periodo de test):

- **A3T-GCN** y **DCRNN** — redes neuronales de grafos espacio-temporales
  (`torch_geometric_temporal`), sobre un grafo de 6 estaciones.
- **ARIMA / SARIMA / ARIMAX / SARIMAX** — modelos estadísticos clásicos,
  ajustados por máxima verosimilitud, con pronóstico walk-forward.
- **ClimaX** — modelo fundacional de clima, ajustado por fine-tuning.

**Resultado central**: **DCRNN** es el modelo ganador y supera a **ClimaX**
de forma estadísticamente significativa en los tres horizontes evaluados
(test de Diebold-Mariano). Frente a **SARIMA** (la mejor variante de la
familia ARIMA), DCRNN gana con significancia en el horizonte corto (3h),
pero a partir de 48h ambos son **estadísticamente equivalentes** — la
ventaja de la red espacio-temporal está concentrada en el corto plazo, no
es uniforme en todos los horizontes. Detalle completo en
[`docs/analisis_resultados.md`](docs/analisis_resultados.md).

---

## Tabla de contenido

1. [Estructura del proyecto](#estructura-del-proyecto)
2. [Instalación](#instalación)
3. [Cómo reproducir el pipeline completo](#cómo-reproducir-el-pipeline-completo)
4. [Metodología por familia de modelos](#metodología-por-familia-de-modelos)
5. [Resultados](#resultados)
6. [Notas metodológicas y limitaciones conocidas](#notas-metodológicas-y-limitaciones-conocidas)
7. [Aplicación de demostración](#aplicación-de-demostración)
8. [Datos y artefactos pesados](#datos-y-artefactos-pesados)

---

## Estructura del proyecto

```
data/
  raw/                    Datos crudos REMMAQ (.xlsx)
  processed/              Parquets consolidados/procesados
artefactos/                Tensor (T,N,F), splits, scaler, grafo (adyacencia,
                            edge_index/edge_weight)
notebooks/                  EDA, imputación, metodología, entrenamiento
                            exploratorio, ejecución de ARIMA y ClimaX
docs/
  metodologia_entrenamiento.md   Metodología detallada de A3T-GCN/DCRNN
                                   (grafo, kernel gaussiano, ventaneo, splits)
  analisis_resultados.md          Comparación final DCRNN vs. ClimaX vs.
                                   SARIMA, tests de Diebold-Mariano, hallazgos
  figuras/                        Figuras usadas en ambos documentos
resultados/
  a3tgcn/, dcrnn/          Checkpoints y métricas por escenario (corto/medio/largo)
  arima/                    Modelos base y de tuning (.pkl, no versionados) +
                            predicciones
  climax/                   Pesos, curvas de convergencia y predicciones de
                            ClimaX fine-tuneado, por escenario
  diebold_mariano/          Resultados de los tests de significancia estadística
  figuras/                  Gráficas y mapas generados para el TFM
  tablas/                   Tablas de métricas exportadas
  exploratorio_legacy/      Resultados de una iteración previa del pipeline
                            (esquema de horizontes h6/h12/h24) -- referencia
                            histórica, no son los resultados finales
src/
  data/                    Descarga, validación, consolidación y limpieza de
                            datos crudos
  features/                Construcción del dataset y preprocesamiento
                            canónico (tensor T×N×F)
  models/
    common.py               Utilidades compartidas por A3T-GCN y DCRNN
                            (dataset, batching de grafo, métricas, loop de
                            entrenamiento)
    a3tgcn/                  Arquitectura + entrenamiento/evaluación de A3T-GCN
    dcrnn/                   Arquitectura + entrenamiento/evaluación de DCRNN
    arima/                   Pipeline ARIMA/SARIMA/ARIMAX/SARIMAX completo
                            (feature engineering, modelos base, tuning,
                            predicciones walk-forward, análisis) + run_pipeline.sh
  evaluation/               Tablas de métricas y tests de Diebold-Mariano
                            (GNN-GNN, ARIMA-GNN, y los 3 modelos por pares)
  visualization/            Generación de mapas y estilo homologado de gráficas
  utils/                    Plantilla de diagramas de flujo (graphviz)
app/                        Servicio de inferencia (FastAPI + Docker) del
                            modelo ganador (DCRNN, escenario corto)
```

## Instalación

Requiere Python 3.10.

```bash
python3.10 -m venv .venv
source .venv/bin/activate

# torch para tu versión de CUDA (o CPU) desde el índice de PyTorch:
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt

# Además, en el sistema (no via pip):
#   - graphviz (binario `dot`), para src/utils/ y los diagramas de docs/figuras/
#   - unrar, si vas a correr src/data/fetch_data.py
```

`app/` es un servicio independiente con su propio `requirements.txt` (variante
CPU, pensada para Docker) — ver [`app/README.md`](app/README.md).

## Cómo reproducir el pipeline completo

Todos los comandos se ejecutan **desde la raíz del repositorio**, salvo que
se indique lo contrario.

### 1. Datos y preprocesamiento canónico (compartido por A3T-GCN y DCRNN)

```bash
python src/data/fetch_data.py                     # descarga datos crudos REMMAQ
python src/data/data_validation.py                 # valida esquema de data/raw/
python src/data/data_consolidation.py               # -> data/processed/mdt_remmaq.parquet
python src/features/build_dataset.py                 # -> mdt_weather_forecasting.parquet
python src/data/data_cleaning.py                      # -> mdt_weather_forecasting_eje.parquet
python src/features/preprocesamiento_canonico.py      # -> tensor + artefactos/ (X, grafo, splits, scaler)
```

### 2. A3T-GCN y DCRNN

```bash
python src/models/a3tgcn/entrenar_a3tgcn.py   # entrena/evalúa A3T-GCN en corto/medio/largo
python src/models/dcrnn/entrenar_dcrnn.py     # entrena/evalúa DCRNN en corto/medio/largo

python src/evaluation/tabla_resultados.py           # tablas de métricas (test / marzo 2026)
python src/evaluation/test_diebold_mariano.py       # significancia A3T-GCN vs. DCRNN
```

### 3. ARIMA / SARIMA / ARIMAX / SARIMAX

```bash
bash src/models/arima/run_pipeline.sh
```

Corre, en orden, las 6 etapas del pipeline (feature engineering → modelos
base → reentrenamiento de faltantes → búsqueda de hiperparámetros →
predicciones walk-forward → análisis de resultados). Cada etapa es
**reanudable**: si se interrumpe, correr el script de nuevo salta lo que ya
esté guardado en `resultados/arima/`, en vez de repetirlo. Tiempo estimado en
una máquina de 16 hilos / 32GB: 3-5 horas, dominado por el ajuste de
SARIMA/SARIMAX (la estacionalidad diaria hace el ajuste mucho más lento que
un ARIMA simple). Ver la sección de [notas metodológicas](#notas-metodológicas-y-limitaciones-conocidas)
para el detalle de los límites de memoria/paralelismo encontrados y cómo se
resolvieron.

### 4. ClimaX

`notebooks/climax_embeddings_*.ipynb` y `notebooks/climax_inferencias_*.ipynb`
(uno de cada por escenario): fine-tuning de un modelo fundacional de clima
preentrenado. Requiere GPU + el paquete `climax` y está pensado para correr
en Google Colab, no en el `.venv` local — los pesos, curvas de convergencia y
predicciones ya generados están en `resultados/climax/`.

### 5. Comparación final de las 3 familias

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/analisis_comparativo_modelos.ipynb
```

Genera las tablas y figuras de `docs/analisis_resultados.md`, y corre los 3
tests de Diebold-Mariano (`src/evaluation/test_diebold_mariano_arima.py` y
`test_diebold_mariano_todos.py`).

---

## Metodología por familia de modelos

Resumen; ver [`docs/metodologia_entrenamiento.md`](docs/metodologia_entrenamiento.md)
(A3T-GCN/DCRNN, definición formal del grafo) y
[`docs/analisis_resultados.md`](docs/analisis_resultados.md) (comparación
completa, incluida esta misma tabla con más detalle) para el desarrollo
completo.

### A3T-GCN

Red de atención espacio-temporal sobre el grafo de las 6 estaciones REMMAQ
(kernel gaussiano de distancias Haversine), combinando convoluciones de
grafo con atención temporal para ponderar horas pasadas relevantes.
**Resultado**: perdió frente a DCRNN en el test de Diebold-Mariano en los
tres escenarios — se descartó de las comparaciones posteriores por ese
resultado, no por limitaciones de implementación.

### DCRNN (ganador de la familia GNN)

Combina convolución de difusión sobre el grafo de estaciones con una GRU
que modela la dinámica temporal. Gana con significancia estadística clara
en el horizonte corto (h=3h). Su ventaja se diluye y deja de ser
significativa a partir de 48h frente a un SARIMA simple — la señal espacial
del grafo aporta menos cuanto más lejano es el horizonte. Requiere entrenar
un modelo por escenario (no hace pronóstico multi-horizonte nativo) y
depende de un grafo de adyacencia fijo.

### ClimaX (modelo fundacional, fine-tuneado)

Parte de un modelo preentrenado sobre reanálisis climático global, ajustado
a los datos locales del DMQ. El escenario largo (h=72h) se entrenó de forma
incompleta (26 de 150 épocas por early stopping), con R² negativo y una
predicción que colapsa a una curva casi plana. Incluso en corto/medio plazo
queda sistemáticamente por debajo de DCRNN y SARIMA en las 4 métricas.
Solo es competitivo (aunque no gane) en estaciones de baja variabilidad
térmica. Corre en Colab/GPU externa, no en el `.venv` local.

### ARIMA / SARIMA / ARIMAX / SARIMAX (familia estadística clásica)

Modelos univariados (ARIMA/SARIMA) y con variables exógenas
(ARIMAX/SARIMAX), ajustados por máxima verosimilitud vía filtro de Kalman,
con pronóstico walk-forward hora a hora. **Hallazgo propio**: dentro de esta
familia, **SARIMA (sin exógenas) supera consistentemente a SARIMAX** en los
3 horizontes — la estacionalidad diaria (`seasonal_order=(1,0,1,24)`)
captura casi toda la señal predecible, y las exógenas meteorológicas
añaden ruido en vez de señal una vez que esa estacionalidad ya está
modelada. Es univariado por estación (no comparte información espacial
como el grafo de DCRNN/A3T-GCN), y el ajuste con estacionalidad es
computacionalmente costoso (ver notas metodológicas).

---

## Resultados

Métricas globales sobre el test completo (2024-01-01 a 2026-03-31):

| Escenario | Modelo | MAE (°C) | RMSE (°C) | R² | MAPE (%) |
|---|---|---|---|---|---|
| Corto (h=3) | ClimaX | 1.567 | 2.124 | 0.626 | 9.55 |
| Corto (h=3) | **DCRNN** | **0.657** | **0.946** | **0.926** | **4.27** |
| Corto (h=3) | SARIMA | 0.808 | 1.139 | 0.893 | 5.24 |
| Medio (h=48) | ClimaX | 1.798 | 2.407 | 0.600 | 10.09 |
| Medio (h=48) | **DCRNN** | **1.040** | **1.405** | **0.837** | **6.88** |
| Medio (h=48) | SARIMA | 1.063 | 1.425 | 0.832 | 7.00 |
| Largo (h=72) | ClimaX | 1.180 | 1.491 | -0.295 | 8.66 |
| Largo (h=72) | DCRNN | 1.094 | 1.454 | 0.826 | 7.16 |
| Largo (h=72) | **SARIMA** | **1.078** | **1.444** | **0.828** | **7.10** |

Test de Diebold-Mariano (significancia estadística, alfa=0.05):

| Par | Corto | Medio | Largo |
|---|---|---|---|
| DCRNN vs. ClimaX | **DCRNN gana** (p≈0) | **DCRNN gana** (p≈1e-175) | **DCRNN gana** (p≈3e-7) |
| DCRNN vs. SARIMA | **DCRNN gana** (p≈4e-123) | Equivalentes (p≈0.10)¹ | Equivalentes (p≈0.29)¹ |

¹ Con densidad de muestreo igualada a ClimaX (necesaria para incluirlo en el
mismo test), el resultado de "medio" pasa a favorecer a DCRNN y el de
"largo" a SARIMA — pero esa muestra queda anclada a 1-2 horas del día por
construcción, así que es menos representativa que esta fila (densidad
completa, todas las horas). Ver `docs/analisis_resultados.md` secciones 3 y
3b para el detalle y la advertencia metodológica completa.

Tablas y figuras completas, desglose por estación, y la redacción completa
del análisis: [`docs/analisis_resultados.md`](docs/analisis_resultados.md).


---

## Aplicación de demostración

`app/` sirve el modelo ganador (DCRNN, escenario corto h=3h) vía una API
FastAPI + un front propio (HTML/CSS/JS sin frameworks), con inferencia real
sobre ventanas históricas del split de test (no gráficos precalculados), un
mapa de las 6 estaciones REMMAQ coloreadas por temperatura pronosticada, y
alertas de frío/calor por umbral. Pensada para desplegarse como contenedor
Docker (p. ej. en Hugging Face Spaces). Instrucciones completas de build y
ejecución local: [`app/README.md`](app/README.md).

---