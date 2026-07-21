#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Genera el diagrama de funcionamiento de la app de alerta temprana
(multi-horizonte: corto/medio/largo) como PNG de alta resolución,
en el mismo estilo visual (Graphviz, paleta pastel) que los diagramas
de arquitectura del TFM (src/utils/estilos_diagramas.ipynb).

Uso:
  python src/utils/generar_diagrama_alerta_app.py
Escribe docs/figuras/diagrama_app_alerta_temprana.png
"""

import os

import graphviz

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SALIDA = os.path.join(RAIZ, "docs", "figuras", "diagrama_app_alerta_temprana")

FONT = "DejaVu Sans"

dot = graphviz.Digraph("AlertaTemprana_MultiHorizonte")
dot.attr(
    rankdir="TB",
    dpi="340",
    bgcolor="white",
    fontname=FONT,
    fontsize="22",
    pad="0.5",
    nodesep="0.45",
    ranksep="0.7",
    labelloc="t",
    label="Funcionamiento de la app de Alerta Temprana — DCRNN (REMMAQ, DMQ Quito)\n"
          "Ahora con selector de horizonte: corto (+3h) · medio (+48h) · largo (+72h)\n ",
)
dot.attr("node", fontname=FONT, fontsize="14")
dot.attr("edge", fontname=FONT, fontsize="11.5", color="#555555", penwidth="1.4", arrowsize="0.8")

# ---------------------------------------------------------------------
# estilos (misma paleta que estilos_diagramas.ipynb)
# ---------------------------------------------------------------------
input_style = dict(shape="box", style="filled,rounded", fillcolor="#E1F5FE", color="#01579B", fontcolor="#01579B")
process_style = dict(shape="box", style="filled", fillcolor="#FFF3E0", color="#E65100", fontcolor="#7a3800")
model_style = dict(shape="box", style="filled,rounded", fillcolor="#E8F5E9", color="#2E7D32", fontcolor="#1b5e20")
db_style = dict(shape="cylinder", style="filled", fillcolor="#E8EAF6", color="#880E4F", fontcolor="#880E4F")
output_style = dict(shape="box", style="filled,rounded", fillcolor="#FCE4EC", color="#880E4F", fontcolor="#880E4F")
decision_style = dict(shape="diamond", style="filled", fillcolor="#FCE4EC", color="#880E4F", fontcolor="#880E4F")
new_style = dict(shape="box", style="filled,rounded,dashed", fillcolor="#FFF9C4", color="#F57F17", fontcolor="#7a5c00", penwidth="2")

# ---------------------------------------------------------------------
# 1 · usuario
# ---------------------------------------------------------------------
with dot.subgraph(name="cluster_1") as c:
    c.attr(label="①  USUARIO", fontsize="15", fontcolor="#37474F", style="rounded,dashed", color="#B0BEC5", margin="18")
    c.node("A", "☺  Elige escenario, fecha/hora,\nestación y umbrales frío/calor (°C)", **input_style)

# ---------------------------------------------------------------------
# 2 · frontend
# ---------------------------------------------------------------------
with dot.subgraph(name="cluster_2") as c:
    c.attr(label="②  FRONTEND — navegador (index.html + app.js)", fontsize="15", fontcolor="#37474F",
           style="rounded,dashed", color="#B0BEC5", margin="18")
    c.node("SEL", "⇄  NUEVO: selector de horizonte\ncorto  ·  medio  ·  largo", **new_style)
    c.node("B", "Primera carga\nGET /api/meta\n(config + rango válido de los 3 escenarios)", **process_style)
    c.node("C", "Cambia escenario, fecha u hora\nGET /api/predict?ts=…&escenario=…", **process_style)
    c.node("SL", "Mueve los umbrales\nde frío / calor", **process_style)
    c.edge("SEL", "B", style="invis")
    c.edge("B", "C", style="invis")
    c.edge("C", "SL", style="invis")

# ---------------------------------------------------------------------
# 3 · backend
# ---------------------------------------------------------------------
with dot.subgraph(name="cluster_3") as c:
    c.attr(label="③  BACKEND — FastAPI (api.py)", fontsize="15", fontcolor="#37474F",
           style="rounded,dashed", color="#B0BEC5", margin="18")
    c.node("D", "timestamp → índice horario j\nrecortado al rango válido del escenario elegido", **process_style)
    c.node("E", "Caché LRU\n_predict_cached(j, escenario)", **db_style)

# ---------------------------------------------------------------------
# 4 · modelo (multi-horizonte)
# ---------------------------------------------------------------------
with dot.subgraph(name="cluster_4") as c:
    c.attr(label="④  MODELO — DCRNN, un checkpoint por horizonte (model.py + data_utils.py)",
           fontsize="15", fontcolor="#37474F", style="rounded,dashed", color="#B0BEC5", margin="18")
    c.node("F", "Artefactos compartidos\ngrafo REMMAQ (6 estaciones) · scaler\nmismos para los 3 escenarios", **db_style)

    with c.subgraph(name="cluster_4a") as g:
        g.attr(label="según escenario elegido — mismo hidden=64, K=2\n ", fontsize="12.5", fontcolor="#607D8B",
               style="dashed", color="#CFD8DC", margin="20")
        g.node("G1", "①  CORTO\ndcrnn_corto_h3_best.pt\nventana 24h  →  +3h", **model_style)
        g.node("G2", "②  MEDIO\ndcrnn_medio_h48_best.pt\nventana 96h  →  +48h", **model_style)
        g.node("G3", "③  LARGO\ndcrnn_largo_h72_best.pt\nventana 168h  →  +72h", **model_style)
        g.edge("G1", "G2", style="invis")
        g.edge("G2", "G3", style="invis")

    c.node("H", "Desnormaliza\nescala del modelo → °C", **process_style)

# ---------------------------------------------------------------------
# 5 · alerta y visualización
# ---------------------------------------------------------------------
with dot.subgraph(name="cluster_5") as c:
    c.attr(label="⑤  ALERTA Y VISUALIZACIÓN — todo en el navegador", fontsize="15", fontcolor="#37474F",
           style="rounded,dashed", color="#B0BEC5", margin="18")
    c.node("I", "≶   pred vs. umbrales", **decision_style)
    c.node("J", "Mapa Leaflet\ncolor por T°(℃) · aro rojo si alerta", **output_style)
    c.node("K", "Banner + medidor\n“N estación(es) en alerta”", **output_style)
    c.node("M", "Tabla por estación\npredicho · real · MAE histórico", **output_style)
    c.node("N", "Traza horaria\neje adaptado a la ventana (24h / 96h / 168h)", **output_style)
    c.edge("J", "K", style="invis")
    c.edge("K", "M", style="invis")
    c.edge("M", "N", style="invis")

# ---------------------------------------------------------------------
# conexiones
# ---------------------------------------------------------------------
dot.edge("A", "SEL")
dot.edge("A", "C")
dot.edge("A", "SL")
dot.edge("SEL", "C", label="  dispara nueva\nconsulta", fontcolor="#7a5c00", color="#F57F17", style="dashed")

dot.edge("C", "D")
dot.edge("D", "E")
dot.edge("E", "I", label="  hay caché", fontsize="11")
dot.edge("E", "F", label="  sin caché")
dot.edge("F", "G1")
dot.edge("F", "G2")
dot.edge("F", "G3")
dot.edge("G1", "H")
dot.edge("G2", "H")
dot.edge("G3", "H")
dot.edge("H", "I", label="  guarda en caché\ny responde")

dot.edge("SL", "I", label="  ⚡ instantáneo: sin backend,\nreutiliza la última predicción",
         style="dashed", color="#F57F17", fontcolor="#7a5c00", penwidth="1.8", constraint="false")

dot.edge("I", "J")
dot.edge("I", "K")
dot.edge("I", "M")
dot.edge("I", "N")

# ---------------------------------------------------------------------
# leyenda
# ---------------------------------------------------------------------
with dot.subgraph(name="cluster_legend") as c:
    c.attr(label="LEYENDA", fontsize="13", fontcolor="#37474F", style="rounded,solid", color="#B0BEC5",
           margin="16", labelloc="t")
    c.node("leg1", "Entrada del usuario", **input_style)
    c.node("leg2", "Proceso (front / backend)", **process_style)
    c.node("leg3", "Datos / caché", **db_style)
    c.node("leg4", "Modelo entrenado", **model_style)
    c.node("leg5", "Salida al usuario", **output_style)
    c.node("leg6", "Nuevo en esta versión", **new_style)
    c.edge("leg1", "leg2", style="invis")
    c.edge("leg2", "leg3", style="invis")
    c.edge("leg3", "leg4", style="invis")
    c.edge("leg4", "leg5", style="invis")
    c.edge("leg5", "leg6", style="invis")

dot.format = "png"
out = dot.render(SALIDA, cleanup=True)
print("Generado:", out)
