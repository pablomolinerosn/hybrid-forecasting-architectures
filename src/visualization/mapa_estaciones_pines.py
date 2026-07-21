# -*- coding: utf-8 -*-
"""
Mapa de ubicaciones de estaciones — Quito, Ecuador
Basemap topográfico + pins estilo Google Maps + nombres de parroquias.
"""

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
import numpy as np
from pathlib import Path

# ======================================================================
# 1. DATOS DE ENTRADA
# ======================================================================
puntos = [
    {"nombre": "Belisario",               "codigo": "Bel", "lat": -0.185170, "lon": -78.495677},
    {"nombre": "Centro",                   "codigo": "Cen", "lat": -0.221428, "lon": -78.513936},
    {"nombre": "Carapungo",                "codigo": "Car", "lat": -0.095393, "lon": -78.449755},
    {"nombre": "Cotocollao",               "codigo": "Cot", "lat": -0.107716, "lon": -78.497268},
    {"nombre": "El Camal",                 "codigo": "Cam", "lat": -0.249970, "lon": -78.510058},
    {"nombre": "Guamaní",                  "codigo": "Gua", "lat": -0.333871, "lon": -78.553583},
    {"nombre": "Los Chillos",              "codigo": "Chi", "lat": -0.297100, "lon": -78.455270},
    {"nombre": "Tumbaco",                  "codigo": "Tum", "lat": -0.215015, "lon": -78.403442},
    {"nombre": "San Antonio de Pichincha", "codigo": "Sap", "lat": -0.008807, "lon": -78.447900},
]

# Parroquias y localidades de referencia (contexto geográfico)
lugares = [
    {"nombre": "Quito",              "lat": -0.2200, "lon": -78.5125, "tipo": "ciudad"},
    {"nombre": "Calderón",           "lat": -0.0980, "lon": -78.4280, "tipo": "parroquia"},
    {"nombre": "Pomasqui",           "lat": -0.0330, "lon": -78.4540, "tipo": "parroquia"},
    {"nombre": "Cumbayá",            "lat": -0.2050, "lon": -78.4380, "tipo": "parroquia"},
    {"nombre": "Conocoto",           "lat": -0.2890, "lon": -78.4840, "tipo": "parroquia"},
    {"nombre": "Sangolquí",          "lat": -0.3310, "lon": -78.4500, "tipo": "parroquia"},
    {"nombre": "Llano Chico",        "lat": -0.1170, "lon": -78.4330, "tipo": "parroquia"},
    {"nombre": "Zámbiza",            "lat": -0.1370, "lon": -78.4370, "tipo": "parroquia"},
    {"nombre": "Nayón",              "lat": -0.1600, "lon": -78.4310, "tipo": "parroquia"},
    {"nombre": "Chillogallo",        "lat": -0.2850, "lon": -78.5380, "tipo": "parroquia"},
    {"nombre": "La Ecuatoriana",     "lat": -0.2720, "lon": -78.5480, "tipo": "parroquia"},
    {"nombre": "Solanda",            "lat": -0.2630, "lon": -78.5200, "tipo": "parroquia"},
    {"nombre": "La Magdalena",       "lat": -0.2380, "lon": -78.5220, "tipo": "parroquia"},
    {"nombre": "Iñaquito",           "lat": -0.1730, "lon": -78.4830, "tipo": "parroquia"},
    {"nombre": "La Carolina",        "lat": -0.1810, "lon": -78.4840, "tipo": "parroquia"},
    {"nombre": "Guápulo",            "lat": -0.2010, "lon": -78.4710, "tipo": "parroquia"},
    {"nombre": "Quitumbe",           "lat": -0.2980, "lon": -78.5500, "tipo": "parroquia"},
    {"nombre": "Turubamba",          "lat": -0.3050, "lon": -78.5440, "tipo": "parroquia"},
    {"nombre": "Amaguaña",           "lat": -0.3710, "lon": -78.5100, "tipo": "parroquia"},
    {"nombre": "Puembo",             "lat": -0.1860, "lon": -78.3740, "tipo": "parroquia"},
    {"nombre": "Pifo",               "lat": -0.2230, "lon": -78.3400, "tipo": "parroquia"},
    {"nombre": "El Quinche",         "lat": -0.1150, "lon": -78.3120, "tipo": "parroquia"},
    {"nombre": "Yaruquí",            "lat": -0.1640, "lon": -78.3270, "tipo": "parroquia"},
    {"nombre": "Llano Grande",       "lat": -0.0880, "lon": -78.4470, "tipo": "parroquia"},
    {"nombre": "San Carlos",         "lat": -0.1380, "lon": -78.4940, "tipo": "parroquia"},
    {"nombre": "Villaflora",         "lat": -0.2440, "lon": -78.5130, "tipo": "parroquia"},
]

gdf = gpd.GeoDataFrame(
    puntos,
    geometry=gpd.points_from_xy([p["lon"] for p in puntos], [p["lat"] for p in puntos]),
    crs="EPSG:4326",
)
gdf_3857 = gdf.to_crs(epsg=3857)

gdf_lug = gpd.GeoDataFrame(
    lugares,
    geometry=gpd.points_from_xy([l["lon"] for l in lugares], [l["lat"] for l in lugares]),
    crs="EPSG:4326",
).to_crs(epsg=3857)

# ======================================================================
# 2. (marcador se dibuja directamente con circle + línea)
# ======================================================================

# ======================================================================
# 3. BASEMAP
# ======================================================================
def intentar_basemap(ax, zoom=14):
    try:
        import contextily as cx
        import xyzservices.providers as xyz
        cx.add_basemap(ax, source=xyz.Esri.WorldTopoMap, zoom=zoom)
        return True
    except Exception as e:
        print(f"Aviso: No se pudieron cargar teselas: {e}")
        print("  → Ejecuta con internet para basemap topográfico.")
        ax.set_facecolor("#E8E6E0")
        geojson = Path(__file__).parent / "data" / "ecuador_boundary.geojson"
        if geojson.exists():
            ecu = gpd.read_file(geojson).to_crs(epsg=3857)
            ecu.boundary.plot(ax=ax, linewidth=0.6, edgecolor="#888", zorder=0)
        return False

# ======================================================================
# 4. FIGURA
# ======================================================================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

DPI = 400
fig, ax = plt.subplots(figsize=(11, 12), dpi=DPI)

margen = 4500
ax.set_xlim(gdf_3857.geometry.x.min() - margen, gdf_3857.geometry.x.max() + margen)
ax.set_ylim(gdf_3857.geometry.y.min() - margen, gdf_3857.geometry.y.max() + margen)

tiene_tiles = intentar_basemap(ax, zoom=14)

# --- Nombres de parroquias / localidades (capa de contexto) ---
xlim = ax.get_xlim()
ylim = ax.get_ylim()

for _, row in gdf_lug.iterrows():
    x, y = row.geometry.x, row.geometry.y
    if xlim[0] <= x <= xlim[1] and ylim[0] <= y <= ylim[1]:
        fs = 11 if row.tipo == "ciudad" else 8.5
        fw = "bold" if row.tipo == "ciudad" else "normal"
        ax.text(
            x, y, row.nombre,
            fontsize=fs, fontweight=fw, color="#444444", fontstyle="italic",
            ha="center", va="center", zorder=3,
            path_effects=[pe.withStroke(linewidth=3, foreground="white", alpha=0.85)],
        )

# --- Pin tipo chincheta (círculo + palo) ---
COLOR_PIN   = "#E53935"   # rojo
COLOR_STICK = "#555555"   # gris oscuro palo

STICK_LEN = 1200   # longitud del palo en metros (Web Mercator)
STICK_W   = 1.8    # grosor del palo en puntos

for _, row in gdf_3857.iterrows():
    px, py = row.geometry.x, row.geometry.y
    # Palo (línea vertical debajo del círculo)
    ax.plot([px, px], [py - STICK_LEN, py], color=COLOR_STICK,
            linewidth=STICK_W, solid_capstyle="round", zorder=4)

# Círculo rojo (encima del palo)
ax.scatter(
    gdf_3857.geometry.x, gdf_3857.geometry.y,
    s=180, marker="o", color=COLOR_PIN, edgecolors=COLOR_PIN,
    linewidth=0.5, zorder=5,
)

# --- Etiquetas de estaciones (nombre completo, en negrita, más grandes) ---
for _, row in gdf_3857.iterrows():
    ax.annotate(
        row.nombre,
        xy=(row.geometry.x, row.geometry.y),
        xytext=(12, 14), textcoords="offset points",
        fontsize=10, fontweight="bold", color="#1A1A1A",
        path_effects=[pe.withStroke(linewidth=3, foreground="white")],
        zorder=7,
    )

ax.set_axis_off()

# ======================================================================
# 5. EXPORTAR
# ======================================================================
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)
fig.savefig(OUT_DIR / "mapa_quito_pins.png", dpi=DPI, bbox_inches="tight", facecolor="white")
# fig.savefig(OUT_DIR / "mapa_quito_pins.pdf", bbox_inches="tight", facecolor="white")
print("Mapa generado en:", OUT_DIR.resolve())