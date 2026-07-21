# -*- coding: utf-8 -*-
"""
Mapa de estaciones — Quito, Ecuador (estilo paper)
Basemap topográfico + área de estudio (DMQ) + inset Ecuador + brújula + marco con coordenadas.
"""

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

# ======================================================================
# 1. DATOS
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

lugares = [
    {"nombre": "Quito",          "lat": -0.2200, "lon": -78.5125, "tipo": "ciudad"},
    {"nombre": "Calderón",       "lat": -0.0980, "lon": -78.4280, "tipo": "parroquia"},
    {"nombre": "Pomasqui",       "lat": -0.0330, "lon": -78.4540, "tipo": "parroquia"},
    {"nombre": "Cumbayá",        "lat": -0.2050, "lon": -78.4380, "tipo": "parroquia"},
    {"nombre": "Conocoto",       "lat": -0.2890, "lon": -78.4840, "tipo": "parroquia"},
    {"nombre": "Sangolquí",      "lat": -0.3310, "lon": -78.4500, "tipo": "parroquia"},
    {"nombre": "Llano Chico",    "lat": -0.1170, "lon": -78.4330, "tipo": "parroquia"},
    {"nombre": "Nayón",          "lat": -0.1600, "lon": -78.4310, "tipo": "parroquia"},
    {"nombre": "Chillogallo",    "lat": -0.2850, "lon": -78.5380, "tipo": "parroquia"},
    {"nombre": "Solanda",        "lat": -0.2630, "lon": -78.5200, "tipo": "parroquia"},
    {"nombre": "Villaflora",     "lat": -0.2440, "lon": -78.5130, "tipo": "parroquia"},
    {"nombre": "Guápulo",        "lat": -0.2010, "lon": -78.4710, "tipo": "parroquia"},
    {"nombre": "Puembo",         "lat": -0.1860, "lon": -78.3740, "tipo": "parroquia"},
    {"nombre": "San Carlos",     "lat": -0.1380, "lon": -78.4940, "tipo": "parroquia"},
    {"nombre": "Pichincha",      "lat": -0.1300, "lon": -78.5700, "tipo": "provincia"},
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

# DMQ boundary
DATA_DIR = Path(__file__).parent / "data"
dmq = gpd.read_file(DATA_DIR / "dmq_boundary.geojson").to_crs(epsg=3857)

# Ecuador boundary
ecuador = gpd.read_file(DATA_DIR / "ecuador_boundary.geojson")

# ======================================================================
# 2. BASEMAP
# ======================================================================
def intentar_basemap(ax, zoom=13):
    try:
        import contextily as cx
        import xyzservices.providers as xyz
        cx.add_basemap(ax, source=xyz.Esri.WorldTopoMap, zoom=zoom)
        return True
    except Exception as e:
        print(f"Aviso: Sin teselas: {e}")
        ax.set_facecolor("#E8E6E0")
        return False

# ======================================================================
# 3. CONFIGURACIÓN
# ======================================================================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

DPI = 400
COLOR_PIN      = "#E53935"
COLOR_STICK    = "#555555"
COLOR_DMQ_FILL = "#E8919180"   # rosado semitransparente
COLOR_DMQ_EDGE = "#B71C1C"

# ======================================================================
# 4. MAPA PRINCIPAL
# ======================================================================
fig = plt.figure(figsize=(10, 11), dpi=DPI)

# Eje principal con marco y gridlines (no set_axis_off)
ax = fig.add_axes([0.10, 0.08, 0.75, 0.82])

margen = 4500
ax.set_xlim(gdf_3857.geometry.x.min() - margen, gdf_3857.geometry.x.max() - 500)
ax.set_ylim(gdf_3857.geometry.y.min() - margen, gdf_3857.geometry.y.max() + margen)

intentar_basemap(ax, zoom=13)

# --- Área de estudio (DMQ) ---
dmq.plot(ax=ax, facecolor=COLOR_DMQ_FILL, edgecolor=COLOR_DMQ_EDGE,
         linewidth=1.2, zorder=2)

# --- Nombres de parroquias ---
xlim = ax.get_xlim()
ylim = ax.get_ylim()
for _, row in gdf_lug.iterrows():
    x, y = row.geometry.x, row.geometry.y
    if xlim[0] <= x <= xlim[1] and ylim[0] <= y <= ylim[1]:
        if row.tipo == "ciudad":
            fs, fw, fc = 11, "bold", "#333333"
        elif row.tipo == "provincia":
            fs, fw, fc = 10, "normal", "#777777"
        else:
            fs, fw, fc = 8, "normal", "#444444"
        ax.text(x, y, row.nombre, fontsize=fs, fontweight=fw, color=fc,
                fontstyle="italic", ha="center", va="center", zorder=3,
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white", alpha=0.85)])

# --- Pins (círculo + palo) ---
STICK_LEN = 1200
STICK_W   = 1.8

for _, row in gdf_3857.iterrows():
    px, py = row.geometry.x, row.geometry.y
    ax.plot([px, px], [py - STICK_LEN, py], color=COLOR_STICK,
            linewidth=STICK_W, solid_capstyle="round", zorder=4)

ax.scatter(gdf_3857.geometry.x, gdf_3857.geometry.y,
           s=200, marker="o", color=COLOR_PIN, edgecolors=COLOR_PIN,
           linewidth=0.5, zorder=5)

# --- Etiquetas de estaciones ---
for _, row in gdf_3857.iterrows():
    ax.annotate(row.nombre.upper(),
                xy=(row.geometry.x, row.geometry.y),
                xytext=(10, 12), textcoords="offset points",
                fontsize=7.5, fontweight="bold", color="#1A1A1A",
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
                zorder=7)

# ======================================================================
# 5. MARCO CON COORDENADAS (lat/lon)
# ======================================================================
# Convertir límites de Web Mercator a lat/lon para etiquetas
from pyproj import Transformer
t_inv = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

x_left, x_right = ax.get_xlim()
y_bottom, y_top = ax.get_ylim()

# Etiquetar en grados
lon_left, lat_bottom = t_inv.transform(x_left, y_bottom)
lon_right, lat_top = t_inv.transform(x_right, y_top)

# Crear ticks en intervalos redondos de 0.1°
lon_ticks_deg = np.arange(np.ceil(lon_left * 10) / 10, lon_right, 0.1)
lat_ticks_deg = np.arange(np.ceil(lat_bottom * 10) / 10, lat_top, 0.1)

t_fwd = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

# Eje X: longitudes
xtick_pos = [t_fwd.transform(lon, lat_bottom)[0] for lon in lon_ticks_deg]
xtick_labels = [f"{abs(lon):.1f}°{'W' if lon < 0 else 'E'}" for lon in lon_ticks_deg]
ax.set_xticks(xtick_pos)
ax.set_xticklabels(xtick_labels, fontsize=7)

# Eje Y: latitudes
ytick_pos = [t_fwd.transform(lon_left, lat)[1] for lat in lat_ticks_deg]
ytick_labels = [f"{abs(lat):.1f}°{'S' if lat < 0 else 'N'}" for lat in lat_ticks_deg]
ax.set_yticks(ytick_pos)
ax.set_yticklabels(ytick_labels, fontsize=7)

ax.tick_params(axis="both", direction="in", length=4, width=0.8)
for spine in ax.spines.values():
    spine.set_linewidth(1.0)
    spine.set_edgecolor("#333333")

# ======================================================================
# 6. BRÚJULA / FLECHA NORTE
# ======================================================================
ax_north = fig.add_axes([0.10, 0.82, 0.06, 0.08], frameon=False)
ax_north.set_xlim(-1, 1)
ax_north.set_ylim(-1, 1.5)
ax_north.set_xticks([])
ax_north.set_yticks([])

# Flecha
ax_north.annotate("", xy=(0, 1.3), xytext=(0, -0.5),
                  arrowprops=dict(arrowstyle="-|>", color="black", lw=2))
ax_north.text(0, 1.5, "N", ha="center", va="bottom", fontsize=13, fontweight="bold")

# ======================================================================
# 7. BARRA DE ESCALA
# ======================================================================
km_per_m = 1 / 1000
scale_km = 10
scale_m = scale_km * 1000
sx0 = x_left + (x_right - x_left) * 0.05
sy0 = y_bottom + (y_top - y_bottom) * 0.04

ax.plot([sx0, sx0 + scale_m], [sy0, sy0], color="black", linewidth=3,
        solid_capstyle="butt", zorder=10)
ax.plot([sx0, sx0], [sy0 - 300, sy0 + 300], color="black", linewidth=1.5, zorder=10)
ax.plot([sx0 + scale_m, sx0 + scale_m], [sy0 - 300, sy0 + 300], color="black", linewidth=1.5, zorder=10)
# Mid tick
ax.plot([sx0 + scale_m/2, sx0 + scale_m/2], [sy0 - 200, sy0 + 200], color="black", linewidth=1, zorder=10)

ax.text(sx0, sy0 + 500, "0", fontsize=6.5, ha="center", va="bottom", zorder=10)
ax.text(sx0 + scale_m/2, sy0 + 500, f"{scale_km//2}", fontsize=6.5, ha="center", va="bottom", zorder=10)
ax.text(sx0 + scale_m, sy0 + 500, f"{scale_km} km", fontsize=6.5, ha="center", va="bottom", zorder=10)

# ======================================================================
# 8. LEYENDA
# ======================================================================
leg_items = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_PIN,
           markeredgecolor=COLOR_PIN, markersize=8, label="REMMAQ"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=COLOR_DMQ_FILL,
           markeredgecolor=COLOR_DMQ_EDGE, markersize=8, label="Study Area"),
]
ax.legend(handles=leg_items, loc="lower right", fontsize=7,
          frameon=True, framealpha=0.9, edgecolor="0.5")

# ======================================================================
# 9. INSET: ECUADOR
# ======================================================================
ax_inset = fig.add_axes([0.62, 0.08, 0.24, 0.22])
ax_inset.set_facecolor("#DCEBF7")

ecuador.plot(ax=ax_inset, facecolor="#E8E6E0", edgecolor="#888888", linewidth=0.5)

# Rectángulo de la zona de estudio
lon_ext = [lon_left, lon_right, lon_right, lon_left, lon_left]
lat_ext = [lat_bottom, lat_bottom, lat_top, lat_top, lat_bottom]
ax_inset.plot(lon_ext, lat_ext, color=COLOR_DMQ_EDGE, linewidth=1.5, zorder=5)
ax_inset.fill(lon_ext, lat_ext, color=COLOR_DMQ_FILL, zorder=4)

ax_inset.set_xlim(-81.5, -75)
ax_inset.set_ylim(-5.2, 1.5)
ax_inset.set_xticks([])
ax_inset.set_yticks([])
for spine in ax_inset.spines.values():
    spine.set_linewidth(0.8)
    spine.set_edgecolor("#555")

ax_inset.text(-78.5, -1.5, "Quito", fontsize=8, fontweight="bold", ha="center",
              color="#333", zorder=6,
              path_effects=[pe.withStroke(linewidth=2, foreground="white")])
ax_inset.text(-78.0, -4.0, "Ecuador", fontsize=10, fontweight="bold", fontstyle="italic",
              ha="center", color="#555", zorder=6)

# ======================================================================
# 10. EXPORTAR
# ======================================================================
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)
fig.savefig(OUT_DIR / "mapa_quito_final.png", dpi=DPI, bbox_inches="tight", facecolor="white")
fig.savefig(OUT_DIR / "mapa_quito_final.pdf", bbox_inches="tight", facecolor="white")
print("Mapa generado en:", OUT_DIR.resolve())
