"use strict";

const NOMBRE_LEGIBLE = { LosChillos: "Los Chillos", ElCamal: "El Camal" };
const legible = (est) => NOMBRE_LEGIBLE[est] || est;

const ICON_SUN = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4.2"/><path d="M12 2.5v2.2M12 19.3v2.2M4.2 4.2l1.6 1.6M18.2 18.2l1.6 1.6M2.5 12h2.2M19.3 12h2.2M4.2 19.8l1.6-1.6M18.2 5.8l1.6-1.6"/></svg>`;
const ICON_MOON = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M20 14.5A8.5 8.5 0 1 1 9.5 4a6.8 6.8 0 0 0 10.5 10.5Z"/></svg>`;

const ESCENARIO_LABEL = {
  corto: "corto plazo",
  medio: "medio plazo",
  largo: "largo plazo",
};

const state = {
  meta: null,
  escenario: "corto",
  lastPrediction: null,
  selectedStation: null,
  openStations: new Set(),
  map: null,
  markers: new Map(),
  pinMarker: null,
  tracePoints: [],
};

const $ = (sel) => document.querySelector(sel);
const escMeta = () => state.meta.escenarios[state.escenario];

function tempToColor(t, vmin, vmax) {
  const frac = vmax <= vmin ? 0 : Math.max(0, Math.min(1, (t - vmin) / (vmax - vmin)));
  const cold = [37, 99, 235], mid = [234, 179, 8], hot = [220, 38, 38];
  let a, b, f;
  if (frac < 0.5) { f = frac / 0.5; [a, b] = [cold, mid]; }
  else { f = (frac - 0.5) / 0.5; [a, b] = [mid, hot]; }
  const rgb = a.map((c, i) => Math.round(c + (b[i] - c) * f));
  return `rgb(${rgb.join(",")})`;
}

function estadoDe(pred, frio, calor) {
  if (pred < frio) return "Frío";
  if (pred > calor) return "Calor";
  return "Normal";
}

function chipClass(estado) {
  if (estado === "Frío") return "chip-frio";
  if (estado === "Calor") return "chip-calor";
  return "chip-normal";
}

function fmtTs(iso) {
  const d = new Date(iso);
  return d.toLocaleString("es-EC", {
    day: "2-digit", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit",
  });
}

// ---------------------------------------------------------------------
// tema (claro/oscuro), aplicado ya mismo para evitar parpadeos
// ---------------------------------------------------------------------
function currentTheme() {
  const saved = localStorage.getItem("theme");
  if (saved) return saved;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  const icon = $("#theme-icon"), label = $("#theme-label");
  if (icon) icon.innerHTML = theme === "dark" ? ICON_SUN : ICON_MOON;
  if (label) label.textContent = theme === "dark" ? "Modo claro" : "Modo oscuro";
  if (state.lastPrediction) renderTrace(); // el canvas lee custom properties de color
}

function initTheme() {
  applyTheme(currentTheme());
  $("#theme-toggle").addEventListener("click", () => {
    const next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
    localStorage.setItem("theme", next);
    applyTheme(next);
  });
}

// ---------------------------------------------------------------------
// carga inicial
// ---------------------------------------------------------------------
async function init() {
  const meta = await (await fetch("/api/meta")).json();
  state.meta = meta;

  const inputEscenario = $("#input-escenario");
  inputEscenario.value = state.escenario;

  applyEscenarioBounds();

  const inputFrio = $("#input-frio");
  const inputCalor = $("#input-calor");
  inputFrio.value = meta.default_thresholds.frio.toFixed(1);
  inputCalor.value = meta.default_thresholds.calor.toFixed(1);
  $("#frio-val").textContent = `${inputFrio.value} °C`;
  $("#calor-val").textContent = `${inputCalor.value} °C`;

  const select = $("#input-estacion");
  select.innerHTML = meta.stations.map((e) => `<option value="${e}">${legible(e)}</option>`).join("");
  state.selectedStation = meta.stations[0];
  state.openStations.add(meta.stations[0]);

  renderEscenarioTexts();
  renderMethodology();
  renderAboutList();
  renderStatList();
  initMap(meta);

  const inputFecha = $("#input-fecha");
  const inputHora = $("#input-hora");

  inputEscenario.addEventListener("change", onEscenarioChange);
  inputFecha.addEventListener("change", onTimeChange);
  inputHora.addEventListener("input", () => {
    $("#hora-val").textContent = String(inputHora.value).padStart(2, "0") + ":00";
  });
  inputHora.addEventListener("change", onTimeChange);
  inputFrio.addEventListener("input", onThresholdChange);
  inputCalor.addEventListener("input", onThresholdChange);
  select.addEventListener("change", () => selectStation(select.value));

  $("#rows-container").addEventListener("toggle", onRowToggle, true);
  $("#rows-container").addEventListener("click", (ev) => {
    const nameEl = ev.target.closest(".col-name");
    if (nameEl) selectStation(nameEl.closest(".row").dataset.estacion);
  });

  attachTraceHover();

  await onTimeChange();
}

// ---------------------------------------------------------------------
// horizonte de pronóstico (corto/medio/largo): cada uno tiene su propio
// checkpoint DCRNN, su propia ventana de entrada y su propio rango de
// fechas válido (una ventana más larga recorta más al inicio del split).
// ---------------------------------------------------------------------
function applyEscenarioBounds() {
  const { valid_range } = escMeta();
  const minD = valid_range.min.slice(0, 10);
  const maxD = valid_range.max.slice(0, 10);
  const maxHora = new Date(valid_range.max).getHours();

  const inputFecha = $("#input-fecha");
  inputFecha.min = minD; inputFecha.max = maxD;
  if (!inputFecha.value || inputFecha.value < minD || inputFecha.value > maxD) {
    inputFecha.value = maxD;
  }

  const inputHora = $("#input-hora");
  inputHora.value = maxHora;
  $("#hora-val").textContent = String(maxHora).padStart(2, "0") + ":00";

  $("#rango-hint").textContent =
    `Rango disponible: ${fmtTs(valid_range.min)} → ${fmtTs(valid_range.max)}`;
}

function renderEscenarioTexts() {
  const { config } = escMeta();
  const lbl = ESCENARIO_LABEL[state.escenario];
  $("#scenario-pill").textContent = `Escenario: ${lbl} (+${config.horizon} h)`;
  $("#col-real-head").textContent = `Real (+${config.horizon}h)`;
  $("#historico-note").textContent =
    `Línea: temperatura observada (últimas ${config.seq_len}h, entrada del modelo). ` +
    `Punto rojo: predicción del DCRNN a ${config.horizon}h del último dato observado, ` +
    `comparable contra el valor real porque el dato es histórico (test). Pasa el ` +
    `mouse sobre la serie para ver el valor exacto de cada hora.`;
}

async function onEscenarioChange() {
  state.escenario = $("#input-escenario").value;
  applyEscenarioBounds();
  renderEscenarioTexts();
  renderMethodology();
  renderAboutList();
  renderStatList();
  await onTimeChange();
}

function onThresholdChange() {
  $("#frio-val").textContent = `${(+$("#input-frio").value).toFixed(1)} °C`;
  $("#calor-val").textContent = `${(+$("#input-calor").value).toFixed(1)} °C`;
  // Los umbrales no llaman al backend: se recalculan en el navegador con la
  // última predicción ya cargada (state.lastPrediction), por eso mover estos
  // sliders es instantáneo aunque cambiar la hora sí requiere una llamada.
  render();
}

function onRowToggle(ev) {
  const est = ev.target.dataset.estacion;
  if (!est) return;
  if (ev.target.open) state.openStations.add(est);
  else state.openStations.delete(est);
}

function selectStation(est) {
  state.selectedStation = est;
  $("#input-estacion").value = est;
  render();
}

// ---------------------------------------------------------------------
// hora actual -> /api/predict
// ---------------------------------------------------------------------
async function onTimeChange() {
  const fecha = $("#input-fecha").value;
  const hora = String($("#input-hora").value).padStart(2, "0");
  const ts = `${fecha}T${hora}:00:00`;

  $("#subhead-title").textContent = "Consultando el modelo…";
  const url = `/api/predict?ts=${encodeURIComponent(ts)}&escenario=${state.escenario}`;
  const resp = await fetch(url);
  if (!resp.ok) {
    $("#subhead-title").textContent = "Error al consultar el modelo.";
    return;
  }
  state.lastPrediction = await resp.json();
  render();
}

// ---------------------------------------------------------------------
// render (puro: no llama a la red, solo usa state.lastPrediction + umbrales)
// ---------------------------------------------------------------------
function render() {
  if (!state.lastPrediction) return;
  const { meta, lastPrediction: pred } = state;
  const frio = +$("#input-frio").value;
  const calor = +$("#input-calor").value;

  $("#subhead-title").textContent =
    `Hora actual (simulada): ${fmtTs(pred.t_actual)} → pronóstico para ${fmtTs(pred.t_pred)}`;

  const vmin = meta.default_thresholds.frio, vmax = meta.default_thresholds.calor;
  const estados = pred.stations.map((s) => estadoDe(s.pred, frio, calor));
  const nAlertas = estados.filter((e) => e !== "Normal").length;

  const banner = $("#alert-banner");
  if (nAlertas === 0) {
    banner.className = "alert-banner ok";
    banner.textContent = "Sin alertas activas en ninguna estación.";
  } else {
    banner.className = "alert-banner alert";
    banner.textContent = `${nAlertas} estación(es) en alerta.`;
  }

  $("#gauge").style.background = `conic-gradient(${nAlertas === 0 ? "var(--good)" : "var(--warn)"} 360deg, var(--line) 0)`;
  $("#gauge-num").textContent = String(nAlertas);
  $("#gauge-title").textContent = nAlertas === 0 ? "Sin alertas" : "Alertas activas";
  $("#gauge-title").style.color = nAlertas === 0 ? "var(--good)" : "var(--warn)";
  $("#gauge-desc").textContent = nAlertas === 0
    ? `Ninguna estación supera los umbrales configurados (${frio.toFixed(1)} °C / ${calor.toFixed(1)} °C).`
    : `${nAlertas} de ${pred.stations.length} estaciones fuera del rango ${frio.toFixed(1)}–${calor.toFixed(1)} °C.`;

  renderRows(pred, estados, vmin, vmax);
  renderMap(pred, estados, vmin, vmax);
  renderTrace();
}

function renderRows(pred, estados) {
  const { metrics_por_estacion, config } = escMeta();
  const html = pred.stations.map((s, i) => {
    const estado = estados[i];
    const m = metrics_por_estacion[s.estacion] || {};
    const mae = m.MAE ?? 0;
    const err = Math.abs(s.pred - s.actual);
    const isOpen = state.openStations.has(s.estacion);
    const isSel = s.estacion === state.selectedStation;
    return `
      <details class="row${isSel ? " selected" : ""}" data-estacion="${s.estacion}" ${isOpen ? "open" : ""}>
        <summary>
          <span class="col-name">${legible(s.estacion)}</span>
          <span class="col-temp">${s.pred.toFixed(1)}°</span>
          <span class="chip ${chipClass(estado)}">${estado}</span>
          <span class="col-error">
            <span class="bar-track"><span class="bar-fill" style="width:${Math.min(100, mae / 1.0 * 100)}%"></span></span>
            <span class="val">${mae.toFixed(2)}°</span>
          </span>
          <span class="col-real">${s.actual.toFixed(1)}°</span>
          <span class="chevron">›</span>
        </summary>
        <div class="row-detail">
          <div class="detail-grid">
            <div class="detail-item"><span class="k">Predicho (+${config.horizon}h)</span><span class="v">${s.pred.toFixed(2)} °C</span></div>
            <div class="detail-item"><span class="k">Real observado</span><span class="v">${s.actual.toFixed(2)} °C</span></div>
            <div class="detail-item"><span class="k">Error (este instante)</span><span class="v">${err.toFixed(2)} °C</span></div>
            <div class="detail-item"><span class="k">MAE histórico (test)</span><span class="v">${(m.MAE ?? 0).toFixed(2)} °C</span></div>
            <div class="detail-item"><span class="k">RMSE histórico</span><span class="v">${(m.RMSE ?? 0).toFixed(2)} °C</span></div>
            <div class="detail-item"><span class="k">R² histórico</span><span class="v">${(m.R2 ?? 0).toFixed(3)}</span></div>
            <div class="detail-item"><span class="k">Ventana de entrada</span><span class="v">${config.seq_len} h</span></div>
            <div class="detail-item"><span class="k">Horizonte</span><span class="v">${config.horizon} h</span></div>
            <div class="detail-item"><span class="k">Última hora observada</span><span class="v">${fmtTs(pred.t_actual)}</span></div>
          </div>
        </div>
      </details>`;
  }).join("");
  $("#rows-container").innerHTML = html;
}

// ---------------------------------------------------------------------
// mapa (Leaflet + OpenStreetMap): un círculo de color por estación +
// un pin distintivo sobre la estación seleccionada (fila abierta o
// elegida en "Trazabilidad por estación").
// ---------------------------------------------------------------------
function initMap(meta) {
  const lats = meta.coords.map((c) => c.lat);
  const lons = meta.coords.map((c) => c.lon);
  const centerLat = (Math.min(...lats) + Math.max(...lats)) / 2;
  const centerLon = (Math.min(...lons) + Math.max(...lons)) / 2;

  const map = L.map("map", { zoomControl: false }).setView([centerLat, centerLon], 11);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
    maxZoom: 18,
  }).addTo(map);
  L.control.zoom({ position: "topright" }).addTo(map);

  meta.coords.forEach((c) => {
    const marker = L.circleMarker([c.lat, c.lon], {
      radius: 9, weight: 2, color: "#ffffff", fillColor: "#8a94a0", fillOpacity: 0.9,
    }).addTo(map).bindPopup("");
    marker.on("click", () => selectStation(c.estacion));
    state.markers.set(c.estacion, marker);
  });

  state.map = map;
}

function renderMap(pred, estados, vmin, vmax) {
  const { config } = escMeta();
  pred.stations.forEach((s, i) => {
    const marker = state.markers.get(s.estacion);
    if (!marker) return;
    marker.setStyle({
      fillColor: tempToColor(s.pred, vmin, vmax),
      color: estados[i] !== "Normal" ? "#dc2626" : "#ffffff",
      weight: estados[i] !== "Normal" ? 3 : 2,
    });
    marker.setPopupContent(
      `<b>${legible(s.estacion)}</b><br>Predicho (+${config.horizon}h): ${s.pred.toFixed(1)} °C<br>` +
      `Real (+${config.horizon}h): ${s.actual.toFixed(1)} °C<br>Estado: ${estados[i]}`
    );
  });

  const coord = state.meta.coords.find((c) => c.estacion === state.selectedStation);
  const sel = pred.stations.find((s) => s.estacion === state.selectedStation);
  if (!coord || !sel) return;
  if (!state.pinMarker) {
    state.pinMarker = L.marker([coord.lat, coord.lon]).addTo(state.map);
  } else {
    state.pinMarker.setLatLng([coord.lat, coord.lon]);
  }
  state.pinMarker.bindPopup(
    `<b>${legible(sel.estacion)}</b><br>Estación seleccionada<br>Predicho (+${config.horizon}h): ${sel.pred.toFixed(1)} °C`
  );
}

// ---------------------------------------------------------------------
// tarjetas estáticas (dependen del escenario seleccionado)
// ---------------------------------------------------------------------
function renderStatList() {
  const m = escMeta().metrics_global;
  const items = [
    ["MAE global", `${m.MAE.toFixed(2)} °C`],
    ["RMSE global", `${m.RMSE.toFixed(2)} °C`],
    ["R² global", m.R2.toFixed(3)],
    ["MAPE global", `${m.MAPE.toFixed(1)} %`],
  ];
  $("#stat-list").innerHTML = items.map(([lbl, num]) =>
    `<li><span class="lbl">${lbl}</span><span class="num">${num}</span></li>`).join("");
}

function renderMethodology() {
  const c = escMeta().config;
  $("#methodology").textContent =
    `DCRNN modela la red de estaciones REMMAQ como una señal sobre un grafo: cada ` +
    `estación es un nodo, y una celda recurrente con convoluciones de difusión ` +
    `(orden de Chebyshev K=${c.K}) captura cómo la temperatura se propaga entre ` +
    `estaciones vecinas a lo largo del tiempo. Se evaluó frente a A3T-GCN, la ` +
    `familia ARIMA/SARIMA/SARIMAX y ClimaX (modelo fundacional de clima con ` +
    `fine-tuning) en los 3 horizontes de pronóstico (corto +3h, medio +48h y ` +
    `largo +72h); en el horizonte corto resultó el mejor modelo. El split de ` +
    `test es una ventana cronológica posterior nunca vista en entrenamiento, ` +
    `para evitar fuga temporal.`;
}

function renderAboutList() {
  const c = escMeta().config;
  const items = [
    ["Arquitectura", "DCRNN"],
    ["Estado oculto", `hidden=${c.hidden}`],
    ["Orden de difusión", `K=${c.K}`],
    ["Ventana → horizonte", `${c.seq_len}h → ${c.horizon}h`],
    ["Tamaño de lote", `${c.batch}`],
    ["Semilla", `${c.seed}`],
    ["Épocas entrenadas", `${c.epochs_corridos}`],
  ];
  $("#about-list").innerHTML = items.map(([lbl, v]) =>
    `<li><span>${lbl}</span><b>${v}</b></li>`).join("");
}

// ---------------------------------------------------------------------
// gráfico de trazabilidad (canvas, sin librerías externas) + tooltip hover
// ---------------------------------------------------------------------
function renderTrace(hoverIdx = -1) {
  if (!state.lastPrediction) return;
  const s = state.lastPrediction.stations.find((x) => x.estacion === state.selectedStation);
  if (!s) return;

  const canvas = $("#trace-canvas");
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.parentElement.clientWidth - 24;
  const cssH = 280;
  canvas.style.width = cssW + "px";
  canvas.style.height = cssH + "px";
  canvas.width = cssW * dpr;
  canvas.height = cssH * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);

  const pad = { l: 42, r: 14, t: 14, b: 26 };
  const w = cssW - pad.l - pad.r, h = cssH - pad.t - pad.b;

  const histPoints = s.history.map((p) => ({ t: new Date(p.t), v: p.v, kind: "obs" }));
  const tPred = new Date(state.lastPrediction.t_pred);
  const predPoint = { t: tPred, v: s.pred, kind: "pred" };
  const actualPoint = { t: tPred, v: s.actual, kind: "real" };

  const allV = histPoints.map((p) => p.v).concat([s.pred, s.actual]);
  const vMin = Math.min(...allV) - 1, vMax = Math.max(...allV) + 1;
  const tMin = histPoints[0].t.getTime(), tMax = tPred.getTime();

  const xOf = (t) => pad.l + ((t - tMin) / (tMax - tMin)) * w;
  const yOf = (v) => pad.t + (1 - (v - vMin) / (vMax - vMin)) * h;

  const styles = getComputedStyle(document.documentElement);
  const line = styles.getPropertyValue("--line").trim();
  const inkFaint = styles.getPropertyValue("--ink-faint").trim();
  const inkMuted = styles.getPropertyValue("--ink-muted").trim();
  const bgRaised = styles.getPropertyValue("--bg-raised").trim() || "#fff";

  // grid + eje Y
  ctx.strokeStyle = line; ctx.lineWidth = 1; ctx.font = "11px " + getComputedStyle(document.body).fontFamily;
  ctx.fillStyle = inkFaint; ctx.textBaseline = "middle";
  const ySteps = 4;
  for (let i = 0; i <= ySteps; i++) {
    const v = vMin + (i / ySteps) * (vMax - vMin);
    const y = yOf(v);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(cssW - pad.r, y); ctx.stroke();
    ctx.textAlign = "right";
    ctx.fillText(v.toFixed(0) + "°", pad.l - 8, y);
  }

  // eje X: ~12 marcas repartidas en toda la ventana, sea de 24h (corto)
  // o de varios días (medio/largo) — con fecha si la ventana pasa de un día.
  const multiDia = (tMax - tMin) > 26 * 3600 * 1000;
  const xStep = Math.max(1, Math.round(histPoints.length / 12));
  ctx.textBaseline = "top"; ctx.textAlign = "center"; ctx.fillStyle = inkMuted;
  histPoints.forEach((p, i) => {
    if (i % xStep !== 0) return;
    const x = xOf(p.t.getTime());
    const lbl = multiDia
      ? p.t.toLocaleDateString("es-EC", { day: "2-digit", month: "2-digit" }) +
        " " + p.t.toLocaleTimeString("es-EC", { hour: "2-digit" })
      : p.t.toLocaleTimeString("es-EC", { hour: "2-digit" });
    ctx.fillText(lbl, x, cssH - pad.b + 8);
  });

  // línea observada
  ctx.beginPath();
  histPoints.forEach((p, i) => {
    const x = xOf(p.t.getTime()), y = yOf(p.v);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.lineTo(xOf(actualPoint.t.getTime()), yOf(actualPoint.v));
  ctx.strokeStyle = "#2563eb"; ctx.lineWidth = 2; ctx.lineJoin = "round"; ctx.stroke();

  // punto real (+3h)
  ctx.beginPath();
  ctx.arc(xOf(actualPoint.t.getTime()), yOf(actualPoint.v), 4, 0, Math.PI * 2);
  ctx.fillStyle = "#2563eb"; ctx.fill();

  // punto predicho (+3h)
  ctx.beginPath();
  ctx.arc(xOf(predPoint.t.getTime()), yOf(predPoint.v), 6, 0, Math.PI * 2);
  ctx.fillStyle = "#dc2626"; ctx.fill();
  ctx.strokeStyle = bgRaised; ctx.lineWidth = 2; ctx.stroke();

  // puntos con coords en px (space CSS) para hit-testing del hover
  const allPoints = histPoints.concat([actualPoint, predPoint]).map((p) => ({
    ...p, x: xOf(p.t.getTime()), y: yOf(p.v),
  }));
  state.tracePoints = allPoints;

  if (hoverIdx >= 0 && allPoints[hoverIdx]) {
    const hp = allPoints[hoverIdx];
    ctx.beginPath();
    ctx.moveTo(hp.x, pad.t); ctx.lineTo(hp.x, cssH - pad.b);
    ctx.strokeStyle = inkFaint; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
    ctx.stroke(); ctx.setLineDash([]);

    ctx.beginPath();
    ctx.arc(hp.x, hp.y, 5.5, 0, Math.PI * 2);
    ctx.fillStyle = hp.kind === "pred" ? "#dc2626" : "#2563eb";
    ctx.fill();
    ctx.strokeStyle = bgRaised; ctx.lineWidth = 2; ctx.stroke();
  }
}

function attachTraceHover() {
  const canvas = $("#trace-canvas");
  const tooltip = $("#trace-tooltip");
  const kindLbl = { pred: "Predicho (DCRNN)", real: "Real (+3h)", obs: "Observado" };

  canvas.addEventListener("mousemove", (ev) => {
    if (!state.tracePoints.length) return;
    const rect = canvas.getBoundingClientRect();
    const mx = ev.clientX - rect.left;
    let idx = 0, best = Infinity;
    state.tracePoints.forEach((p, i) => {
      const d = Math.abs(p.x - mx);
      if (d < best) { best = d; idx = i; }
    });
    renderTrace(idx);
    const p = state.tracePoints[idx];
    tooltip.innerHTML = `<b>${p.v.toFixed(1)}°C</b><br>${kindLbl[p.kind]} · ` +
      p.t.toLocaleString("es-EC", { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" });
    tooltip.style.left = `${p.x + 12}px`;
    tooltip.style.top = `${p.y + 12}px`;
    tooltip.classList.add("visible");
  });

  canvas.addEventListener("mouseleave", () => {
    tooltip.classList.remove("visible");
    renderTrace(-1);
  });
}

window.addEventListener("resize", () => { if (state.lastPrediction) renderTrace(); });

initTheme();
init();
