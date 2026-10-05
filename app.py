"""
Dashboard Fintual — Megacap + Large Cap.

Corre local con:  streamlit run app.py   (o doble click en Dashboard.bat)
"""

import datetime
import json
import os
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

import hashlib
import importlib

import opciones
import sectores

# Streamlit Cloud, al recibir un push, vuelve a ejecutar app.py pero puede seguir
# usando la versión anterior de los módulos que tenía en memoria
importlib.reload(opciones)
importlib.reload(sectores)
# cambia cuando cambia opciones.py: invalida lo cacheado con la versión anterior
VERSION_OPCIONES = hashlib.md5(open(opciones.__file__, "rb").read()
                               + open(sectores.__file__, "rb").read()).hexdigest()

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
PRECIOS_DIR = os.path.join(DATA_DIR, "precios")
FUND_DIR = os.path.join(DATA_DIR, "fundamentales")

RSI_PERIOD, RSI_ALTO, RSI_BAJO = 5, 70, 30
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 6, 13, 5
COL_RSI = f"RSI {RSI_PERIOD}"
RETORNOS = {"1m %": 21, "3m %": 63, "6m %": 126, "1a %": 252}

FUND_ROWS = [
    ("TotalRevenue", "Ingresos"),
    ("GrossProfit", "Utilidad bruta"),
    ("OperatingIncome", "Utilidad operacional"),
    ("NetIncome", "Utilidad neta"),
    ("DilutedEPS", "EPS diluido"),
    ("EBITDA", "EBITDA"),
    ("FreeCashFlow", "Flujo de caja libre"),
    ("CapitalExpenditure", "CAPEX"),
    ("TotalDebt", "Deuda total"),
    ("CashAndCashEquivalents", "Caja y equivalentes"),
]

st.set_page_config(page_title="Fintual Dashboard", layout="wide", page_icon="📈")


# ---------------------------------------------------------------------------
# Indicadores
# ---------------------------------------------------------------------------

def rsi(close, period=RSI_PERIOD):
    """RSI de Wilder (inicializado con media simple)."""
    delta = close.diff().to_numpy(dtype=float)
    gan, per = np.clip(delta, 0, None), np.clip(-delta, 0, None)
    out = np.full(len(close), np.nan)
    if len(close) <= period:
        return pd.Series(out, index=close.index)
    g, p = gan[1:period + 1].mean(), per[1:period + 1].mean()
    for i in range(period, len(close)):
        if i > period:
            g = (g * (period - 1) + gan[i]) / period
            p = (p * (period - 1) + per[i]) / period
        out[i] = 100.0 if p == 0 else 100 - 100 / (1 + g / p)
    return pd.Series(out, index=close.index)


def senal_rsi(v):
    if pd.isna(v):
        return "Sin datos"
    return "Sobrecompra" if v >= RSI_ALTO else "Sobreventa" if v <= RSI_BAJO else "Neutral"


def macd(close):
    """Devuelve (línea MACD, línea de señal, histograma)."""
    linea = (close.ewm(span=MACD_FAST, adjust=False).mean()
             - close.ewm(span=MACD_SLOW, adjust=False).mean())
    senal = linea.ewm(span=MACD_SIGNAL, adjust=False).mean()
    return linea, senal, linea - senal


def senal_macd(hist):
    """Estado según el signo del histograma y si sube o baja respecto a la sesión anterior."""
    if len(hist) < 2 or hist.iloc[-2:].isna().any():
        return "Sin datos"
    act, prev = hist.iloc[-1], hist.iloc[-2]
    if act >= 0:
        return "Alcista" if act >= prev else "Perdiendo fuerza"
    return "Pre-cruce" if act >= prev else "Bajista"


# ---------------------------------------------------------------------------
# Seguimiento
# ---------------------------------------------------------------------------
# En Streamlit Cloud el disco se borra en cada redeploy, así que la lista se
# guarda en un GitHub Gist privado (configurado en los secrets de la app).
# Sin secrets (uso local) se guarda en data/seguimiento.json y data/lineas.json.

GIST_FILE = "seguimiento.json"
LINEAS = "lineas.json"   # líneas de tendencia guardadas, en el mismo Gist


def config_gist():
    try:
        s = st.secrets["seguimiento"]
        return s["gist_id"], {"Authorization": f"Bearer {s['token']}",
                              "Accept": "application/vnd.github+json"}
    except Exception:
        return None, None


def archivo_gist(files, nombre):
    """El archivo pedido dentro del Gist. Para el seguimiento se acepta cualquier
    otro nombre (el Gist se creó con "gistfile1.txt"). None si no existe."""
    if nombre in files:
        return files[nombre]
    if nombre == GIST_FILE:
        return next((f for n, f in files.items() if n != LINEAS), None)
    return None


@st.cache_data(ttl=60, show_spinner=False)
def leer_json(nombre):
    gist_id, headers = config_gist()
    try:
        if gist_id:
            r = requests.get(f"https://api.github.com/gists/{gist_id}", headers=headers, timeout=15)
            r.raise_for_status()
            f = archivo_gist(r.json()["files"], nombre)
            return json.loads(f["content"].strip() or "{}") if f else {}
        with open(os.path.join(DATA_DIR, nombre), encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        st.error(f"No se pudo leer {nombre}: {e}")
        return {}


def guardar_json(nombre, datos):
    texto = json.dumps(datos, ensure_ascii=False, indent=1)
    gist_id, headers = config_gist()
    if gist_id:
        url = f"https://api.github.com/gists/{gist_id}"
        f = archivo_gist(requests.get(url, headers=headers, timeout=15).json()["files"], nombre)
        r = requests.patch(url, headers=headers, timeout=15,
                           json={"files": {f["filename"] if f else nombre: {"content": texto}}})
        r.raise_for_status()
    else:
        with open(os.path.join(DATA_DIR, nombre), "w", encoding="utf-8") as f:
            f.write(texto)
    leer_json.clear()


def leer_seguimiento():
    """{ticker: {"fecha": "YYYY-MM-DD", "precio": float}}"""
    return leer_json(GIST_FILE)


def guardar_seguimiento(seg):
    guardar_json(GIST_FILE, seg)


def leer_lineas(ticker):
    """[{"x0": "YYYY-MM-DD", "y0": float, "x1": ..., "y1": ...}, ...]"""
    return leer_json(LINEAS).get(ticker, [])


def guardar_lineas(ticker, lineas):
    todas = dict(leer_json(LINEAS))
    if lineas:
        todas[ticker] = lineas
    else:
        todas.pop(ticker, None)
    guardar_json(LINEAS, todas)


def agregar_seguimiento(ticker, precio):
    seg = dict(leer_seguimiento())
    hoy = datetime.datetime.now(ZoneInfo("America/New_York")).date()   # el servidor corre en UTC
    seg[ticker] = {"fecha": hoy.isoformat(), "precio": round(float(precio), 2)}
    guardar_seguimiento(seg)


def quitar_seguimiento(ticker):
    seg = dict(leer_seguimiento())
    seg.pop(ticker, None)
    guardar_seguimiento(seg)


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def cargar_info():
    with open(os.path.join(DATA_DIR, "company_info.json"), encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(ttl=300, show_spinner=False)
def cargar_sello():
    try:
        with open(os.path.join(DATA_DIR, "_ultima_actualizacion.json"), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def cargar_precios(ticker):
    path = os.path.join(PRECIOS_DIR, f"{ticker}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
    return df if len(df) else None


@st.cache_data(show_spinner=False)
def cargar_fund(ticker, tipo):
    path = os.path.join(FUND_DIR, f"{ticker}_{tipo}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df.sort_values("Date").reset_index(drop=True) if len(df) else None


def multiplos(ticker, precio):
    """P/E (EPS TTM, o del último año) y EV/EBITDA; márgenes y crecimiento anuales."""
    out = {"P/E": None, "EV/EBITDA": None, "Crec. ingresos %": None, "Marg. neto %": None}
    dfa, dfq = cargar_fund(ticker, "anual"), cargar_fund(ticker, "trimestral")
    eps = None
    if dfq is not None and len(dfq) >= 4 and dfq["DilutedEPS"].tail(4).notna().all():
        eps = dfq["DilutedEPS"].tail(4).sum()
    if dfa is None:
        return out
    u = dfa.iloc[-1]
    if eps is None and pd.notna(u["DilutedEPS"]):
        eps = u["DilutedEPS"]
    if eps and eps > 0:
        out["P/E"] = precio / eps
    if pd.notna(u["EBITDA"]) and u["EBITDA"] > 0 and pd.notna(u["DilutedAverageShares"]):
        ev = precio * u["DilutedAverageShares"] + np.nan_to_num(u["TotalDebt"]) \
            - np.nan_to_num(u["CashAndCashEquivalents"])
        out["EV/EBITDA"] = ev / u["EBITDA"]
    out["Crec. ingresos %"] = u.get("TotalRevenue_var_pct")
    if pd.notna(u["TotalRevenue"]) and u["TotalRevenue"] and pd.notna(u["NetIncome"]):
        out["Marg. neto %"] = u["NetIncome"] / u["TotalRevenue"] * 100
    return out


@st.cache_data(show_spinner="Calculando resumen de todas las empresas...")
def resumen(_sello_key):
    info = cargar_info()
    filas = []
    for t, i in info.items():
        df = cargar_precios(t)
        if df is None or len(df) < 5:
            continue
        close = df["Close"]
        precio = float(close.iloc[-1])
        r = rsi(close).iloc[-1]
        hist = macd(close)[2]
        fila = {"Ticker": t, "Nombre": i.get("name", t), "Universo": i.get("universe", "?"),
                "Sector": i.get("sector", "Sin clasificar"), "Precio": precio,
                COL_RSI: r, "Señal RSI": senal_rsi(r),
                "MACD hist %": hist.iloc[-1] / precio * 100, "Señal MACD": senal_macd(hist)}
        for col, n in RETORNOS.items():
            fila[col] = (precio / close.iloc[-n - 1] - 1) * 100 if len(close) > n else None
        fila.update(multiplos(t, precio))
        filas.append(fila)
    return pd.DataFrame(filas).sort_values("Ticker").reset_index(drop=True)


def fmt_money(x):
    if pd.isna(x):
        return "—"
    for div, suf in [(1e12, "T"), (1e9, "B"), (1e6, "M")]:
        if abs(x) >= div:
            return f"{x / div:,.2f}{suf}"
    return f"{x:,.2f}"


def fmt_pct(x):
    return "—" if pd.isna(x) else f"{x:+.1f}%"


def colorear(v):
    if isinstance(v, str) and v.startswith("+"):
        return "color: #3ecf8e"
    if isinstance(v, str) and v.startswith("-") and v != "—":
        return "color: #ef5a6f"
    return ""


def tabla_fundamental(df, anual):
    """Métricas en filas y periodos en columnas (en anual, con su variación YoY)."""
    cols, var_cols = {}, []
    for j, (_, r) in enumerate(df.tail(5 if anual else 8).iterrows()):
        h = f"FY{str(r['Date'])[:4]}" if anual else str(r["Date"])[:7]
        cols[h] = [f"{r[k]:.2f}" if k == "DilutedEPS" and pd.notna(r[k]) else fmt_money(r[k])
                   for k, _ in FUND_ROWS]
        var = "_var_pct" if anual else "_YoY_pct"
        vals = [fmt_pct(r.get(k + var)) for k, _ in FUND_ROWS]
        if j > 0 and any(v != "—" for v in vals):
            var_cols.append(f"{h} {'YoY' if anual else 'a/a'}")
            cols[var_cols[-1]] = vals
    tabla = pd.DataFrame(cols, index=[lbl for _, lbl in FUND_ROWS])
    return tabla.style.map(colorear, subset=var_cols)


@st.cache_data(ttl=900, show_spinner="Descargando opciones...")
def cargar_muros(ticker, precio, version=""):
    """(muros, error). Cacheado 15 min: las opciones cambian durante el día."""
    try:
        return opciones.muros(ticker, precio), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


@st.cache_data(ttl=1800, show_spinner="Buscando noticias...")
def cargar_noticias(ticker, version=""):
    try:
        return opciones.noticias(ticker), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


@st.cache_data(ttl=900, show_spinner="Descargando velas de 1 hora...")
def cargar_velas_1h(ticker, version=""):
    try:
        return opciones.velas_1h(ticker), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


COLOR_MURO = {"C": ["#3ecf8e", "#8fe0b9"], "P": ["#ef5a6f", "#f0a8b0"]}


COLOR_ESTADO = {"Rompió ↑": "#e6b45e", "Rompió ↓": "#e6b45e", "Probando": "#5ea8e6",
                "Rebotó": "#3ecf8e", "Sin tocar": "#8b93a3"}

# botones para dibujar en los gráficos (barra de arriba a la derecha)
DIBUJO = {"displaylogo": False,
          "modeBarButtonsToAdd": ["drawline", "drawopenpath", "drawrect", "eraseshape"]}


def lista_muros(venc):
    """[(etiqueta, strike, oi, color)] de un vencimiento."""
    return [(f"{lado}W{i + 1}", k, oi, COLOR_MURO[lado][i])
            for lado, nombre in [("C", "calls"), ("P", "puts")]
            for i, (k, oi) in enumerate(venc[nombre])] if venc else []


def grafico_muros(velas, venc, estados):
    """Velas de 1h de la semana con los walls extendidos hasta el vencimiento y,
    a la derecha, el perfil de volumen de la semana (mismo eje de precios)."""
    ahora = velas.index[-1]
    # cierre del día de vencimiento (15:59, para no caer en el corte de la noche)
    fin_exp = pd.Timestamp(venc["exp"]) + pd.Timedelta(hours=15, minutes=59) if venc else None
    x_fin = max(fin_exp, ahora + pd.Timedelta(hours=1)) if venc else ahora + pd.Timedelta(hours=1)
    muros_ = lista_muros(venc)

    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, column_widths=[0.82, 0.18],
                        horizontal_spacing=0.01)
    fig.add_trace(go.Candlestick(
        x=velas.index, open=velas["Open"], high=velas["High"], low=velas["Low"], close=velas["Close"],
        increasing_line_color="#3ecf8e", decreasing_line_color="#ef5a6f", name="Velas 1h"), row=1, col=1)

    # rango de precios: velas + muros, con un pequeño margen
    ys = [velas["Low"].min(), velas["High"].max()] + [k for _, k, _, _ in muros_]
    pad = (max(ys) - min(ys)) * 0.05
    y_lo, y_hi = min(ys) - pad, max(ys) + pad

    # perfil de volumen: gris, y del color del muro dentro de su zona (±1,5%)
    precios, vol, alto = opciones.perfil_volumen(velas, y_lo, y_hi, n=70)
    colores = []
    for p in precios:
        c = "rgba(139,147,163,0.5)"
        for _, k, _, color in muros_:
            if abs(p / k - 1) <= opciones.ZONA:
                c = color
        colores.append(c)
    fig.add_trace(go.Bar(x=vol, y=precios, orientation="h", marker_color=colores, width=alto * 0.9,
                         hovertemplate="%{y:,.2f}: %{x:,.0f} acciones<extra>Perfil de volumen</extra>"),
                  row=1, col=2)
    if vol.sum():
        poc = precios[vol.argmax()]
        fig.add_annotation(x=1, xref="x2 domain", y=poc, yref="y2", text=f"POC {poc:,.2f}",
                           showarrow=False, xanchor="right", yanchor="bottom", font=dict(size=9))

    for tag, k, oi, color in muros_:
        est = estados.get(tag, "")
        fig.add_hrect(y0=k * (1 - opciones.ZONA), y1=k * (1 + opciones.ZONA), fillcolor=color,
                      opacity=0.07, line_width=0, row=1, col=1)
        fig.add_trace(go.Scatter(x=[velas.index[0], x_fin], y=[k, k], mode="lines",
                                 line=dict(color=color, width=1.5, dash="dot"),
                                 hovertemplate=f"{tag} {k:,.2f} · OI {oi:,} · {est}<extra></extra>"),
                      row=1, col=1)
        fig.add_hline(y=k, line=dict(color=color, width=1, dash="dot"), row=1, col=2)
        fig.add_annotation(x=x_fin, y=k, text=f"{tag} {k:,.2f} · {est}", showarrow=False,
                           xanchor="right", yanchor="bottom", font=dict(size=10, color=color))
    if venc:
        fig.add_vline(x=fin_exp, line=dict(color="#e6b45e", width=1, dash="dash"), row=1, col=1)
        fig.add_annotation(x=fin_exp, y=0, yref="y domain", text=f"Vence {venc['exp']}", showarrow=False,
                           xanchor="right", yanchor="bottom", font=dict(size=10, color="#e6b45e"))
        fig.add_vrect(x0=ahora, x1=x_fin, fillcolor="#e6b45e", opacity=0.04, line_width=0, row=1, col=1)
    fig.add_vline(x=ahora, line=dict(color="#8b93a3", width=1, dash="dash"), row=1, col=1)
    fig.update_yaxes(range=[y_lo, y_hi])
    fig.update_xaxes(range=[velas.index[0] - pd.Timedelta(minutes=30), x_fin],
                     rangeslider_visible=False, row=1, col=1,
                     rangebreaks=[dict(bounds=["sat", "mon"]), dict(bounds=[16, 9.5], pattern="hour")])
    fig.update_xaxes(showticklabels=False, showgrid=False, row=1, col=2)
    fig.update_layout(height=520, template="plotly_dark", showlegend=False, margin=dict(t=10, b=10),
                      bargap=0, newshape=dict(line=dict(color="#e6b45e", width=2)))
    return fig


PCT = st.column_config.NumberColumn(format="%+.1f%%")
COLUMNAS = {
    "Precio": st.column_config.NumberColumn(format="%.2f"),
    COL_RSI: st.column_config.NumberColumn(format="%.1f"),
    "MACD hist %": st.column_config.NumberColumn(format="%+.2f%%"),
    "1m %": PCT, "3m %": PCT, "6m %": PCT, "1a %": PCT, "Crec. ingresos %": PCT,
    "Desde que se agregó %": PCT,
    "Precio al agregar": st.column_config.NumberColumn(format="%.2f"),
    "Marg. neto %": st.column_config.NumberColumn(format="%.1f%%"),
    "P/E": st.column_config.NumberColumn(format="%.1fx"),
    "EV/EBITDA": st.column_config.NumberColumn(format="%.1fx"),
}


PESTANAS = ["la lista", f"{COL_RSI} < {RSI_BAJO}", "Seguimiento"]   # para el botón de volver


def mostrar_tabla(tabla, key, pestana, height=650):
    """Tabla ordenable; al marcar la casilla de una fila se abre esa empresa.
    `pestana` (índice) queda anotada para que el botón de volver regrese a ella."""
    tabla = tabla.reset_index(drop=True)
    sel = st.dataframe(tabla, hide_index=True, width="stretch", height=height,
                       on_select="rerun", selection_mode="single-row", key=key,
                       column_config=COLUMNAS)
    if sel.selection.rows:
        st.session_state["_abrir"] = tabla.iloc[sel.selection.rows[0]]["Ticker"]
        st.session_state["_pestana_origen"] = pestana
        del st.session_state[key]   # limpia la selección para la vuelta
        st.rerun()


# ---------------------------------------------------------------------------
# Rotación sectorial
# ---------------------------------------------------------------------------

N_CALOR = 20   # períodos del mapa de calor
COLOR_CUADRANTE = {"Liderando": "#3ecf8e", "Debilitándose": "#e6b45e",
                   "Rezagado": "#ef5a6f", "Mejorando": "#5ea8e6"}


@st.cache_data(show_spinner="Calculando la rotación sectorial...")
def datos_sectores(frecuencia, universos, _sello_key, version=""):
    """Series por sector + RRG del universo elegido (`_sello_key` invalida al llegar precios nuevos)."""
    info_ = cargar_info()
    tickers = [t for t, i in info_.items() if i.get("universe") in universos]
    precios = {t: df for t in tickers if (df := cargar_precios(t)) is not None}
    cierre, dolares = sectores.paneles(precios, frecuencia)
    s = sectores.series(cierre, dolares, {t: info_[t].get("sector", "Sin clasificar") for t in precios},
                        frecuencia)
    ratio, mom = sectores.rrg(s["rs"], frecuencia)
    return s, ratio, mom


def periodo_en_curso(fecha, frecuencia):
    """¿El último período todavía no cierra? (hora de Nueva York, cierre ~16:15)."""
    ahora = datetime.datetime.now(ZoneInfo("America/New_York"))
    hoy, abierto = ahora.date(), (ahora.hour, ahora.minute) < (16, 15)
    if frecuencia == "Diaria":
        return fecha.date() == hoy and abierto
    return fecha.date() > hoy or (fecha.date() == hoy and abierto)


def colorear_num(v):
    if isinstance(v, (int, float)) and pd.notna(v):
        return "color: #3ecf8e" if v > 0 else "color: #ef5a6f" if v < 0 else ""
    return ""


def grafico_rrg(ratio, mom, fecha, cola=6):
    """Gráfico de rotación relativa: cada sector con su cola de los últimos períodos."""
    r, m = ratio.loc[:fecha].tail(cola), mom.loc[:fecha].tail(cola)
    fig = go.Figure()
    colores = px_colores()
    for i, sec in enumerate(r.columns):
        x, y = r[sec], m[sec]
        if x.isna().all() or y.isna().all():
            continue
        c = colores[i % len(colores)]
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", line=dict(color=c, width=1.5),
                                 marker=dict(size=[4] * (len(x) - 1) + [11], color=c),
                                 name=sec, hovertemplate=f"{sec}<br>RS-Ratio %{{x:.2f}}<br>"
                                                         "RS-Momentum %{y:.2f}<extra></extra>"))
        fig.add_annotation(x=x.iloc[-1], y=y.iloc[-1], text=sec, showarrow=False, yshift=12,
                           font=dict(size=10, color=c))
    # cuadrantes simétricos alrededor de 100
    dx = max(abs(r.stack() - 100).max(), 0.5) * 1.15
    dy = max(abs(m.stack() - 100).max(), 0.5) * 1.15
    for x0, x1, y0, y1, txt, c, xa, ya in [
            (100, 100 + dx, 100, 100 + dy, "Liderando", "#3ecf8e", "right", "top"),
            (100, 100 + dx, 100 - dy, 100, "Debilitándose", "#e6b45e", "right", "bottom"),
            (100 - dx, 100, 100 - dy, 100, "Rezagado", "#ef5a6f", "left", "bottom"),
            (100 - dx, 100, 100, 100 + dy, "Mejorando", "#5ea8e6", "left", "top")]:
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, fillcolor=c, opacity=0.06,
                      line_width=0, layer="below")
        fig.add_annotation(x=x1 if xa == "right" else x0, y=y1 if ya == "top" else y0, text=txt,
                           showarrow=False, xanchor=xa, yanchor=ya, font=dict(size=12, color=c))
    fig.add_hline(y=100, line=dict(color="#8b93a3", width=1))
    fig.add_vline(x=100, line=dict(color="#8b93a3", width=1))
    fig.update_xaxes(range=[100 - dx, 100 + dx], title_text="RS-Ratio (fuerza relativa)")
    fig.update_yaxes(range=[100 - dy, 100 + dy], title_text="RS-Momentum (aceleración)")
    fig.update_layout(height=600, template="plotly_dark", showlegend=False, margin=dict(t=10, b=10))
    return fig


def px_colores():
    return ["#5ea8e6", "#e6b45e", "#3ecf8e", "#ef5a6f", "#b98af0", "#f08ac8",
            "#6de0d8", "#d8e06d", "#f0a35e", "#8fa8ff", "#c9c9c9"]


def grafico_calor(df, fecha, frecuencia, clave):
    """Sectores (filas) x últimos períodos (columnas)."""
    d = df.loc[:fecha].tail(N_CALOR)
    d = d[d.iloc[-1].sort_values(ascending=False).index]          # el mejor del período, arriba
    x = [f"{f:%b-%y}" if frecuencia == "Mensual" else f"{f:%d-%b}" for f in d.index]
    centro = 50 if clave == "alza" else 0
    lim = (d - centro).abs().quantile(0.95).max() or 1
    fig = go.Figure(go.Heatmap(
        z=d.T.values, x=x, y=list(d.columns), colorscale="RdYlGn", zmid=centro,
        zmin=centro - lim, zmax=centro + lim, xgap=1, ygap=1,
        hovertemplate="%{y} · %{x}: %{z:.2f}<extra></extra>"))
    fig.update_layout(height=40 * len(d.columns) + 80, template="plotly_dark", margin=dict(t=10, b=10))
    fig.update_yaxes(autorange="reversed")
    return fig


# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------

@st.cache_resource
def ultimo_sello():
    return {"valor": None}


# cuando la Action publica precios nuevos cambia el sello: se descarta todo lo
# cacheado para que la app muestre los datos nuevos sin reiniciarse
sello = cargar_sello()
if ultimo_sello()["valor"] != sello:
    st.cache_data.clear()
    ultimo_sello()["valor"] = sello

info = cargar_info()
data = resumen(json.dumps(sello))

# un click en una fila del listado deja el ticker pendiente; se aplica aquí,
# antes de crear los widgets que dependen de él
if "_abrir" in st.session_state:
    st.session_state["vista"] = "Empresa"
    st.session_state["ticker"] = st.session_state.pop("_abrir")
if st.session_state.pop("_volver", False):
    st.session_state["vista"] = "General"
    st.session_state["_ir_a_pestana"] = st.session_state.get("_pestana_origen", 0)

st.sidebar.title("📈 Fintual")
vista = st.sidebar.radio("Vista", ["General", "Sectores", "Empresa"], horizontal=True, key="vista")
universos = sorted(data["Universo"].unique())
sel_uni = st.sidebar.multiselect("Universo", universos, default=universos)
sector = st.sidebar.selectbox("Sector", ["Todos"] + sorted(data["Sector"].unique()))

filtrado = data[data["Universo"].isin(sel_uni)]
if sector != "Todos":
    filtrado = filtrado[filtrado["Sector"] == sector]

st.sidebar.markdown(f"**{len(filtrado)}** de {len(data)} empresas")
if sello:
    st.sidebar.caption(f"Precios actualizados: {sello.get('fecha', '?')} · "
                       f"última sesión: {sello.get('ultima_sesion', '?')}")


# ---------------------------------------------------------------------------
# Vista general
# ---------------------------------------------------------------------------

if vista == "General":
    st.title("Resumen general")
    st.caption("Marca la casilla de una fila para abrir la empresa · click en el título de una columna "
               f"para ordenar. {COL_RSI}: sobre {RSI_ALTO} = sobrecompra, bajo {RSI_BAJO} = sobreventa · "
               f"MACD {MACD_FAST}/{MACD_SLOW}/{MACD_SIGNAL}: Pre-cruce = histograma negativo pero subiendo.")

    sobreventa = filtrado[filtrado[COL_RSI] < RSI_BAJO]
    etiquetas = ["Todas", f"{COL_RSI} < {RSI_BAJO} ({len(sobreventa)})", "⭐ Seguimiento"]
    # al volver desde una empresa se abre la pestaña de donde se vino
    destino = st.session_state.pop("_ir_a_pestana", None)
    if destino is not None:
        st.session_state["pestanas"] = etiquetas[destino]
    if st.session_state.get("pestanas") not in etiquetas:   # el conteo del título cambió
        st.session_state.pop("pestanas", None)
    tab_todas, tab_rsi, tab_seg = st.tabs(etiquetas, key="pestanas", on_change="rerun")

    with tab_todas:
        mostrar_tabla(filtrado, "tabla_general", 0)
        st.caption("P/E con EPS diluido de los últimos 4 trimestres. EV/EBITDA con el último año fiscal. "
                   "Crecimiento y margen del último año fiscal.")

    with tab_rsi:
        if sobreventa.empty:
            st.info(f"Ninguna empresa del filtro actual tiene {COL_RSI} bajo {RSI_BAJO}.")
        else:
            mostrar_tabla(sobreventa.sort_values(COL_RSI), "tabla_rsi", 1)

    with tab_seg:
        seg = leer_seguimiento()
        a1, a2, a3, a4 = st.columns([3, 1, 3, 1], vertical_alignment="bottom")
        nuevo = a1.selectbox("Agregar empresa", [t for t in data["Ticker"] if t not in seg],
                             index=None, placeholder="Escribe un ticker...")
        if a2.button("Agregar", disabled=nuevo is None):
            agregar_seguimiento(nuevo, data.loc[data["Ticker"] == nuevo, "Precio"].iloc[0])
            st.rerun()
        quitar = a3.selectbox("Quitar empresa", sorted(seg), index=None, placeholder="Elige un ticker...")
        if a4.button("Quitar", disabled=quitar is None):
            quitar_seguimiento(quitar)
            st.rerun()

        if not seg:
            st.info("Aún no sigues ninguna empresa. Agrégalas aquí o con el botón ☆ en la vista Empresa.")
        else:
            t = data[data["Ticker"].isin(seg)].copy()
            t.insert(4, "Agregada", t["Ticker"].map(lambda x: seg[x]["fecha"]))
            t.insert(5, "Precio al agregar", t["Ticker"].map(lambda x: seg[x]["precio"]))
            t.insert(7, "Desde que se agregó %", (t["Precio"] / t["Precio al agregar"] - 1) * 100)
            mostrar_tabla(t, "tabla_seg", 2, height=min(650, 38 + 35 * len(t)))
        if config_gist()[0]:
            st.caption("✅ Guardado en GitHub Gist: se mantiene entre sesiones y dispositivos. "
                       "No depende de los filtros del panel izquierdo.")
        else:
            st.caption("⚠️ Guardado en data/seguimiento.json (archivo local). En Streamlit Cloud se pierde "
                       "al redeployar: configura los secrets [seguimiento] gist_id y token.")


# ---------------------------------------------------------------------------
# Vista sectores (rotación)
# ---------------------------------------------------------------------------

elif vista == "Sectores":
    st.title("Rotación sectorial")
    st.caption("Cómo se mueven los sectores entre sí, en precio y en volumen. Pesos iguales: cada empresa "
               "pesa lo mismo dentro de su sector. 'Mercado' = promedio de todas las empresas del universo "
               "elegido a la izquierda (el filtro de sector no aplica aquí).")

    frec = st.radio("Frecuencia", ["Diaria", "Semanal", "Mensual"], horizontal=True, key="frec_sect")
    s, ratio, mom = datos_sectores(frec, tuple(sorted(sel_uni)), json.dumps(sello), VERSION_OPCIONES)
    fechas = list(s["rs"].index[-60:])
    en_curso = periodo_en_curso(fechas[-1], frec)
    formato = {"Diaria": "{:%d-%b-%Y}", "Semanal": "semana al {:%d-%b-%Y}", "Mensual": "{:%b-%Y}"}[frec]
    etiqueta = {f: formato.format(f)
                   + (" (en curso)" if f == fechas[-1] and en_curso else "") for f in fechas}
    fecha = st.select_slider("Período (muévelo para ver rotaciones pasadas)", options=fechas,
                             value=fechas[-2] if en_curso else fechas[-1],
                             format_func=lambda f: etiqueta[f])
    if fecha == fechas[-1] and en_curso:
        st.warning("Período en curso: el retorno es parcial y el volumen todavía está incompleto, así que "
                   "el volumen relativo sale bajo. Para comparar volumen, usa el último período completo.")

    tb = sectores.tabla(s, ratio, mom, fecha, frec)
    numericas = ["Retorno %", "vs mercado (pp)", "Δ participación (pp)", "Relativo 1m %", "Relativo 3m %"]
    st.dataframe(
        tb.style.map(colorear_num, subset=numericas)
                .map(lambda v: f"color: {COLOR_CUADRANTE.get(v, '')}; font-weight: 600", subset=["Cuadrante"]),
        hide_index=True, width="stretch", column_config={
            "Retorno %": st.column_config.NumberColumn(format="%+.2f%%"),
            "vs mercado (pp)": st.column_config.NumberColumn(format="%+.2f"),
            "% al alza": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
            "Vol. relativo": st.column_config.NumberColumn(format="%.2fx"),
            "Part. volumen %": st.column_config.NumberColumn(format="%.1f%%"),
            "Δ participación (pp)": st.column_config.NumberColumn(format="%+.2f"),
            "Relativo 1m %": st.column_config.NumberColumn(format="%+.1f%%"),
            "Relativo 3m %": st.column_config.NumberColumn(format="%+.1f%%"),
        })
    st.caption(
        "**vs mercado** = retorno del sector menos el del mercado · **% al alza** = parte de las empresas del "
        "sector que subió (amplitud) · **Vol. relativo** = volumen en dólares del período / su promedio de los "
        f"{sectores.VENTANA_VOL[frec]} anteriores · **Part. volumen** = parte del volumen en dólares del mercado que "
        "se transó en el sector · **Δ participación** = participación actual menos su promedio: positivo = "
        "está entrando más dinero que lo habitual · **Relativo 1m/3m** = cuánto le ganó (o perdió) al "
        "mercado en ese plazo. Una rotación se ve como sectores con **vs mercado** y **Δ participación** "
        "positivos (entra dinero y suben más que el resto) mientras otros pierden ambas cosas.")

    st.subheader("Gráfico de rotación relativa (RRG)")
    st.plotly_chart(grafico_rrg(ratio, mom, fecha), width="stretch")
    ventana, rezago = sectores.RRG[frec]
    st.caption(
        f"Eje X: **RS-Ratio** = fuerza relativa del sector contra el mercado (sobre 100 = le gana a su "
        f"promedio de {ventana} períodos). Eje Y: **RS-Momentum** = si esa fuerza acelera (sobre 100) o frena. "
        "La cola muestra los últimos 6 períodos. Las rotaciones suelen avanzar en el sentido de las agujas del "
        "reloj: **Mejorando → Liderando → Debilitándose → Rezagado**. Un sector que pasa de Rezagado a "
        "Mejorando es candidato a recibir la próxima rotación.")

    st.subheader("Mapa de calor: cómo ha ido rotando")
    metrica = st.radio("Métrica", ["Retorno vs mercado (pp)", "Δ participación de volumen (pp)",
                                   "Amplitud (% al alza)"], horizontal=True)
    clave = {"Retorno vs mercado (pp)": "rel", "Δ participación de volumen (pp)": "d_part",
             "Amplitud (% al alza)": "alza"}[metrica]
    st.plotly_chart(grafico_calor(s[clave], fecha, frec, clave), width="stretch")
    st.caption(f"Últimos {N_CALOR} períodos hasta el elegido. Verde = el sector le ganó al mercado / ganó "
               "participación de volumen / subió la mayoría de sus empresas; rojo = lo contrario.")


# ---------------------------------------------------------------------------
# Vista empresa
# ---------------------------------------------------------------------------

else:
    lista = filtrado["Ticker"].tolist()
    if not lista:
        st.warning("No hay empresas con el filtro actual.")
        st.stop()
    if st.session_state.get("ticker") not in lista:
        st.session_state["ticker"] = "NVDA" if "NVDA" in lista else lista[0]
    ticker = st.sidebar.selectbox("Empresa", lista, key="ticker")
    fila = data[data["Ticker"] == ticker].iloc[0]
    ci = info.get(ticker, {})

    origen = PESTANAS[st.session_state.get("_pestana_origen", 0)]
    if st.button(f"← Volver a {origen}"):
        st.session_state["_volver"] = True
        st.rerun()

    seg = leer_seguimiento()
    h1, h2 = st.columns([5, 1], vertical_alignment="center")
    h1.title(f"{fila['Nombre']} ({ticker})")
    if ticker in seg:
        if h2.button("★ Quitar de seguimiento"):
            quitar_seguimiento(ticker)
            st.rerun()
    elif h2.button("☆ Agregar a seguimiento"):
        agregar_seguimiento(ticker, fila["Precio"])
        st.rerun()
    st.caption(f"{fila['Universo']} · {fila['Sector']} · {ci.get('industry', '')}")
    if ci.get("desc"):
        st.write(ci["desc"])

    c = st.columns(6)
    c[0].metric("Precio", f"{fila['Precio']:,.2f}")
    c[1].metric(COL_RSI, f"{fila[COL_RSI]:.1f}" if pd.notna(fila[COL_RSI]) else "—",
                fila["Señal RSI"], delta_color="off")
    c[2].metric(f"MACD {MACD_FAST}/{MACD_SLOW}/{MACD_SIGNAL}", f"{fila['MACD hist %']:+.2f}%",
                fila["Señal MACD"], delta_color="off",
                help="Histograma (MACD − señal) como % del precio")
    c[3].metric("1 año", fmt_pct(fila["1a %"]))
    c[4].metric("P/E", f"{fila['P/E']:.1f}x" if pd.notna(fila["P/E"]) else "—")
    c[5].metric("EV/EBITDA", f"{fila['EV/EBITDA']:.1f}x" if pd.notna(fila["EV/EBITDA"]) else "—")

    # ---- últimas noticias ----
    with st.expander("📰 Últimas noticias", expanded=False):
        notas, err_n = cargar_noticias(ticker, VERSION_OPCIONES)
        if err_n:
            st.caption(f"No se pudieron cargar las noticias: {err_n}")
        elif not notas:
            st.caption("Yahoo no tiene noticias recientes para esta empresa.")
        else:
            for nt in notas:
                cuando = f"{nt['fecha']:%d-%b %H:%M} NY" if nt["fecha"] is not None else ""
                otros = f" · también: {', '.join(nt['otros'][:4])}" if nt["otros"] else ""
                st.markdown(f"[{nt['titulo']}]({nt['link']})  \n"
                            f"<small style='color:#8b93a3'>{nt['fuente']} · {cuando}{otros}</small>",
                            unsafe_allow_html=True)
            st.caption("Fuente: Yahoo Finance (se actualiza cada 30 minutos).")

    # ---- gráfico: precio, volumen, RSI y MACD con el mismo eje de tiempo ----
    df = cargar_precios(ticker)
    rango = st.radio("Rango", ["3M", "6M", "1A", "2A", "5A"], index=2, horizontal=True)
    n = {"3M": 63, "6M": 126, "1A": 252, "2A": 504, "5A": 1260}[rango]
    m_linea, m_senal, m_hist = (x.tail(n) for x in macd(df["Close"]))
    d, r = df.tail(n), rsi(df["Close"]).tail(n)

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.5, 0.12, 0.19, 0.19])
    if n <= 252:
        fig.add_trace(go.Candlestick(x=d["Date"], open=d["Open"], high=d["High"], low=d["Low"],
                                     close=d["Close"], name="Precio",
                                     increasing_line_color="#3ecf8e",
                                     decreasing_line_color="#ef5a6f"), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=d["Date"], y=d["Close"], name="Cierre",
                                 line=dict(color="#e6e8ec", width=1.5)), row=1, col=1)
    fig.add_trace(go.Bar(x=d["Date"], y=d["Volume"], name="Volumen",
                         marker_color=np.where(d["Close"] >= d["Open"], "#3ecf8e", "#ef5a6f")),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=d["Date"], y=r, name=COL_RSI, line=dict(color="#b98af0", width=1.3)),
                  row=3, col=1)
    fig.add_hline(y=RSI_ALTO, line=dict(color="#ef5a6f", width=1, dash="dot"), row=3, col=1)
    fig.add_hline(y=RSI_BAJO, line=dict(color="#3ecf8e", width=1, dash="dot"), row=3, col=1)
    fig.add_trace(go.Bar(x=d["Date"], y=m_hist, name="Histograma",
                         marker_color=np.where(m_hist >= 0, "#3ecf8e", "#ef5a6f")), row=4, col=1)
    fig.add_trace(go.Scatter(x=d["Date"], y=m_linea, name="MACD",
                             line=dict(color="#e6e8ec", width=1.2)), row=4, col=1)
    fig.add_trace(go.Scatter(x=d["Date"], y=m_senal, name="Señal",
                             line=dict(color="#e6b45e", width=1.2, dash="dash")), row=4, col=1)
    fig.update_yaxes(title_text="Vol.", row=2, col=1)
    fig.update_yaxes(range=[0, 100], title_text=COL_RSI, row=3, col=1)
    fig.update_yaxes(title_text=f"MACD {MACD_FAST}/{MACD_SLOW}/{MACD_SIGNAL}", row=4, col=1)
    # líneas de tendencia guardadas: se prolongan hasta el último día del gráfico
    lineas = leer_lineas(ticker)
    ini, fin = d["Date"].iloc[0], d["Date"].iloc[-1]
    for ln in lineas:
        x0, x1 = pd.Timestamp(ln["x0"]), pd.Timestamp(ln["x1"])
        pend = (ln["y1"] - ln["y0"]) / max((x1 - x0).days, 1)
        # se dibuja solo el tramo visible: desde el inicio del rango (si la línea
        # parte antes) hasta el último día del gráfico
        xa, xb = max(x0, ini), max(x1, fin)
        if xa >= xb:
            continue
        fig.add_trace(go.Scatter(x=[xa, xb], y=[ln["y0"] + pend * (xa - x0).days,
                                                ln["y0"] + pend * (xb - x0).days],
                                 mode="lines", line=dict(color="#e6b45e", width=1.6),
                                 hoverinfo="skip"), row=1, col=1)
    fig.update_yaxes(range=[d["Low"].min() * 0.97, d["High"].max() * 1.03], row=1, col=1)

    # el eje X queda fijo al período de las velas (las líneas no lo estiran)
    fig.update_xaxes(range=[ini - pd.Timedelta(days=1), fin + pd.Timedelta(days=1)],
                     rangebreaks=[dict(bounds=["sat", "mon"])], rangeslider_visible=False)
    fig.update_layout(height=800, template="plotly_dark", showlegend=False,
                      margin=dict(t=10, b=10), bargap=0.1,
                      newshape=dict(line=dict(color="#e6b45e", width=2)))
    st.plotly_chart(fig, width="stretch", config=DIBUJO)

    with st.expander(f"✏️ Líneas de tendencia guardadas ({len(lineas)})"):
        st.caption("Los botones de dibujo del gráfico (arriba a la derecha) sirven para trazar rápido, "
                   "pero esas líneas se pierden al recargar. Las que agregues aquí quedan guardadas y "
                   "se prolongan hasta hoy.")
        with st.form("nueva_linea", clear_on_submit=True):
            inicio = d.iloc[max(len(d) - 63, 0)]
            f1, f2, f3, f4 = st.columns(4)
            x0 = f1.date_input("Desde", inicio["Date"].date())
            y0 = f2.number_input("Precio desde", value=round(float(inicio["Close"]), 2), format="%.2f")
            x1 = f3.date_input("Hasta", fin.date())
            y1 = f4.number_input("Precio hasta", value=round(float(d["Close"].iloc[-1]), 2), format="%.2f")
            if st.form_submit_button("Guardar línea"):
                if x1 <= x0:
                    st.error("La fecha 'Hasta' debe ser posterior a 'Desde'.")
                else:
                    guardar_lineas(ticker, lineas + [{"x0": x0.isoformat(), "y0": y0,
                                                      "x1": x1.isoformat(), "y1": y1}])
                    st.rerun()
        for i, ln in enumerate(lineas):
            l1, l2 = st.columns([5, 1], vertical_alignment="center")
            l1.write(f"{ln['x0']} ({ln['y0']:,.2f}) → {ln['x1']} ({ln['y1']:,.2f})")
            if l2.button("Borrar", key=f"borrar_linea_{i}"):
                guardar_lineas(ticker, lineas[:i] + lineas[i + 1:])
                st.rerun()

    # ---- call/put walls sobre las velas de 1h de la última semana ----
    st.subheader("Call / put walls")
    paredes, err_op = cargar_muros(ticker, float(fila["Precio"]), VERSION_OPCIONES)
    velas, err_v = cargar_velas_1h(ticker, VERSION_OPCIONES)
    if err_op or not paredes:
        st.info(f"Sin cadena de opciones para {ticker}" + (f": {err_op}" if err_op else "."))
    if err_v or velas is None or velas.empty:
        st.info(f"No se pudieron cargar las velas de 1 hora: {err_v or 'sin datos'}")
    else:
        venc = None
        if paredes:
            etiquetas = [f"{p['exp']} ({p['dte']} DTE)" for p in paredes]
            venc = paredes[etiquetas.index(st.radio("Vencimiento", etiquetas, horizontal=True))]
        estados = {tag: opciones.estado_muro(velas, k) for tag, k, _, _ in lista_muros(venc)}
        st.plotly_chart(grafico_muros(velas, venc, estados), width="stretch", config=DIBUJO)
        if venc:
            precio_act = float(velas["Close"].iloc[-1])
            tabla_m = pd.DataFrame([{
                "Muro": tag, "Strike": k, "Open Interest": oi,
                "Esta semana": estados[tag],
                "Distancia %": (k / precio_act - 1) * 100,
                "Vol. semana en zona %": opciones.volumen_en_zona(velas, k),
            } for tag, k, oi, _ in lista_muros(venc)])
            st.dataframe(
                tabla_m.style.map(lambda v: f"color: {COLOR_ESTADO.get(v, '')}; font-weight: 600",
                                  subset=["Esta semana"]),
                hide_index=True, width="content", column_config={
                    "Strike": st.column_config.NumberColumn(format="%.2f"),
                    "Open Interest": st.column_config.NumberColumn(format="%d"),
                    "Distancia %": PCT,
                    "Vol. semana en zona %": st.column_config.NumberColumn(format="%.1f%%")})
            z = opciones.ZONA * 100
            st.caption(
                f"**CW** = call walls, **PW** = put walls: los 2 strikes con mayor Open Interest por lado, a "
                f"±25% del precio. La franja de cada muro es su zona de contacto (±{z:g}% del strike). "
                "Las líneas llegan hasta el cierre del día de vencimiento (límite DTE).")
            st.caption(
                "**Esta semana** (velas de 1h de los últimos 5 días hábiles): **Rompió ↑/↓** = empezó la "
                "semana de un lado del strike y ahora está del otro · **Probando** = está dentro de la "
                "zona · **Rebotó** = llegó a la zona y se alejó sin cruzar · **Sin tocar** = no llegó. "
                "A la derecha, el **perfil de volumen** de la semana (acciones transadas por nivel de "
                "precio); en color, el volumen dentro de la zona de cada muro, y **POC** = el nivel con "
                "más volumen. Los muros son el Open Interest de hoy: Yahoo no entrega el histórico, así "
                "que no se sabe si el muro ya estaba ahí al empezar la semana. Datos en vivo de Yahoo, "
                "cacheados 15 minutos.")

    # ---- fundamentales ----
    st.subheader("Fundamentales")
    t_anual, t_trim = st.tabs(["Anual", "Trimestral"])
    for tab, tipo, anual in [(t_anual, "anual", True), (t_trim, "trimestral", False)]:
        with tab:
            f = cargar_fund(ticker, tipo)
            if f is None:
                st.info("Sin datos fundamentales.")
            else:
                st.dataframe(tabla_fundamental(f, anual), width="stretch")
