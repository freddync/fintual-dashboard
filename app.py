"""
Dashboard Fintual — Megacap + Large Cap.

Corre local con:  streamlit run app.py   (o doble click en Dashboard.bat)
"""

import copy
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
import hmac
import importlib

import opciones
import backtest
import sectores

# Streamlit Cloud, al recibir un push, vuelve a ejecutar app.py pero puede seguir
# usando la versión anterior de los módulos que tenía en memoria
importlib.reload(opciones)
importlib.reload(sectores)
importlib.reload(backtest)
# cambia cuando cambia opciones.py: invalida lo cacheado con la versión anterior
VERSION_OPCIONES = hashlib.md5(open(opciones.__file__, "rb").read()
                               + open(sectores.__file__, "rb").read()
                               + open(backtest.__file__, "rb").read()).hexdigest()

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
# Contraseña de acceso
# ---------------------------------------------------------------------------
# La clave va en los secrets de Streamlit Cloud: `password = "..."` (arriba de la
# sección [seguimiento]). Si no está configurada, la app queda abierta.

def clave_configurada():
    for leer in (lambda: st.secrets["password"], lambda: st.secrets["seguimiento"]["password"]):
        try:
            return str(leer())
        except Exception:
            pass
    return None


def pedir_clave():
    clave = clave_configurada()
    if clave is None or st.session_state.get("_acceso_ok"):
        return
    st.title("📈 Fintual")
    with st.form("acceso"):
        intento = st.text_input("Contraseña", type="password")
        if st.form_submit_button("Entrar"):
            if hmac.compare_digest(intento.encode(), clave.encode()):
                st.session_state["_acceso_ok"] = True
                st.rerun()
            st.error("Contraseña incorrecta.")
    st.stop()


pedir_clave()


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


def senal_macd(linea, senal, hist):
    """Estado según el histograma (MACD − señal), si sube o baja respecto a la sesión
    anterior y si ambas líneas están bajo cero.

    Cruce alcista  el MACD cruzó sobre su señal en la última sesión, con ambas bajo cero
    Pre-cruce      histograma negativo pero subiendo, con ambas líneas bajo cero
    Alcista        histograma positivo y subiendo
    Perdiendo fuerza  histograma positivo pero cayendo
    Bajista        el resto (incluye histograma negativo subiendo con las líneas sobre
                   cero: eso no cuenta como pre-cruce)
    """
    if len(hist) < 2 or hist.iloc[-2:].isna().any():
        return "Sin datos"
    act, prev = hist.iloc[-1], hist.iloc[-2]
    bajo_cero = linea.iloc[-1] < 0 and senal.iloc[-1] < 0
    if prev < 0 <= act and bajo_cero:
        return "Cruce alcista"
    if act >= 0:
        return "Alcista" if act >= prev else "Perdiendo fuerza"
    return "Pre-cruce" if act >= prev and bajo_cero else "Bajista"


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
        return next((f for n, f in files.items() if n not in (LINEAS, CARTERA)), None)
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


CARTERA = "cartera.json"   # compras registradas a mano, en el mismo Gist
UMBRALES_VENTA = [60, 70, 80]
STOPS = [0, 10, 15, 20, 25]   # % de pérdida; 0 = sin stop


def leer_cartera():
    """{"posiciones": [...], "vendidas": [...], "umbral": 70}"""
    c = copy.deepcopy(leer_json(CARTERA)) or {}
    c.setdefault("posiciones", [])
    c.setdefault("vendidas", [])
    c.setdefault("umbral", 70)
    c.setdefault("stop", 20)
    return c


def guardar_cartera(c):
    guardar_json(CARTERA, c)


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
        m_lin, m_sen, hist = macd(close)
        fila = {"Ticker": t, "Nombre": i.get("name", t), "Universo": i.get("universe", "?"),
                "Sector": i.get("sector", "Sin clasificar"), "Precio": precio,
                COL_RSI: r, "Señal RSI": senal_rsi(r),
                "MACD hist %": hist.iloc[-1] / precio * 100, "Señal MACD": senal_macd(m_lin, m_sen, hist)}
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


def fmt_cantidad(x):
    """Cantidad de acciones con hasta 9 decimales, sin ceros sobrantes."""
    return f"{x:,.9f}".rstrip("0").rstrip(".")


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


@st.cache_data(ttl=3600, show_spinner="Calculando el benchmark SPY...")
def cargar_benchmark(fecha, version=""):
    try:
        return opciones.benchmark_desde("SPY", fecha), None
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
    # pestaña Señales
    "Fecha señal": st.column_config.DateColumn(format="DD-MM-YYYY"),
    "Compra": st.column_config.DateColumn(format="DD-MM-YYYY"),
    "RSI señal": st.column_config.NumberColumn(format="%.1f"),
    "Umbral": st.column_config.NumberColumn(format="< %d"),
    "Precio compra": st.column_config.NumberColumn(format="%.2f"),
    "Precio actual / venta": st.column_config.NumberColumn(format="%.2f"),
    "Retorno %": PCT,
    "vs SMA 200 %": st.column_config.NumberColumn(format="%+.1f%%"),
    "vs sector 5d": st.column_config.NumberColumn(
        format="%+.1f pp", help="Retorno de la acción en los 5 días previos a la señal menos el de sus pares del "
                                "mismo sector. Muy negativo (≤ −5 pp) = cayó mucho más que su sector: en el "
                                "backtest, esas señales tuvieron el rebote más fuerte."),
    # pestaña Mi cartera
    "Fecha compra": st.column_config.DateColumn(format="DD-MM-YYYY"),
    "Fecha venta": st.column_config.DateColumn(format="DD-MM-YYYY"),
    "Cantidad": st.column_config.NumberColumn(format="%.9f"),
    "Precio venta": st.column_config.NumberColumn(format="%.2f"),
    "Precio actual": st.column_config.NumberColumn(format="%.2f"),
    "Precio stop": st.column_config.NumberColumn(format="%.2f"),
    "Dist. al stop %": st.column_config.NumberColumn(format="%+.1f%%"),
    "Invertido": st.column_config.NumberColumn(format="$%,.2f"),
    "Valor actual": st.column_config.NumberColumn(format="$%,.2f"),
    "Ganancia $": st.column_config.NumberColumn(format="$%+,.2f"),
    "Ganancia %": PCT,
}


PESTANAS = ["la lista", f"{COL_RSI} < {RSI_BAJO}", "Seguimiento", "Hoy"]   # para el botón de volver


# Señales que se ordenan por su significado y no alfabéticamente
ORDEN_SENAL = {
    "Señal RSI": {"Sobreventa": 0, "Neutral": 1, "Sobrecompra": 2},
    "Señal MACD": {"Bajista": 0, "Perdiendo fuerza": 1, "Pre-cruce": 2, "Cruce alcista": 3, "Alcista": 4},
}
SIN_ORDEN = "(sin orden)"


def ordenar(tabla, key):
    """Controles "Ordenar por" de una tabla. El orden elegido se guarda en una clave
    propia de session_state: las claves de los widgets se borran cuando la tabla no
    se dibuja (al abrir una empresa), y así el orden sobrevive a la ida y vuelta."""
    guardado = st.session_state.setdefault(f"_orden_{key}", [SIN_ORDEN, False])
    opciones_ = [SIN_ORDEN] + list(tabla.columns)
    o1, o2, _ = st.columns([2, 1, 3], vertical_alignment="bottom")
    col = o1.selectbox("Ordenar por", opciones_, key=f"w_col_{key}",
                       index=opciones_.index(guardado[0]) if guardado[0] in opciones_ else 0)
    asc = o2.toggle("Ascendente", value=guardado[1], key=f"w_asc_{key}")
    st.session_state[f"_orden_{key}"] = [col, asc]
    if col == SIN_ORDEN:
        return tabla
    clave = (lambda s: s.map(ORDEN_SENAL[col])) if col in ORDEN_SENAL else None
    return tabla.sort_values(col, ascending=asc, na_position="last", kind="mergesort", key=clave)


def mostrar_tabla(tabla, key, pestana, height=650, colorear=None):
    """Tabla ordenable; al marcar la casilla de una fila se abre esa empresa.
    La vista y la `pestana` (índice dentro de General) quedan anotadas para que el
    botón de volver regrese a ellas. `colorear` = columnas en verde/rojo según su signo."""
    tabla = ordenar(tabla, key).reset_index(drop=True)
    datos = tabla.style.map(colorear_num, subset=colorear) if colorear else tabla
    sel = st.dataframe(datos, hide_index=True, width="stretch", height=height,
                       on_select="rerun", selection_mode="single-row", key=key,
                       column_config=COLUMNAS)
    if sel.selection.rows:
        st.session_state["_abrir"] = tabla.iloc[sel.selection.rows[0]]["Ticker"]
        st.session_state["_pestana_origen"] = pestana
        st.session_state["_vista_origen"] = st.session_state.get("vista", "General")
        del st.session_state[key]   # limpia la selección para la vuelta
        st.rerun()


# ---------------------------------------------------------------------------
# Rotación sectorial
# ---------------------------------------------------------------------------

N_CALOR = 20   # períodos del mapa de calor
COLOR_FLUJO = {"Entrando": "#3ecf8e", "Saliendo": "#ef5a6f"}
UNIDAD_FREC = {"Diaria": "días", "Semanal": "sem.", "Mensual": "meses"}
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


@st.cache_data(show_spinner="Comparando el mismo tramo de cada período...")
def datos_tramos(periodo, n, universos, _sello_key, version=""):
    """Ranking y participación de volumen por sector en el mismo tramo del período en curso y los n anteriores."""
    info_ = cargar_info()
    precios = {t: df for t, i in info_.items() if i.get("universe") in universos and (df := cargar_precios(t)) is not None}
    cierre, dolares = sectores.paneles(precios, "Diaria")
    return sectores.tramos(cierre, dolares, {t: info_[t].get("sector", "Sin clasificar") for t in precios}, periodo, n)


def color_rank(n_total):
    """Fondo verde para los primeros puestos y rojo para los últimos (sin matplotlib)."""
    def f(v):
        if pd.isna(v):
            return ""
        x = (v - 1) / max(n_total - 1, 1)            # 0 = primero, 1 = último
        if x <= 0.5:
            return f"background-color: rgba(62,207,142,{0.55 * (1 - 2 * x):.2f})"
        return f"background-color: rgba(239,90,111,{0.55 * (2 * x - 1):.2f})"
    return f


@st.cache_data(show_spinner="Corriendo el backtest...")
def correr_backtest(compra, venta, costo, universos, _sello_key, version="", entrada="rsi", ventana=10,
                    salida="rsi"):
    """Operaciones de la regla elegida en todas las empresas del universo, con el estado
    de su sector (RRG diario, flujo de volumen) el día de la señal."""
    info_ = cargar_info()
    precios = {t: df for t, i in info_.items()
               if i.get("universe") in universos and (df := cargar_precios(t)) is not None}
    ops = backtest.correr(precios, {t: info_[t].get("sector", "Sin clasificar") for t in precios},
                          RSI_PERIOD, compra, venta, costo, entrada, ventana, salida)
    if ops.empty:
        return ops
    s, ratio, mom = datos_sectores("Diaria", universos, _sello_key, version)
    return backtest.agregar_sector(ops, s, ratio, mom)


# Configuración de la señal elegida en el backtest (MACD golden cross por abajo + RSI < umbral
# el mismo día; venta con RSI > 70). Umbral por sector según la comparación 20/30/40.
UMBRAL_SECTOR = {
    "Tecnología": 20, "Industrial": 20, "Consumo discrecional": 20, "Financiero": 20,
    "Bienes raíces": 30, "Energía": 30, "Consumo básico": 30,
    "Servicios públicos": 40, "Materiales": 40,
}   # Salud y Servicios de comunicación quedan fuera: la señal no les agregó valor
CUADRANTES_EXCLUIDOS = {"Debilitándose"}   # sector perdiendo fuerza: la señal rinde la mitad
# Filtro SMA 200 por tipo de acción: las cíclicas rinden mejor si la señal ocurre BAJO su media de 200
# días (caída fuerte que rebota); las defensivas y de materias primas, solo si siguen SOBRE ella (si
# cayeron bajo la media suelen seguir cayendo). Backtest: Sharpe 1,46 -> 1,92 y caída máx. -9% -> -7%.
CICLICOS = {"Tecnología", "Industrial", "Consumo discrecional", "Financiero"}
SMA_LARGA = 200


def con_estado_sector(df, universos):
    """Agrega 'Cuadrante sector' y 'Flujo sector' (estado actual, RRG diario de la última sesión)
    justo después de la columna Sector. Mismo criterio que la pestaña Hoy (⭐ = Mejorando)."""
    s, ratio, mom = datos_sectores("Diaria", universos, json.dumps(sello), VERSION_OPCIONES)
    f = ratio.dropna(how="all").index[-1]
    cuad = {sec: sectores.cuadrante(ratio.at[f, sec], mom.at[f, sec]) for sec in ratio.columns}
    dp = s["d_part"].loc[f]
    flujo = {sec: "Entrando" if v > 0 else "Saliendo" if v <= 0 else "Sin datos" for sec, v in dp.items()}
    df = df.copy()
    i = df.columns.get_loc("Sector") + 1
    df.insert(i, "Cuadrante sector", [("⭐ " if cuad.get(x) == "Mejorando" else "") + cuad.get(x, "Sin datos")
                                      for x in df["Sector"]])
    df.insert(i + 1, "Flujo sector", [flujo.get(x, "Sin datos") for x in df["Sector"]])
    return df


REZAGO_FUERTE = -5   # pp bajo el sector en 5 días: en el backtest, el grupo de mejor rebote


@st.cache_data(show_spinner=False)
def vs_sector_5d(universos, _sello_key, version=""):
    """DataFrame fechas x tickers: retorno 5 días de la acción − retorno 5 días de sus pares del
    mismo sector (excluyéndola, igual peso), en puntos porcentuales."""
    info_ = cargar_info()
    tickers = [t for t, i in info_.items() if i.get("universe") in universos]
    C = pd.DataFrame({t: df.set_index("Date")["Close"] for t in tickers
                      if (df := cargar_precios(t)) is not None}).sort_index()
    R = C.pct_change(5, fill_method=None) * 100
    out = pd.DataFrame(index=R.index, columns=R.columns, dtype=float)
    sec_de = {t: info_[t].get("sector", "Sin clasificar") for t in R.columns}
    for s_ in set(sec_de.values()):
        cols = [t for t in R.columns if sec_de[t] == s_]
        sub = R[cols]
        suma, n = sub.sum(axis=1, min_count=1), sub.notna().sum(axis=1)
        pares = (suma.values[:, None] - sub.values) / (n - 1).where(n > 1).values[:, None]
        out[cols] = sub.values - pares
    return out


@st.cache_data(show_spinner="Buscando señales...")
def senales_config(universos, _sello_key, version=""):
    """Todas las señales de la configuración, cada una con su seguimiento (abierta, vendida...)."""
    partes = []
    for u in sorted(set(UMBRAL_SECTOR.values())):
        o = correr_backtest(u, RSI_ALTO, 0.1, universos, _sello_key, version, "macd_y_rsi", 10)
        if not o.empty:
            sect = [x for x, v in UMBRAL_SECTOR.items() if v == u]
            partes.append(o[o["Sector"].isin(sect)].assign(Umbral=u))
    if not partes:
        return pd.DataFrame()
    o = pd.concat(partes)
    o = o[~o["Cuadrante sector"].isin(CUADRANTES_EXCLUIDOS)].copy()
    # distancia del cierre del día de la señal a la SMA 200
    dist, smas = [], {}
    for t, f, c_ in zip(o["Ticker"], o["Señal"], o["Cierre señal"]):
        if t not in smas:
            df_t = cargar_precios(t)
            smas[t] = (df_t.set_index("Date")["Close"].rolling(SMA_LARGA).mean() if df_t is not None
                       else pd.Series(dtype=float))
        sma = smas[t].get(f, np.nan)
        dist.append((c_ / sma - 1) * 100 if pd.notna(sma) and sma else np.nan)
    o["vs SMA 200 %"] = dist
    ciclica = o["Sector"].isin(CICLICOS)
    o = o[(ciclica & (o["vs SMA 200 %"] < 0)) | (~ciclica & (o["vs SMA 200 %"] > 0))]
    rel = vs_sector_5d(universos, _sello_key, version)
    o["vs sector 5d"] = [rel.at[f, t] if (f in rel.index and t in rel.columns) else np.nan
                         for t, f in zip(o["Ticker"], o["Señal"])]
    return o.sort_values(["Señal", "Ticker"], ascending=[False, True])


def estado_senal(fila, provisional):
    if fila["Pendiente"]:
        return "🟡 Provisional (sesión en curso)" if provisional else "🟢 Comprar en la apertura"
    if fila["Abierta"]:
        return "🔴 Vender en la apertura" if fila["Venta pendiente"] else "🔵 Abierta"
    return "⚪ Vendida"


def tabla_senales(df, nombres, provisional, ult):
    """Tabla para mostrar señales de la configuración."""
    return pd.DataFrame({
        "Ticker": df["Ticker"],
        "Nombre": df["Ticker"].map(nombres),
        "Sector": df["Sector"],
        "Fecha señal": df["Señal"],
        "Estado": [estado_senal(f, provisional and f["Señal"] == ult) for _, f in df.iterrows()],
        "RSI señal": df["RSI señal"],
        "Umbral": df["Umbral"],
        "Cuadrante sector": [("⭐ " if q == "Mejorando" else "") + q for q in df["Cuadrante sector"]],
        "Flujo sector": df["Flujo sector"],
        "vs SMA 200 %": df["vs SMA 200 %"],
        "vs sector 5d": df["vs sector 5d"],
        "Compra": df["Entrada"],
        "Precio compra": df["Precio entrada"],
        "Precio actual / venta": df["Precio salida"],
        "Retorno %": df["Retorno %"],
        "Días": df["Días"],
    })


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


def grafico_riesgo_macro(od, macro, desde):
    """Ofensivos/defensivos arriba y, debajo, una fila por variable macro (mismo eje de tiempo)."""
    filas = 1 + len(macro)
    fig = make_subplots(rows=filas, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.32] + [0.68 / len(macro)] * len(macro) if macro else [1],
                        subplot_titles=["Ofensivos / defensivos (base 100)"] + list(macro))
    base = od[od.index >= desde]
    serie = base / base.iloc[0] * 100
    media = (od.rolling(50).mean() / base.iloc[0] * 100)[od.index >= desde]
    fig.add_trace(go.Scatter(x=serie.index, y=serie, line=dict(color="#5ea8e6", width=1.8),
                             name="Ofensivos / defensivos"), row=1, col=1)
    fig.add_trace(go.Scatter(x=media.index, y=media, line=dict(color="#8b93a3", width=1, dash="dot"),
                             name="Promedio 50 días"), row=1, col=1)
    fig.add_hline(y=100, line=dict(color="#2a2f3a", width=1), row=1, col=1)
    colores = ["#e6b45e", "#b98af0", "#ef5a6f", "#3ecf8e"]
    for i, (nombre, s) in enumerate(macro.items(), start=2):
        s = s[s.index >= desde]
        fig.add_trace(go.Scatter(x=s.index, y=s, line=dict(color=colores[(i - 2) % 4], width=1.4),
                                 name=nombre), row=i, col=1)
    fig.update_xaxes(range=[desde, max(od.index[-1], *(s.index[-1] for s in macro.values()))])
    fig.update_annotations(font_size=12, x=0, xanchor="left")
    fig.update_layout(height=260 + 150 * len(macro), template="plotly_dark", showlegend=False,
                      margin=dict(t=30, b=10), hovermode="x unified")
    return fig


@st.cache_data(ttl=3600, show_spinner="Descargando tasas, petróleo, VIX y dólar...")
def cargar_macro(version=""):
    """{nombre: serie diaria} de las variables macro. Devuelve (series, errores)."""
    series, errores = {}, []
    for nombre, simbolo in opciones.MACRO.items():
        try:
            series[nombre] = opciones.serie_diaria(simbolo)
        except Exception as e:
            errores.append(f"{nombre}: {type(e).__name__}")
    return series, ", ".join(errores)


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
    st.session_state["vista"] = st.session_state.get("_vista_origen", "General")
    if st.session_state["vista"] == "General":
        st.session_state["_ir_a_pestana"] = st.session_state.get("_pestana_origen", 0)

st.sidebar.title("📈 Fintual")
vista = st.sidebar.radio("Vista", ["General", "Sectores", "Mi cartera", "Empresa"], horizontal=True, key="vista")
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
if clave_configurada() is None and config_gist()[0]:
    st.sidebar.warning("La app no tiene contraseña: cualquiera con el link puede verla y modificar tu cartera. "
                       "Agrega `password = \"...\"` en los Secrets de Streamlit Cloud.")


# ---------------------------------------------------------------------------
# Vista general
# ---------------------------------------------------------------------------

if vista == "General":
    st.title("Resumen general")
    st.caption("Marca la casilla de una fila para abrir la empresa · usa **Ordenar por** para que el orden "
               "se mantenga al volver de una empresa (el click en el título de una columna también ordena, "
               f"pero ese orden se pierde). {COL_RSI}: sobre {RSI_ALTO} = sobrecompra, bajo {RSI_BAJO} = sobreventa · "
               f"MACD {MACD_FAST}/{MACD_SLOW}/{MACD_SIGNAL}: **Cruce alcista** = el MACD cruzó sobre su señal en la última sesión con ambas líneas bajo cero · **Pre-cruce** = histograma negativo pero subiendo, también con ambas bajo cero.")

    sobreventa = filtrado[filtrado[COL_RSI] < RSI_BAJO]
    etiquetas = ["Todas", f"{COL_RSI} < {RSI_BAJO} ({len(sobreventa)})", "⭐ Seguimiento", "🛒 Hoy"]
    # al volver desde una empresa se abre la pestaña de donde se vino
    destino = st.session_state.pop("_ir_a_pestana", None)
    if destino is not None and destino < len(etiquetas):
        st.session_state["pestanas"] = etiquetas[destino]
    if st.session_state.get("pestanas") not in etiquetas:   # el conteo del título cambió
        st.session_state.pop("pestanas", None)
    tab_todas, tab_rsi, tab_seg, tab_hoy = st.tabs(etiquetas, key="pestanas", on_change="rerun")

    with tab_todas:
        mostrar_tabla(con_estado_sector(filtrado, tuple(sorted(sel_uni))), "tabla_general", 0)
        st.caption("P/E con EPS diluido de los últimos 4 trimestres. EV/EBITDA con el último año fiscal. "
                   "Crecimiento y margen del último año fiscal.")

    with tab_rsi:
        if sobreventa.empty:
            st.info(f"Ninguna empresa del filtro actual tiene {COL_RSI} bajo {RSI_BAJO}.")
        else:
            mostrar_tabla(con_estado_sector(sobreventa.sort_values(COL_RSI), tuple(sorted(sel_uni))), "tabla_rsi", 1)

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
            t = con_estado_sector(t, tuple(sorted(sel_uni)))
            mostrar_tabla(t, "tabla_seg", 2, height=min(650, 38 + 35 * len(t)))
            st.caption("**Cuadrante sector** y **Flujo sector**: estado actual del sector de cada empresa en la "
                       "rotación diaria (vista Sectores). ⭐ Mejorando = el contexto en que mejor rindió la señal; "
                       "Flujo = si el sector está ganando o perdiendo participación en el volumen transado.")
        if config_gist()[0]:
            st.caption("✅ Guardado en GitHub Gist: se mantiene entre sesiones y dispositivos. "
                       "No depende de los filtros del panel izquierdo.")
        else:
            st.caption("⚠️ Guardado en data/seguimiento.json (archivo local). En Streamlit Cloud se pierde "
                       "al redeployar: configura los secrets [seguimiento] gist_id y token.")

    with tab_hoy:
        if st.session_state.get("pestanas") != "🛒 Hoy":
            st.caption("Abre esta pestaña para calcular las órdenes del día.")
        else:
            sen = senales_config(tuple(sorted(sel_uni)), json.dumps(sello), VERSION_OPCIONES)
            if sector != "Todos" and not sen.empty:
                sen = sen[sen["Sector"] == sector]
            if sen.empty:
                st.info("No hay señales con la configuración y los filtros actuales.")
            else:
                ult = sen["Señal"].max()
                provisional = periodo_en_curso(ult, "Diaria")
                nombres = data.set_index("Ticker")["Nombre"]
                st.warning(
                    "Esta lista es el resultado **mecánico de tu configuración** (la que salió del backtest), no una "
                    "recomendación de inversión: no soy asesor financiero. Antes de operar revisa cada empresa, "
                    "confirma que esté disponible en Fintual y decide el monto según tu propio criterio.")
                if provisional:
                    st.info(f"La sesión del {ult:%d-%b} sigue abierta: las señales de hoy son **provisionales** y "
                            "pueden desaparecer antes del cierre. Las definitivas quedan después de las 16:00 NY.")

                compras = sen[sen["Pendiente"]].copy()
                compras["_orden"] = (compras["Cuadrante sector"] != "Mejorando").astype(int)
                compras = compras.sort_values(["_orden", "RSI señal"])
                ventas = sen[sen["Venta pendiente"]]
                k = st.columns(3)
                k[0].metric("Para comprar", f"{len(compras)}", f"señales del {ult:%d-%b}", delta_color="off")
                k[1].metric("Para vender", f"{len(ventas)}", f"{COL_RSI} sobre {RSI_ALTO}", delta_color="off")
                k[2].metric("Posiciones que siguen abiertas",
                            f"{(sen['Abierta'] & ~sen['Pendiente'] & ~sen['Venta pendiente']).sum()}")

                st.subheader("🟢 Comprar")
                if compras.empty:
                    st.caption("Ninguna empresa cumple la regla de compra en la última sesión.")
                else:
                    tc = tabla_senales(compras, nombres, provisional, ult)
                    tc = tc[["Ticker", "Nombre", "Sector", "Estado", "vs sector 5d", "RSI señal", "Umbral",
                             "vs SMA 200 %", "Cuadrante sector", "Flujo sector", "Precio actual / venta"]].rename(
                        columns={"Precio actual / venta": "Último cierre"})
                    mostrar_tabla(tc, "tabla_compras", 3, height=min(450, 38 + 35 * len(tc)))
                    st.caption(f"Regla: la compra se hace en la **apertura siguiente** a la señal del {ult:%d-%b}. "
                               "Primero van las de sector ⭐ Mejorando (el contexto que mejor rindió) y luego por "
                               f"{COL_RSI} más bajo. **Último cierre** es referencial: el precio real será el de la "
                               "apertura.")
                    fuertes = tc.loc[tc["vs sector 5d"] <= REZAGO_FUERTE, "Ticker"].tolist()
                    st.caption(f"**vs sector 5d** = cuánto más (o menos) cayó la acción que sus pares del sector en la "
                               f"última semana. En el backtest, las señales con {REZAGO_FUERTE} pp o menos rindieron 2-3 "
                               "veces más que el resto (+7,7% de exceso por operación contra ~2-3%), así que sirve "
                               "para elegir si no puedes tomar todas. "
                               + (f"Hoy cumplen: **{', '.join(fuertes)}**." if fuertes else "Hoy ninguna cumple."))

                st.subheader("🔴 Vender")
                if ventas.empty:
                    st.caption(f"Ninguna posición abierta tiene el {COL_RSI} sobre {RSI_ALTO}.")
                else:
                    tv = tabla_senales(ventas, nombres, provisional, ult)
                    tv = tv[["Ticker", "Nombre", "Sector", "Fecha señal", "Compra", "Precio compra",
                             "Precio actual / venta", "Retorno %", "Días"]].rename(
                        columns={"Precio actual / venta": "Último cierre"})
                    mostrar_tabla(tv, "tabla_ventas", 3, height=min(450, 38 + 35 * len(tv)))
                    st.caption(f"Posiciones de la regla cuyo {COL_RSI} cerró sobre {RSI_ALTO}: la venta se hace en "
                               "la apertura siguiente. Solo aplica si compraste esa posición siguiendo la regla.")


# ---------------------------------------------------------------------------
# Vista mi cartera
# ---------------------------------------------------------------------------

elif vista == "Mi cartera":
    st.title("Mi cartera")
    st.caption("Las acciones que compraste, con su ganancia y una señal de venta cuando el RSI supera tu "
               "umbral.")
    cart = leer_cartera()
    precios_hoy = data.set_index("Ticker")

    # ---- tus reglas de venta (se guardan) ----
    u1, u3, u2 = st.columns([2, 3, 4], vertical_alignment="bottom")
    umbral_cart = u1.radio(f"Tu señal de venta: {COL_RSI} >", UMBRALES_VENTA,
                           index=UMBRALES_VENTA.index(cart["umbral"]) if cart["umbral"] in UMBRALES_VENTA else 1,
                           horizontal=True, key="umbral_cartera")
    if umbral_cart != cart["umbral"]:
        cart["umbral"] = umbral_cart
        guardar_cartera(cart)
        st.rerun()
    stop_cart = u3.radio("Stop de pérdida", STOPS,
                         index=STOPS.index(cart["stop"]) if cart["stop"] in STOPS else 3,
                         format_func=lambda x: "Sin stop" if x == 0 else f"−{x}%", horizontal=True, key="stop_cartera")
    if stop_cart != cart["stop"]:
        cart["stop"] = stop_cart
        guardar_cartera(cart)
        st.rerun()
    u2.caption(f"En el backtest, {COL_RSI} > 70 fue el mejor equilibrio para vender. Un stop amplio de −20% casi "
               "no restó rentabilidad y limitó la peor pérdida de −57% a −20%; stops ajustados (−5% a −10%) la "
               "empeoraron mucho.")

    # ---- posiciones abiertas ----
    if not cart["posiciones"]:
        st.info("Todavía no registras compras. Agrégalas abajo, en **Registrar una compra**.")
    else:
        filas_c = []
        for pos in cart["posiciones"]:
            t = pos["ticker"]
            act = precios_hoy["Precio"].get(t, np.nan)
            rsi_t = precios_hoy[COL_RSI].get(t, np.nan)
            invertido = pos["cantidad"] * pos["precio"]
            valor = pos["cantidad"] * act
            cruzados = [u for u in UMBRALES_VENTA if pd.notna(rsi_t) and rsi_t > u]
            # stop: si desde la compra algún mínimo diario tocó el nivel (o el precio actual está bajo él)
            nivel = pos["precio"] * (1 - stop_cart / 100) if stop_cart else np.nan
            hist_t = cargar_precios(t)
            minimo = (hist_t.loc[hist_t["Date"] >= pd.Timestamp(pos["fecha"]), "Low"].min()
                      if hist_t is not None else np.nan)
            toco_stop = bool(stop_cart) and ((pd.notna(minimo) and minimo <= nivel) or (pd.notna(act) and act <= nivel))
            if toco_stop:
                senal = f"🔴 VENDER (tocó stop −{stop_cart}%)"
            elif pd.notna(rsi_t) and rsi_t > umbral_cart:
                senal = f"🔴 VENDER ({COL_RSI} > {umbral_cart})"
            elif cruzados:
                senal = f"🟠 {COL_RSI} > {max(cruzados)} (tu umbral es {umbral_cart})"
            else:
                senal = "🟢 Mantener"
            filas_c.append({
                "Ticker": t, "Nombre": precios_hoy["Nombre"].get(t, t),
                "Ganancia $": valor - invertido, "Ganancia %": (act / pos["precio"] - 1) * 100,
                "Señal de venta": senal, COL_RSI: rsi_t,
                "Fecha compra": pd.Timestamp(pos["fecha"]),
                "Días": (pd.Timestamp.now().normalize() - pd.Timestamp(pos["fecha"])).days,
                "Cantidad": pos["cantidad"], "Precio compra": pos["precio"], "Precio actual": act,
                "Precio stop": nivel, "Dist. al stop %": (act / nivel - 1) * 100 if stop_cart else np.nan,
                "Invertido": invertido, "Valor actual": valor,
                "Señal MACD": precios_hoy["Señal MACD"].get(t, ""),
            })
        tc = pd.DataFrame(filas_c)
        vender_ya = tc["Señal de venta"].str.startswith("🔴")
        if vender_ya.any():
            detalle = [f"{f_['Ticker']} ({f_['Señal de venta'].split('(')[1].rstrip(')')})"
                       for _, f_ in tc[vender_ya].iterrows()]
            st.error(f"**Señal de venta en {vender_ya.sum()} posición(es):** " + ", ".join(detalle) +
                     ". Según las reglas del backtest, la venta por RSI es en la próxima apertura; la del stop, "
                     "apenas el precio toca el nivel.")
        k = st.columns(4)
        inv, val = tc["Invertido"].sum(), tc["Valor actual"].sum()
        k[0].metric("Posiciones abiertas", f"{len(tc)}")
        k[1].metric("Invertido", f"${inv:,.2f}")
        k[2].metric("Valor actual", f"${val:,.2f}", f"{(val / inv - 1) * 100:+.2f}%" if inv else None)
        k[3].metric("Ganancia no realizada", f"${val - inv:+,.2f}")
        mostrar_tabla(tc, "tabla_cartera", 0, height=min(450, 38 + 35 * len(tc)),
                  colorear=["Ganancia $", "Ganancia %"])
        st.caption(f"**Precio actual** y **{COL_RSI}** = último dato de la actualización automática (se refresca "
                   "cada hora en horario de mercado; durante la sesión la barra del día es parcial). 🔴 = el RSI "
                   "superó tu umbral, o el precio tocó tu stop (se revisa el mínimo de cada día desde la compra) · "
                   f"🟠 = el RSI superó otro de los umbrales ({', '.join(map(str, UMBRALES_VENTA))}) pero no el tuyo. "
                   "**Dist. al stop** = cuánto le falta al precio para tocarlo. Ojo: el stop no protege de los saltos "
                   "de un día para otro (si la acción abre bajo el stop, la venta será a ese precio menor). Montos en "
                   "la moneda en que registraste el precio.")

    # ---- comparación con SPY ----
    todas = cart["posiciones"] + cart["vendidas"]
    if todas:
        st.subheader("Tu cartera vs. SPY")
        invertido_total = sum(x["cantidad"] * x["precio"] for x in todas)
        valor_abiertas = sum(x["cantidad"] * (precios_hoy["Precio"].get(x["ticker"], np.nan)
                                              if pd.notna(precios_hoy["Precio"].get(x["ticker"], np.nan))
                                              else x["precio"]) for x in cart["posiciones"])
        cobrado_ventas = sum(x["cantidad"] * x["precio_venta"] for x in cart["vendidas"])
        valor_cartera = valor_abiertas + cobrado_ventas
        fecha0 = min(pd.Timestamp(x["fecha"]) for x in todas)
        bench, err_b = cargar_benchmark(fecha0.date().isoformat(), VERSION_OPCIONES)
        if err_b:
            st.caption(f"No se pudo calcular el benchmark SPY: {err_b}")
        else:
            f_spy, apertura_spy, factor_spy, ultimo_spy = bench
            valor_spy = invertido_total * factor_spy
            ret_c = (valor_cartera / invertido_total - 1) * 100
            ret_s = (factor_spy - 1) * 100
            b = st.columns(3)
            b[0].metric("Tu cartera", f"${valor_cartera:,.2f}", f"{ret_c:+.2f}%")
            b[1].metric(f"SPY desde el {f_spy:%d-%m-%Y}", f"${valor_spy:,.2f}", f"{ret_s:+.2f}%")
            b[2].metric("Diferencia", f"${valor_cartera - valor_spy:+,.2f}", f"{ret_c - ret_s:+.2f} pp",
                        help="Positivo = tu cartera le va ganando a SPY.")
            if ret_c >= ret_s:
                st.success(f"Le vas ganando a SPY por **{ret_c - ret_s:.2f} puntos**.")
            else:
                st.warning(f"SPY te va ganando por **{ret_s - ret_c:.2f} puntos**.")
            st.caption(
                f"**Benchmark:** los mismos ${invertido_total:,.2f} que invertiste en total, comprados de una vez "
                f"en SPY a la apertura del {f_spy:%d-%m-%Y} (tu compra más antigua; precio US$ {apertura_spy:,.2f}), "
                f"con dividendos reinvertidos. Último precio SPY: US$ {ultimo_spy:,.2f}. **Tu cartera:** valor actual "
                "de las posiciones abiertas + lo cobrado en ventas (no incluye impuestos ni costos de cambio). "
                "Ojo: el benchmark supone todo el dinero invertido desde el primer día; si fuiste comprando de a "
                "poco, SPY tuvo más tiempo el dinero trabajando, así que la comparación es exigente para tu cartera.")

    # ---- registrar una compra ----
    with st.expander("➕ Registrar una compra", expanded=not cart["posiciones"]):
        with st.form("nueva_compra", clear_on_submit=True):
            f1, f2, f3, f4 = st.columns(4)
            t_new = f1.selectbox("Ticker", sorted(data["Ticker"]), index=None, placeholder="Escribe un ticker...")
            fecha_new = f2.date_input("Fecha de compra", datetime.date.today())
            cant_new = f3.number_input("Cantidad (acciones)", min_value=0.0, value=None, step=1.0,
                                       format="%.9f", help="Acepta fracciones de acción, hasta 9 decimales.")
            precio_new = f4.number_input("Precio de compra", min_value=0.0, value=None, step=0.01, format="%.4f")
            if st.form_submit_button("Guardar compra"):
                if not t_new or not cant_new or not precio_new or cant_new <= 0 or precio_new <= 0:
                    st.error("Completa el ticker, una cantidad mayor a 0 y el precio de compra.")
                else:
                    cart["posiciones"].append({
                        "id": datetime.datetime.now().strftime("%Y%m%d%H%M%S%f"),
                        "ticker": t_new, "fecha": fecha_new.isoformat(),
                        "cantidad": float(cant_new), "precio": float(precio_new)})
                    guardar_cartera(cart)
                    st.rerun()

    # ---- registrar una venta / eliminar ----
    if cart["posiciones"]:
        with st.expander("➖ Registrar una venta o eliminar un registro"):
            etiqueta_pos = {p_["id"]: f"{p_['ticker']} · {fmt_cantidad(p_['cantidad'])} acc. a {p_['precio']:,.2f} "
                                      f"({p_['fecha']})" for p_ in cart["posiciones"]}
            sel_id = st.selectbox("Posición", list(etiqueta_pos), format_func=etiqueta_pos.get)
            pos_sel = next(p_ for p_ in cart["posiciones"] if p_["id"] == sel_id)
            with st.form("venta"):
                v1, v2 = st.columns(2)
                precio_v = v1.number_input("Precio de venta", min_value=0.0, step=0.01, format="%.4f",
                                           value=float(precios_hoy["Precio"].get(pos_sel["ticker"], 0) or 0))
                fecha_v = v2.date_input("Fecha de venta", datetime.date.today())
                b1, b2 = st.columns(2)
                vendida = b1.form_submit_button("Registrar venta")
                borrar = b2.form_submit_button("Eliminar (fue un error)")
            if vendida and precio_v > 0:
                cart["posiciones"] = [p_ for p_ in cart["posiciones"] if p_["id"] != sel_id]
                cart["vendidas"].append({**pos_sel, "precio_venta": float(precio_v),
                                         "fecha_venta": fecha_v.isoformat()})
                guardar_cartera(cart)
                st.rerun()
            if borrar:
                cart["posiciones"] = [p_ for p_ in cart["posiciones"] if p_["id"] != sel_id]
                guardar_cartera(cart)
                st.rerun()

    # ---- historial de ventas ----
    if cart["vendidas"]:
        st.subheader("Historial de ventas")
        hv = pd.DataFrame([{
            "Ticker": v["ticker"], "Nombre": precios_hoy["Nombre"].get(v["ticker"], v["ticker"]),
            "Ganancia $": v["cantidad"] * (v["precio_venta"] - v["precio"]),
            "Ganancia %": (v["precio_venta"] / v["precio"] - 1) * 100,
            "Fecha compra": pd.Timestamp(v["fecha"]), "Fecha venta": pd.Timestamp(v["fecha_venta"]),
            "Días": (pd.Timestamp(v["fecha_venta"]) - pd.Timestamp(v["fecha"])).days,
            "Cantidad": v["cantidad"], "Precio compra": v["precio"], "Precio venta": v["precio_venta"],
            "Invertido": v["cantidad"] * v["precio"],
        } for v in cart["vendidas"]]).sort_values("Fecha venta", ascending=False)
        k = st.columns(3)
        k[0].metric("Ventas", f"{len(hv)}", f"{(hv['Ganancia %'] > 0).mean() * 100:.0f}% con ganancia",
                    delta_color="off")
        k[1].metric("Ganancia realizada", f"${hv['Ganancia $'].sum():+,.2f}")
        k[2].metric("Retorno prom. por venta", f"{hv['Ganancia %'].mean():+.2f}%")
        st.dataframe(hv.style.map(colorear_num, subset=["Ganancia $", "Ganancia %"]), hide_index=True,
                 width="stretch", column_config=COLUMNAS)

    if config_gist()[0]:
        st.caption("✅ Tu cartera se guarda en tu GitHub Gist privado: se mantiene entre sesiones y dispositivos.")
    else:
        st.caption("⚠️ Guardada en data/cartera.json (archivo local). En Streamlit Cloud se pierde al "
                   "redeployar: configura los secrets [seguimiento] gist_id y token.")


# ---------------------------------------------------------------------------
# Vista sectores (rotación)
# ---------------------------------------------------------------------------

elif vista == "Sectores":
    st.title("Rotación sectorial")
    st.caption("Cómo se mueven los sectores entre sí, en precio y en volumen. Pesos iguales: cada empresa "
               "pesa lo mismo dentro de su sector. 'Mercado' = promedio de todas las empresas del universo "
               "elegido a la izquierda (el filtro de sector no aplica aquí).")

    pest_sect = ["🔄 Rotación sectorial", "📅 Semana, mes y día a día", "🧭 Qué está moviendo la rotación"]
    if st.session_state.get("tabs_sectores") not in pest_sect:
        st.session_state.pop("tabs_sectores", None)
    tab_rot, tab_tramo, tab_mueve = st.tabs(pest_sect, key="tabs_sectores", on_change="rerun")

    with tab_rot:
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
                    .map(lambda v: f"color: {COLOR_CUADRANTE.get(v, '')}; font-weight: 600", subset=["Cuadrante"])
                    .map(lambda v: f"color: {COLOR_FLUJO.get(v, '')}; font-weight: 600", subset=["Flujo"]),
            hide_index=True, width="stretch", column_config={
                "Retorno %": st.column_config.NumberColumn(format="%+.2f%%"),
                "vs mercado (pp)": st.column_config.NumberColumn(format="%+.2f"),
                "% al alza": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
                "Vol. relativo": st.column_config.NumberColumn(format="%.2fx"),
                "Part. volumen %": st.column_config.NumberColumn(format="%.1f%%"),
                "Δ participación (pp)": st.column_config.NumberColumn(format="%+.2f"),
                "Racha": st.column_config.NumberColumn(
                    format=f"%d {UNIDAD_FREC[frec]}",
                    help="Períodos seguidos con el mismo flujo. Uno solo puede ser ruido; varios seguidos es más confiable."),
                "Relativo 1m %": st.column_config.NumberColumn(format="%+.1f%%"),
                "Relativo 3m %": st.column_config.NumberColumn(format="%+.1f%%"),
            })
        st.caption(
            "**vs mercado** = retorno del sector menos el del mercado · **% al alza** = parte de las empresas del "
            "sector que subió (amplitud) · **Vol. relativo** = volumen en dólares del período / su promedio de los "
            f"{sectores.VENTANA_VOL[frec]} anteriores · **Part. volumen** = parte del volumen en dólares del mercado que "
            "se transó en el sector · **Δ participación** = participación actual menos su promedio: positivo = "
            "está entrando más dinero que lo habitual · **Flujo** = **Entrando** si la Δ participación es positiva, **Saliendo** si es negativa; la **Racha** dice cuántos períodos seguidos lleva así · **Relativo 1m/3m** = cuánto le ganó (o perdió) al "
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

    with tab_tramo:
        # se calcula solo con la pestaña abierta
        if st.session_state.get("tabs_sectores") != pest_sect[1]:
            st.caption("Abre esta pestaña para calcular la comparación.")
        else:
            c1, c2 = st.columns([1, 2])
            per = c1.radio("Comparar", ["Mes", "Semana", "Día"], horizontal=True, key="tramo_periodo")
            n_per = c2.select_slider("Períodos anteriores", [3, 4, 5, 6, 8, 10, 12, 20],
                                  value={"Mes": 6, "Semana": 8, "Día": 5}[per], key=f"tramo_n_{per}")
            tr = datos_tramos(per, n_per, tuple(sorted(sel_uni)), json.dumps(sello), VERSION_OPCIONES)
            k, et = tr["k"], tr["etiquetas"]
            unidad = {"Mes": "del mes", "Semana": "de la semana", "Día": ""}[per]
            if per == "Día":
                st.caption(f"Cada columna es **una rueda** (la primera, {et[0]}, es la más reciente; puede estar incompleta "
                           "si la sesión sigue abierta). Sirve para seguir día a día cómo va la semana.")
            else:
                st.caption(f"Van **{k} {'rueda' if k == 1 else 'ruedas'} {unidad}**: de cada {per.lower()} se toman solo sus primeras "
                           f"{k} ruedas, así se compara el mismo tramo (por ejemplo, del 1 al día de hoy de cada mes). "
                           f"La primera columna ({et[0]}) es el período en curso; su última rueda puede estar incompleta si la "
                           "sesión sigue abierta.")

            st.subheader("Ranking de los sectores en el mismo tramo")
            rk = tr["rank"]
            cambio = rk[et[1]] - rk[et[0]]
            tabla_rk = rk.copy()
            tabla_rk.insert(0, "Cambio", ["=" if c == 0 else (f"↑{c}" if c > 0 else f"↓{-c}") for c in cambio])
            tabla_rk = tabla_rk.sort_values(et[0]).rename_axis("Sector").reset_index()
            st.dataframe(
                tabla_rk.style.map(color_rank(len(rk)), subset=et)
                        .map(lambda v: "color: #3ecf8e; font-weight: 600" if str(v).startswith("↑")
                             else "color: #ef5a6f; font-weight: 600" if str(v).startswith("↓") else "", subset=["Cambio"]),
                hide_index=True, width="stretch",
                column_config={e: st.column_config.NumberColumn(e, format="%d°") for e in et})
            st.caption(f"Puesto de cada sector según su retorno (pesos iguales) {'en cada rueda' if per == 'Día' else f'en las primeras {k} ruedas de cada período'}: "
                       f"1° = el que más subió. **Cambio** = cuántos puestos subió (↑) o bajó (↓) frente al mismo tramo "
                       f"{ {'Mes': 'del mes', 'Semana': 'de la semana', 'Día': 'de la rueda'}[per] } anterior.")

            st.subheader("Volumen: participación y flujo en el mismo tramo")
            pa = tr["part"]
            delta = pa[et[0]] - pa[et[1]]
            signo = pa.iloc[:, ::-1].diff(axis=1).iloc[:, ::-1].drop(columns=et[-1])   # Δ de cada período vs el anterior
            racha = []
            for sec in pa.index:
                d0 = np.sign(signo.loc[sec].iloc[0]); r = 0
                for v in signo.loc[sec]:
                    if np.sign(v) == d0 and d0 != 0:
                        r += 1
                    else:
                        break
                racha.append(r)
            tabla_v = pd.DataFrame({"Sector": pa.index, "Flujo": np.where(delta > 0, "Entrando", "Saliendo"),
                                    "Δ participación (pp)": delta.values, "Racha": racha})
            tabla_v = pd.concat([tabla_v, pa.reset_index(drop=True)], axis=1).sort_values("Δ participación (pp)", ascending=False)
            st.dataframe(
                tabla_v.style.map(colorear_num, subset=["Δ participación (pp)"])
                       .map(lambda v: f"color: {COLOR_FLUJO.get(v, '')}; font-weight: 600", subset=["Flujo"]),
                hide_index=True, width="stretch",
                column_config={"Δ participación (pp)": st.column_config.NumberColumn(format="%+.2f"),
                               "Racha": st.column_config.NumberColumn(format=f"%d { {'Mes': 'meses', 'Semana': 'sem.', 'Día': 'días'}[per] }",
                                                                      help="Períodos seguidos con el mismo flujo, comparando siempre el mismo tramo."),
                               **{e: st.column_config.NumberColumn(e, format="%.1f%%") for e in et}})
            st.caption(f"**Participación** = parte del volumen en dólares del universo que se transó en el sector durante "
                       f"{'cada rueda' if per == 'Día' else f'las primeras {k} ruedas de cada período'}. **Δ participación** = {et[0]} menos {et[1]}{'' if per == 'Día' else ' (mismo tramo)'}. "
                       "**Flujo** = **Entrando** si el sector ganó participación, **Saliendo** si la perdió; la **Racha** dice "
                       "cuántos períodos seguidos lleva así.")

    with tab_mueve:
        if st.session_state.get("tabs_sectores") != pest_sect[2]:
            st.caption("Abre esta pestaña para calcular los factores.")
        else:
            ventana_m = st.radio("Ventana", ["6M", "1A", "2A", "5A"], index=1, horizontal=True, key="ventana_macro")
            desde = pd.Timestamp.now().normalize() - pd.DateOffset(months={"6M": 6, "1A": 12, "2A": 24, "5A": 60}[ventana_m])
            sd, _, _ = datos_sectores("Diaria", tuple(sorted(sel_uni)), json.dumps(sello), VERSION_OPCIONES)
            macro, err_m = cargar_macro(VERSION_OPCIONES)
            if err_m:
                st.caption(f"Algunas series macro no se pudieron cargar: {err_m}")

            od = sectores.ofensivo_defensivo(sd["ret"])
            m = st.columns(1 + len(macro))
            od_v = od[od.index >= desde]
            m[0].metric("Ofensivos / defensivos", f"{(od_v.iloc[-1] / od_v.iloc[0] - 1) * 100:+.1f}%",
                        f"{(od.iloc[-1] / od.iloc[-6] - 1) * 100:+.1f}% en 1 sem.",
                        help="Cuánto le han ganado los ofensivos a los defensivos en la ventana elegida.")
            for col, (nombre, serie) in zip(m[1:], macro.items()):
                cambio = (serie.iloc[-1] - serie.iloc[-6]) if nombre.startswith("Tasa") else \
                    (serie.iloc[-1] / serie.iloc[-6] - 1) * 100
                col.metric(nombre, f"{serie.iloc[-1]:,.2f}",
                           f"{cambio:+.2f} pp en 1 sem." if nombre.startswith("Tasa") else f"{cambio:+.1f}% en 1 sem.",
                           delta_color="off")
            st.plotly_chart(grafico_riesgo_macro(od, macro, desde), width="stretch")
            st.caption(
                f"**Ofensivos / defensivos** = {', '.join(sectores.OFENSIVOS)} contra "
                f"{', '.join(sectores.DEFENSIVOS)} (pesos iguales), base 100 al inicio de la ventana. Sube = el mercado "
                "busca riesgo; baja = busca refugio. Es la rotación más marcada en los datos: Tecnología y los "
                "defensivos se mueven casi siempre en sentidos opuestos. La línea punteada es su promedio de 50 días. "
                "Debajo, los motores macro: **tasa del bono a 10 años**, **petróleo WTI**, **VIX** (miedo esperado) y "
                "**DXY** (fuerza del dólar).")

            c_mom, c_sens = st.columns([2, 3])
            with c_mom:
                st.subheader("Momentum de los sectores")
                mom_t = sectores.momentum(sd["rs"])
                plazos = list(sectores.PLAZOS_MOMENTUM)
                st.dataframe(mom_t.style.map(colorear_num, subset=plazos), hide_index=True, width="stretch",
                             column_config={p: st.column_config.NumberColumn(format="%+.1f%%") for p in plazos})
                st.caption("Cuánto le ganó (o perdió) cada sector al mercado en cada plazo. A un mes hay algo de "
                           "persistencia: el sector que viene ganando tiende a seguir ganando un poco el mes siguiente. "
                           "A una semana, no.")
            with c_sens:
                st.subheader("Sensibilidad a la macro")
                if not macro:
                    st.caption("Sin series macro disponibles en este momento.")
                    st.stop()
                sens = sectores.sensibilidad_macro(sd["rel"][sd["rel"].index >= desde],
                                                   {k: v[v.index >= desde - pd.Timedelta(days=10)] for k, v in macro.items()})
                st.dataframe(sens.style.map(colorear_num).format("{:+.2f}"), width="stretch")
                st.caption("Correlación entre el retorno semanal del sector **contra el mercado** y el cambio semanal de "
                           "cada variable, en la ventana elegida (de −1 a +1). Ejemplo: positiva con la tasa = el sector "
                           "le gana al mercado las semanas en que suben las tasas. Con ventanas cortas hay pocas semanas: "
                           "léelo como una guía, no como algo exacto.")


        # ---------------------------------------------------------------------------
        # Vista empresa
        # ---------------------------------------------------------------------------


else:
    lista = filtrado["Ticker"].tolist()
    actual = st.session_state.get("ticker")
    if actual in set(data["Ticker"]) and actual not in lista:
        lista = [actual] + lista   # abierta desde una tabla que no usa los filtros (p. ej. Seguimiento o Mi cartera)
    if not lista:
        st.warning("No hay empresas con el filtro actual.")
        st.stop()
    if st.session_state.get("ticker") not in lista:
        st.session_state["ticker"] = "NVDA" if "NVDA" in lista else lista[0]
    ticker = st.sidebar.selectbox("Empresa", lista, key="ticker")
    fila = data[data["Ticker"] == ticker].iloc[0]
    ci = info.get(ticker, {})

    origen = ("Mi cartera" if st.session_state.get("_vista_origen") == "Mi cartera"
              else PESTANAS[min(st.session_state.get("_pestana_origen", 0), len(PESTANAS) - 1)])
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
                help="Histograma (MACD − señal) como % del precio. Cruce alcista y Pre-cruce solo cuentan con ambas líneas bajo cero.")
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
