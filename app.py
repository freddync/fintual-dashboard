"""
Dashboard Fintual — Megacap + Large Cap.

Corre local con:  streamlit run app.py   (o doble click en Dashboard.bat)
"""

import datetime
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
PRECIOS_DIR = os.path.join(DATA_DIR, "precios")
FUND_DIR = os.path.join(DATA_DIR, "fundamentales")
SEGUIMIENTO = os.path.join(DATA_DIR, "seguimiento.json")

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
# Sin secrets (uso local) se guarda en data/seguimiento.json.

GIST_FILE = "seguimiento.json"


def config_gist():
    try:
        s = st.secrets["seguimiento"]
        return s["gist_id"], {"Authorization": f"Bearer {s['token']}",
                              "Accept": "application/vnd.github+json"}
    except Exception:
        return None, None


@st.cache_data(ttl=60, show_spinner=False)
def leer_seguimiento():
    """{ticker: {"fecha": "YYYY-MM-DD", "precio": float}}"""
    gist_id, headers = config_gist()
    try:
        if gist_id:
            r = requests.get(f"https://api.github.com/gists/{gist_id}", headers=headers, timeout=15)
            r.raise_for_status()
            return json.loads(r.json()["files"][GIST_FILE]["content"] or "{}")
        with open(SEGUIMIENTO, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        st.error(f"No se pudo leer el seguimiento: {e}")
        return {}


def guardar_seguimiento(seg):
    texto = json.dumps(seg, ensure_ascii=False, indent=1)
    gist_id, headers = config_gist()
    if gist_id:
        r = requests.patch(f"https://api.github.com/gists/{gist_id}", headers=headers, timeout=15,
                           json={"files": {GIST_FILE: {"content": texto}}})
        r.raise_for_status()
    else:
        with open(SEGUIMIENTO, "w", encoding="utf-8") as f:
            f.write(texto)
    leer_seguimiento.clear()


def agregar_seguimiento(ticker, precio):
    seg = dict(leer_seguimiento())
    seg[ticker] ={"fecha": datetime.date.today().isoformat(), "precio": round(float(precio), 2)}
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


def mostrar_tabla(tabla, key, height=650):
    """Tabla ordenable; al marcar la casilla de una fila se abre esa empresa."""
    tabla = tabla.reset_index(drop=True)
    sel = st.dataframe(tabla, hide_index=True, width="stretch", height=height,
                       on_select="rerun", selection_mode="single-row", key=key,
                       column_config=COLUMNAS)
    if sel.selection.rows:
        st.session_state["_abrir"] = tabla.iloc[sel.selection.rows[0]]["Ticker"]
        del st.session_state[key]   # limpia la selección para la vuelta
        st.rerun()


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

st.sidebar.title("📈 Fintual")
vista = st.sidebar.radio("Vista", ["General", "Empresa"], horizontal=True, key="vista")
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
    tab_todas, tab_rsi, tab_seg = st.tabs(
        ["Todas", f"{COL_RSI} < {RSI_BAJO} ({len(sobreventa)})", "⭐ Seguimiento"])

    with tab_todas:
        mostrar_tabla(filtrado, "tabla_general")
        st.caption("P/E con EPS diluido de los últimos 4 trimestres. EV/EBITDA con el último año fiscal. "
                   "Crecimiento y margen del último año fiscal.")

    with tab_rsi:
        if sobreventa.empty:
            st.info(f"Ninguna empresa del filtro actual tiene {COL_RSI} bajo {RSI_BAJO}.")
        else:
            mostrar_tabla(sobreventa.sort_values(COL_RSI), "tabla_rsi")

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
            mostrar_tabla(t, "tabla_seg", height=min(650, 38 + 35 * len(t)))
        st.caption("El seguimiento se guarda en data/seguimiento.json: se mantiene al cerrar y volver a "
                   "abrir el dashboard, y no depende de los filtros del panel izquierdo.")


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
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])], rangeslider_visible=False)
    fig.update_layout(height=800, template="plotly_dark", showlegend=False,
                      margin=dict(t=10, b=10), bargap=0.1)
    st.plotly_chart(fig, width="stretch")

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
