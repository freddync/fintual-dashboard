"""
opciones.py
-----------
Call/put walls (los strikes con mayor Open Interest) de los vencimientos más
próximos, put/call ratio, y velas de 1 hora de la última semana.

La cadena de opciones viene de Cboe (cotizaciones diferidas, una sola descarga con
todos los vencimientos). Yahoo queda de respaldo: de noche y de madrugada deja el
Open Interest en cero en casi todos los strikes, y los muros salían vacíos o mal.
"""

import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

N_VENCIMIENTOS = 2   # los 2 vencimientos más próximos
N_MUROS = 2          # muros por lado
VENTANA = 0.25       # strikes a ±25% del precio
DIAS_PUT_CALL = 60   # put/call "total": vencimientos a 60 días o menos (ahí está casi todo el volumen)
MIN_OI_VALIDO = 0.3  # si menos del 30% de los strikes con volumen trae OI, el OI de la fuente viene vacío
CBOE = "https://cdn.cboe.com/api/global/delayed_quotes/options/{}.json"

HEADERS = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")}


def _hoy():
    return datetime.datetime.now(ZoneInfo("America/New_York")).date()   # el servidor corre en UTC


def cadena_cboe(ticker):
    """Todos los contratos del ticker: DataFrame exp (date), tipo (C/P), strike, oi, vol."""
    r = requests.get(CBOE.format(ticker.replace("-", ".")), headers=HEADERS, timeout=20)
    r.raise_for_status()
    o = pd.DataFrame(r.json()["data"]["options"])
    m = o["option"].str.extract(r"^(?P<raiz>.+?)(?P<exp>\d{6})(?P<tipo>[CP])(?P<k>\d{8})$")
    # solo la raíz estándar: las ajustadas (NVDA1, tras un split o fusión) no entregan 100 acciones
    m = m[m["raiz"] == m["raiz"].value_counts().idxmax()]
    o = o.loc[m.index]
    return pd.DataFrame({"exp": pd.to_datetime(m["exp"], format="%y%m%d").dt.date, "tipo": m["tipo"],
                         "strike": m["k"].astype(float) / 1000,
                         "oi": pd.to_numeric(o["open_interest"], errors="coerce").fillna(0),
                         "vol": pd.to_numeric(o["volume"], errors="coerce").fillna(0)})


def cadena_yahoo(ticker):
    """Respaldo: lo mismo desde Yahoo, solo los vencimientos a DIAS_PUT_CALL días o menos."""
    import yfinance as yf

    t, hoy, partes = yf.Ticker(ticker), _hoy(), []
    for exp in t.options or []:
        dte = (datetime.date.fromisoformat(exp) - hoy).days
        if dte < 0:
            continue
        if dte > DIAS_PUT_CALL and partes:
            break
        c = t.option_chain(exp)
        for tipo, df in (("C", c.calls), ("P", c.puts)):
            partes.append(pd.DataFrame({"exp": datetime.date.fromisoformat(exp), "tipo": tipo, "strike": df["strike"],
                                        "oi": pd.to_numeric(df["openInterest"], errors="coerce").fillna(0),
                                        "vol": pd.to_numeric(df["volume"], errors="coerce").fillna(0)}))
    return pd.concat(partes, ignore_index=True) if partes else pd.DataFrame(columns=["exp", "tipo", "strike", "oi", "vol"])


def cadena(ticker):
    """(contratos, fuente). Cboe primero; Yahoo si Cboe falla o no tiene el ticker."""
    try:
        df = cadena_cboe(ticker)
        if not df.empty:
            return df, "Cboe"
    except Exception:
        pass
    return cadena_yahoo(ticker), "Yahoo"


def oi_valido(df):
    """¿La fuente trae Open Interest? (Yahoo lo deja en cero de madrugada)."""
    d = df[df["vol"] > 0]
    return d.empty or (d["oi"] > 0).mean() >= MIN_OI_VALIDO


def _top(df, precio, col):
    """Los N_MUROS strikes con mayor `col` dentro de la ventana, como [(strike, valor), ...]."""
    d = df[(df["strike"] >= precio * (1 - VENTANA)) & (df["strike"] <= precio * (1 + VENTANA)) & (df[col] > 0)]
    top = d.sort_values(col, ascending=False).head(N_MUROS)
    return [(float(k), int(v)) for k, v in zip(top["strike"], top[col])]


def conteos(df):
    """Contratos de calls y puts (OI y volumen) y sus put/call ratios."""
    c, p = df[df["tipo"] == "C"], df[df["tipo"] == "P"]
    out = {"oi_call": int(c["oi"].sum()), "oi_put": int(p["oi"].sum()),
           "vol_call": int(c["vol"].sum()), "vol_put": int(p["vol"].sum())}
    out["pc_oi"] = out["oi_put"] / out["oi_call"] if out["oi_call"] else None
    out["pc_vol"] = out["vol_put"] / out["vol_call"] if out["vol_call"] else None
    return out


def muros(df, precio, base="oi"):
    """[{exp, dte, calls: [(strike, oi)], puts: [(strike, oi)], oi_call, ..., pc_vol}, ...]
    de los N_VENCIMIENTOS más próximos. `base` = "vol" arma los muros con el volumen."""
    hoy, out = _hoy(), []
    for exp, g in sorted(df[df["exp"] >= hoy].groupby("exp"), key=lambda x: x[0]):
        calls, puts = _top(g[g["tipo"] == "C"], precio, base), _top(g[g["tipo"] == "P"], precio, base)
        if calls or puts:
            out.append({"exp": exp.isoformat(), "dte": (exp - hoy).days, "calls": calls, "puts": puts, **conteos(g)})
        if len(out) == N_VENCIMIENTOS:
            break
    return out


def put_call(df, dias=DIAS_PUT_CALL):
    """Put/call de todos los vencimientos a `dias` días o menos."""
    hoy = _hoy()
    return conteos(df[(df["exp"] >= hoy) & (df["exp"] <= hoy + datetime.timedelta(days=dias))])


def put_call_plazos(df):
    """Put/call OI del próximo vencimiento (y sus DTE), de los vencimientos a 30 días o menos y a 60 o menos."""
    hoy = _hoy()
    vivos = df[df["exp"] >= hoy]
    if vivos.empty:
        return None
    prox = vivos["exp"].min()
    p, t30, t60 = conteos(vivos[vivos["exp"] == prox]), put_call(df, 30), put_call(df, 60)
    return {"DTE próx.": (prox - hoy).days, "P/C OI próx.": p["pc_oi"], "P/C OI 30d": t30["pc_oi"],
            "P/C OI 60d": t60["pc_oi"], "OI put 60d": t60["oi_put"], "OI call 60d": t60["oi_call"],
            "P/C volumen 60d": t60["pc_vol"]}


def noticias(ticker, n=10):
    """Últimas noticias del ticker según la búsqueda de Yahoo Finance."""
    r = requests.get("https://query2.finance.yahoo.com/v1/finance/search",
                     params={"q": ticker, "newsCount": n, "quotesCount": 0},
                     headers=HEADERS, timeout=15)
    r.raise_for_status()
    out = []
    for x in r.json().get("news", []):
        out.append({
            "titulo": x.get("title", ""),
            "fuente": x.get("publisher", ""),
            "link": x.get("link", ""),
            "fecha": pd.to_datetime(x.get("providerPublishTime"), unit="s", utc=True)
                       .tz_convert("America/New_York") if x.get("providerPublishTime") else None,
            "otros": [t for t in x.get("relatedTickers") or [] if t != ticker],
        })
    return out


MACRO = {
    "Tasa 10 años (%)": "^TNX",
    "Petróleo WTI (USD)": "CL=F",
    "VIX": "^VIX",
    "Dólar DXY": "DX-Y.NYB",
}


def benchmark_desde(simbolo, fecha):
    """Compra en la APERTURA de `fecha` (o del primer día hábil siguiente) y su valor hoy,
    con dividendos reinvertidos. Devuelve (fecha de compra, precio de apertura, factor de
    crecimiento total desde esa apertura, último precio)."""
    inicio = int(pd.Timestamp(fecha).timestamp()) - 86400 * 7
    r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{simbolo}",
                     params={"period1": inicio, "period2": int(pd.Timestamp.now().timestamp()) + 86400,
                             "interval": "1d", "events": "div,splits"}, headers=HEADERS, timeout=20)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame({"open": q["open"], "close": q["close"],
                       "adj": res["indicators"]["adjclose"][0]["adjclose"]},
                      index=pd.to_datetime(res["timestamp"], unit="s").normalize()).dropna()
    df = df[df.index >= pd.Timestamp(fecha).normalize()]
    if df.empty:
        raise RuntimeError(f"Sin datos de {simbolo} desde {fecha}")
    d0 = df.iloc[0]
    apertura_ajustada = d0["open"] * d0["adj"] / d0["close"]     # apertura en la misma escala que el ajustado
    return df.index[0], float(d0["open"]), float(df["adj"].iloc[-1] / apertura_ajustada), float(df["close"].iloc[-1])


def serie_diaria(simbolo, rango="5y"):
    """Cierres diarios de un índice o futuro de Yahoo, indexados por fecha."""
    r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{simbolo}",
                     params={"range": rango, "interval": "1d"}, headers=HEADERS, timeout=20)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    idx = pd.to_datetime(res["timestamp"], unit="s").normalize()
    s = pd.Series(res["indicators"]["quote"][0]["close"], index=idx, dtype=float).dropna()
    return s[~s.index.duplicated(keep="last")]


def velas_1h(ticker):
    """Velas de 1 hora de los últimos 5 días hábiles, en hora de Nueva York."""
    r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
                     params={"range": "5d", "interval": "60m"}, headers=HEADERS, timeout=20)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    idx = (pd.to_datetime(res["timestamp"], unit="s", utc=True)
           .tz_convert("America/New_York").tz_localize(None))
    df = pd.DataFrame({"Open": q["open"], "High": q["high"], "Low": q["low"],
                       "Close": q["close"], "Volume": q["volume"]}, index=idx).dropna(subset=["Close"])
    df["Volume"] = df["Volume"].fillna(0)
    # Yahoo agrega un "tick" de cierre a las 16:00 que no es una vela
    return df[~((df.index.hour == 16) & (df.index.minute == 0))]


# ---------------------------------------------------------------------------
# Perfil de volumen y comportamiento del precio en cada muro (última semana)
# ---------------------------------------------------------------------------

ZONA = 0.015   # zona de contacto de un muro: ±1,5% del strike


def perfil_volumen(velas, lo, hi, n=60):
    """Acciones transadas por nivel de precio: el volumen de cada vela se reparte
    en partes iguales entre su mínimo y su máximo. Devuelve (centros, volumen, alto)."""
    bordes = np.linspace(lo, hi, n + 1)
    vol = np.zeros(n)
    for l, h, v in zip(velas["Low"], velas["High"], velas["Volume"]):
        if not v:
            continue
        if h <= l:
            vol[np.clip(np.searchsorted(bordes, l) - 1, 0, n - 1)] += v
            continue
        cruce = np.clip(np.minimum(bordes[1:], h) - np.maximum(bordes[:-1], l), 0, None)
        vol += v * cruce / (h - l)
    return (bordes[:-1] + bordes[1:]) / 2, vol, bordes[1] - bordes[0]


def volumen_en_zona(velas, strike):
    """% del volumen de la semana transado dentro de la zona del muro."""
    _, vol, _ = perfil_volumen(velas, strike * (1 - ZONA), strike * (1 + ZONA), n=1)
    total = velas["Volume"].sum()
    return vol[0] / total * 100 if total else None


def estado_muro(velas, strike):
    """¿Qué hizo el precio con el muro durante la semana?

    Rompió ↑ / ↓  empezó la semana de un lado del strike y ahora cierra del otro
    Probando      está dentro de la zona del muro (±1,5%)
    Rebotó        llegó a la zona (o la perforó con mecha) y se alejó sin cruzar
    Sin tocar     nunca llegó a la zona
    """
    primero, ultimo = velas["Close"].iloc[0], velas["Close"].iloc[-1]
    if (primero >= strike) != (ultimo >= strike):
        return "Rompió ↑" if ultimo >= strike else "Rompió ↓"
    if abs(ultimo / strike - 1) <= ZONA:
        return "Probando"
    # punto más cercano al que llegó, desde su lado del muro
    if ultimo > strike:
        toco = velas["Low"].min() <= strike * (1 + ZONA)
    else:
        toco = velas["High"].max() >= strike * (1 - ZONA)
    return "Rebotó" if toco else "Sin tocar"
