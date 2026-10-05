"""
opciones.py
-----------
Call/put walls (los strikes con mayor Open Interest) de los vencimientos más
próximos, y velas de 1 hora de la última semana. Se descarga en vivo desde
Yahoo (yfinance resuelve la cookie/crumb que exigen las opciones).
"""

import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

N_VENCIMIENTOS = 2   # los 2 vencimientos más próximos
N_MUROS = 2          # muros por lado
VENTANA = 0.25       # strikes a ±25% del precio

HEADERS = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")}


def _top_oi(df, precio):
    """Los N_MUROS strikes con mayor OI dentro de la ventana, como [(strike, oi), ...]."""
    if df is None or df.empty:
        return []
    d = df[["strike", "openInterest"]].apply(pd.to_numeric, errors="coerce").fillna(0)
    d = d[(d["strike"] >= precio * (1 - VENTANA)) & (d["strike"] <= precio * (1 + VENTANA))
          & (d["openInterest"] > 0)]
    top = d.sort_values("openInterest", ascending=False).head(N_MUROS)
    return [(float(k), int(oi)) for k, oi in zip(top["strike"], top["openInterest"])]


def muros(ticker, precio):
    """[{exp, dte, calls: [(strike, oi)], puts: [(strike, oi)]}, ...]"""
    import yfinance as yf

    t = yf.Ticker(ticker)
    hoy = datetime.datetime.now(ZoneInfo("America/New_York")).date()   # el servidor corre en UTC
    out = []
    for exp in t.options or []:
        dte = (datetime.date.fromisoformat(exp) - hoy).days
        if dte < 0:
            continue
        cadena = t.option_chain(exp)
        calls, puts = _top_oi(cadena.calls, precio), _top_oi(cadena.puts, precio)
        if calls or puts:
            out.append({"exp": exp, "dte": dte, "calls": calls, "puts": puts})
        if len(out) == N_VENCIMIENTOS:
            break
    return out


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
