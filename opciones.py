"""
opciones.py
-----------
Call/put walls (los strikes con mayor Open Interest) de los vencimientos más
próximos, y velas de 1 hora de la última semana. Se descarga en vivo desde
Yahoo (yfinance resuelve la cookie/crumb que exigen las opciones).
"""

import datetime
from zoneinfo import ZoneInfo

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
                       "Close": q["close"]}, index=idx).dropna()
    # Yahoo agrega un "tick" de cierre a las 16:00 que no es una vela
    return df[~((df.index.hour == 16) & (df.index.minute == 0))]
