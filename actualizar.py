"""
actualizar.py
-------------
Descarga desde Yahoo los datos de TODAS las empresas (Megacap + Large Cap)
y los deja en una sola carpeta: data/.

    python actualizar.py precios          (~10-15 min)
    python actualizar.py fundamentales    (~20-30 min, cada trimestre basta)
    python actualizar.py todo

La lista de empresas vive en data/company_info.json.
Si un ticker falla, se conserva su archivo anterior.
"""

import datetime
import json
import os
import sys
import time
from zoneinfo import ZoneInfo

import pandas as pd
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PRECIOS_DIR = os.path.join(DATA_DIR, "precios")
FUND_DIR = os.path.join(DATA_DIR, "fundamentales")
COMPANY_INFO = os.path.join(DATA_DIR, "company_info.json")
SELLO = os.path.join(DATA_DIR, "_ultima_actualizacion.json")

N_ROWS_KEEP = 1250   # ~5 años de sesiones diarias
PAUSA = 0.4          # segundos entre peticiones, para no gatillar el rate-limit de Yahoo

HEADERS = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")}

FUND_METRICS = [
    "TotalRevenue", "CostOfRevenue", "GrossProfit", "OperatingIncome", "NetIncome",
    "BasicEPS", "DilutedEPS", "EBITDA", "TotalAssets", "TotalLiabilitiesNetMinorityInterest",
    "TotalEquityGrossMinorityInterest", "TotalDebt", "CashAndCashEquivalents",
    "FreeCashFlow", "OperatingCashFlow", "CapitalExpenditure", "DilutedAverageShares",
]


def get_json(url, params, reintentos=3):
    for intento in range(reintentos):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 503):
                time.sleep(5 + intento * 10)
                continue
        except Exception:
            pass
        time.sleep(2 + intento * 3)
    return None


def tickers():
    with open(COMPANY_INFO, encoding="utf-8") as f:
        return sorted(json.load(f).keys())


# ---------------------------------------------------------------------------
# Precios
# ---------------------------------------------------------------------------

def bajar_precios(ticker):
    d = get_json(f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
                 {"range": "10y", "interval": "1d"})
    try:
        res = d["chart"]["result"][0]
        q = res["indicators"]["quote"][0]
        df = pd.DataFrame({
            "Date": pd.to_datetime(res["timestamp"], unit="s").strftime("%Y-%m-%d"),
            "Open": q["open"], "High": q["high"], "Low": q["low"],
            "Close": q["close"], "Volume": q["volume"],
        }).dropna()
    except Exception:
        return None
    if df.empty:
        return None
    for c in ["Open", "High", "Low", "Close"]:
        df[c] = df[c].round(2)
    df["Volume"] = df["Volume"].astype("int64")
    return df.drop_duplicates("Date", keep="last").tail(N_ROWS_KEEP)


def actualizar_precios():
    os.makedirs(PRECIOS_DIR, exist_ok=True)
    lista = tickers()
    print(f"=== Precios: {len(lista)} empresas ===", flush=True)
    ok, fallidos, ultima = 0, [], None
    for i, t in enumerate(lista, 1):
        df = bajar_precios(t)
        if df is None:
            fallidos.append(t)
        else:
            df.to_csv(os.path.join(PRECIOS_DIR, f"{t}.csv"), index=False, lineterminator="\n")
            ok += 1
            ultima = max(ultima or "", df["Date"].iloc[-1])
        if i % 50 == 0 or i == len(lista):
            print(f"  [{i}/{len(lista)}] ok={ok} fallidos={len(fallidos)}", flush=True)
        time.sleep(PAUSA)

    if fallidos:
        print("Fallidos (se conserva su archivo anterior):", ", ".join(fallidos))
    if len(fallidos) > 0.2 * len(lista):
        # sin sello: el dashboard sigue mostrando la fecha de la última corrida buena
        sys.exit(f"ERROR: falló el {len(fallidos) / len(lista):.0%} de los tickers "
                 "(probablemente Yahoo está limitando las peticiones).")
    ahora = datetime.datetime.now(ZoneInfo("America/New_York"))
    actualizar_sello(fecha=ahora.strftime("%Y-%m-%d %H:%M NY"), ultima_sesion=ultima,
                     ok=ok, fallidos=len(fallidos))


def actualizar_sello(**campos):
    """Sello que lee el dashboard; cuando cambia, la app recarga sus datos."""
    try:
        with open(SELLO, encoding="utf-8") as f:
            sello = json.load(f)
    except Exception:
        sello = {}
    sello.update(campos)
    with open(SELLO, "w", encoding="utf-8") as f:
        json.dump(sello, f, indent=1)


# ---------------------------------------------------------------------------
# Fundamentales (anual + trimestral, con variaciones %)
# ---------------------------------------------------------------------------

def bajar_fundamentales(ticker, prefijo):
    d = get_json(
        f"https://query2.finance.yahoo.com/ws/fundamentals-timeseries/v1/finance/timeseries/{ticker}",
        {"symbol": ticker, "type": ",".join(prefijo + m for m in FUND_METRICS),
         "period1": 0, "period2": int(time.time())})
    try:
        resultados = d["timeseries"]["result"] or []
    except Exception:
        return None
    por_fecha = {}
    for res in resultados:
        tipo = (res.get("meta", {}).get("type") or [""])[0]
        metrica = tipo[len(prefijo):]
        if metrica not in FUND_METRICS:
            continue
        for e in res.get(tipo) or []:
            if e:
                por_fecha.setdefault(e["asOfDate"], {})[metrica] = e.get("reportedValue", {}).get("raw")
    if not por_fecha:
        return None
    df = pd.DataFrame.from_dict(por_fecha, orient="index").reindex(columns=FUND_METRICS)
    df.index.name = "Date"
    return df.sort_index().reset_index()


def agregar_variaciones(df, anual):
    """Anual: *_var_pct vs año anterior. Trimestral: *_QoQ_pct y *_YoY_pct."""
    out = df.copy()
    for m in FUND_METRICS:
        s = df[m].astype(float)
        if anual:
            out[f"{m}_var_pct"] = (s - s.shift(1)) / s.shift(1).abs() * 100
        else:
            out[f"{m}_QoQ_pct"] = (s - s.shift(1)) / s.shift(1).abs() * 100
            out[f"{m}_YoY_pct"] = (s - s.shift(4)) / s.shift(4).abs() * 100
    return out.replace([float("inf"), float("-inf")], None)


def actualizar_fundamentales():
    os.makedirs(FUND_DIR, exist_ok=True)
    lista = tickers()
    print(f"=== Fundamentales: {len(lista)} empresas ===", flush=True)
    fallidos = []
    for i, t in enumerate(lista, 1):
        for prefijo, nombre, anual in [("annual", "anual", True), ("quarterly", "trimestral", False)]:
            df = bajar_fundamentales(t, prefijo)
            if df is None:
                fallidos.append(f"{t}/{nombre}")
            else:
                agregar_variaciones(df, anual).to_csv(
                    os.path.join(FUND_DIR, f"{t}_{nombre}.csv"), index=False, lineterminator="\n")
            time.sleep(PAUSA)
        if i % 50 == 0 or i == len(lista):
            print(f"  [{i}/{len(lista)}] fallidos={len(fallidos)}", flush=True)
    if fallidos:
        print("Fallidos (se conserva su archivo anterior):", ", ".join(fallidos))
    actualizar_sello(fundamentales=datetime.date.today().isoformat())


if __name__ == "__main__":
    que = sys.argv[1] if len(sys.argv) > 1 else "precios"
    t0 = time.time()
    if que in ("precios", "todo"):
        actualizar_precios()
    if que in ("fundamentales", "todo"):
        actualizar_fundamentales()
    print(f"\nListo en {(time.time() - t0) / 60:.1f} min.")
