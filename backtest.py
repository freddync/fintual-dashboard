"""
backtest.py
-----------
Backtest de reglas simples por empresa. Venta: cuando el RSI sube sobre un
umbral. Compra, a elección (ver ENTRADAS):
  rsi            RSI bajo el umbral de compra
  macd           golden cross del MACD "por abajo": la línea MACD cruza sobre su
                 señal con ambas líneas bajo cero
  macd_y_rsi     golden cross y RSI bajo el umbral el MISMO día
  macd_tras_rsi  golden cross con un RSI bajo el umbral en los últimos N días
                 (incluido el día del cruce)

Supuestos (para no mirar el futuro):
  - El RSI se calcula con el CIERRE del día t; la orden se ejecuta en la
    APERTURA del día t+1.
  - Una sola posición por empresa a la vez (las señales de compra repetidas
    mientras ya se está comprado se ignoran).
  - Las posiciones que siguen abiertas al final se valoran al último cierre y
    se marcan como "abiertas".
  - Costo por operación ida y vuelta, en % (comisiones + spread).

Para saber si la regla agrega valor, cada operación se compara con lo que la
MISMA acción rinde en promedio en cualquier ventana de igual cantidad de días
("base"). Exceso = retorno de la operación − base.
"""

import numpy as np
import pandas as pd

ENTRADAS = {
    "rsi": "RSI < umbral",
    "macd": "MACD golden cross por abajo",
    "macd_y_rsi": "MACD golden cross + RSI < umbral (mismo día)",
    "macd_tras_rsi": "MACD golden cross luego de un RSI < umbral",
}
MACD = (6, 13, 5)   # igual que en el dashboard


def rsi(close, period):
    """RSI de Wilder (inicializado con media simple), igual que en el dashboard."""
    delta = np.diff(close, prepend=np.nan)
    gan, per = np.clip(delta, 0, None), np.clip(-delta, 0, None)
    out = np.full(len(close), np.nan)
    if len(close) <= period:
        return out
    g, p = gan[1:period + 1].mean(), per[1:period + 1].mean()
    for i in range(period, len(close)):
        if i > period:
            g = (g * (period - 1) + gan[i]) / period
            p = (p * (period - 1) + per[i]) / period
        out[i] = 100.0 if p == 0 else 100 - 100 / (1 + g / p)
    return out


def golden_cross(cierre):
    """True el día en que la línea MACD cruza sobre su señal con ambas bajo cero."""
    rapida, lenta, sen = MACD
    c = pd.Series(cierre)
    linea = c.ewm(span=rapida, adjust=False).mean() - c.ewm(span=lenta, adjust=False).mean()
    senal = linea.ewm(span=sen, adjust=False).mean()
    hist = (linea - senal).to_numpy()
    prev = np.concatenate([[np.nan], hist[:-1]])
    return (prev < 0) & (hist >= 0) & (linea.to_numpy() < 0) & (senal.to_numpy() < 0)


def senal_compra(cierre, r, entrada, compra, ventana):
    """Arreglo booleano: en qué días (al cierre) hay señal de compra."""
    bajo = np.nan_to_num(r, nan=100) < compra
    if entrada == "rsi":
        return bajo
    golden = golden_cross(cierre)
    if entrada == "macd":
        return golden
    if entrada == "macd_y_rsi":
        return golden & bajo
    # macd_tras_rsi: hubo RSI bajo el umbral en los últimos `ventana` días
    reciente = pd.Series(bajo).rolling(ventana + 1, min_periods=1).max().to_numpy() > 0
    return golden & reciente


def operaciones(ticker, df, periodo=5, compra=30, venta=70, costo=0.1, entrada="rsi", ventana=10):
    """Lista de operaciones de una empresa."""
    fechas = df["Date"].to_numpy()
    aper = df["Open"].to_numpy(dtype=float)
    cierre = df["Close"].to_numpy(dtype=float)
    r = rsi(cierre, periodo)
    comprar = senal_compra(cierre, r, entrada, compra, ventana)
    n = len(df)
    ops, i_ent = [], None
    for t in range(n - 1):
        if np.isnan(r[t]):
            continue
        if i_ent is None and comprar[t]:
            i_ent = t + 1                                    # entra en la apertura siguiente
            senal = t
        elif i_ent is not None and r[t] > venta and t + 1 > i_ent:
            ops.append(_op(ticker, fechas, aper, senal, i_ent, t + 1, aper[t + 1], costo, False))
            i_ent = None
    if i_ent is not None and i_ent < n:                      # sigue abierta: al último cierre
        ops.append(_op(ticker, fechas, aper, senal, i_ent, n - 1, cierre[-1], costo, True))
    return ops


def _op(ticker, fechas, aper, i_senal, i_ent, i_sal, precio_sal, costo, abierta):
    return {
        "Ticker": ticker,
        "Señal": pd.Timestamp(fechas[i_senal]),
        "Entrada": pd.Timestamp(fechas[i_ent]),
        "Salida": pd.Timestamp(fechas[i_sal]),
        "Precio entrada": aper[i_ent],
        "Precio salida": precio_sal,
        "Días": int(i_sal - i_ent),
        "Retorno %": (precio_sal / aper[i_ent] - 1) * 100 - costo,
        "Abierta": abierta,
    }


def base_por_dias(df, dias):
    """Retorno promedio (%) de la acción en cualquier ventana de `dias` sesiones,
    de apertura a apertura: lo que se habría ganado sin ninguna señal."""
    a = df["Open"].to_numpy(dtype=float)
    out = {}
    for d in set(dias):
        if 0 < d < len(a):
            out[d] = float(np.nanmean(a[d:] / a[:-d] - 1) * 100)
    return out


def correr(precios, sector_de, periodo=5, compra=30, venta=70, costo=0.1, entrada="rsi", ventana=10):
    """Operaciones de todas las empresas, con su base y su sector."""
    filas = []
    for t, df in precios.items():
        ops = operaciones(t, df, periodo, compra, venta, costo, entrada, ventana)
        if not ops:
            continue
        base = base_por_dias(df, [o["Días"] for o in ops])
        for o in ops:
            o["Base %"] = base.get(o["Días"], np.nan)
            o["Exceso %"] = o["Retorno %"] - o["Base %"]
            o["Sector"] = sector_de.get(t, "Sin clasificar")
        filas += ops
    return pd.DataFrame(filas)


def agregar_sector(ops, s, ratio, mom):
    """Estado del sector de cada operación el día de la señal (rotación sectorial)."""
    from sectores import cuadrante

    def en(serie, f, sec):
        try:
            return serie.at[f, sec]
        except KeyError:
            return np.nan

    ops = ops.copy()
    ops["Cuadrante sector"] = [cuadrante(en(ratio, f, sec), en(mom, f, sec))
                               for f, sec in zip(ops["Señal"], ops["Sector"])]
    # flujo de volumen: participación del sector vs su promedio de 20 días
    ops["Flujo sector"] = ["Entrando" if v > 0 else "Saliendo" if v <= 0 else "Sin datos"
                           for v in (en(s["d_part"], f, sec) for f, sec in zip(ops["Señal"], ops["Sector"]))]
    # tendencia relativa del sector: le gana o pierde contra el mercado en el último mes
    ops["Sector vs mercado 1m"] = [_rel_1m(s["rs"], f, sec) for f, sec in zip(ops["Señal"], ops["Sector"])]
    return ops


def _rel_1m(rs, f, sec, n=21):
    try:
        i = rs.index.get_loc(f)
    except KeyError:
        return "Sin datos"
    if i < n or sec not in rs.columns:
        return "Sin datos"
    return "Le gana" if rs[sec].iloc[i] >= rs[sec].iloc[i - n] else "Le pierde"


def resumen(ops, por=None):
    """Estadísticas de las operaciones cerradas, en total o agrupadas por `por`."""
    c = ops[~ops["Abierta"]]

    def stats(g):
        return pd.Series({
            "Operaciones": len(g),
            "% ganadoras": (g["Retorno %"] > 0).mean() * 100,
            "Retorno prom. %": g["Retorno %"].mean(),
            "Retorno mediano %": g["Retorno %"].median(),
            "Base prom. %": g["Base %"].mean(),
            "Exceso prom. %": g["Exceso %"].mean(),
            "% le gana a la base": (g["Exceso %"] > 0).mean() * 100,
            "Días prom.": g["Días"].mean(),
            "Peor %": g["Retorno %"].min(),
            "Factor de ganancia": (g.loc[g["Retorno %"] > 0, "Retorno %"].sum()
                                   / max(-g.loc[g["Retorno %"] < 0, "Retorno %"].sum(), 1e-9)),
        })

    if por is None:
        return stats(c)
    return c.groupby(por).apply(stats, include_groups=False).sort_values("Exceso prom. %", ascending=False)


def acotar(ops, limite=30):
    """Acota retorno y exceso de cada operación a ±limite %, para que un caso extremo
    (ej. una acción que sube 1.000%) no domine los promedios."""
    ops = ops.copy()
    for c in ["Retorno %", "Exceso %"]:
        ops[c] = ops[c].clip(-limite, limite)
    return ops


def comparar_umbrales(por_umbral, por, min_ops=30):
    """Exceso promedio y n° de operaciones por grupo (`por`) para cada umbral de compra,
    y el umbral con mayor exceso entre los que tienen al menos `min_ops` operaciones."""
    exceso, n = {}, {}
    for u, ops in por_umbral.items():
        r = resumen(ops, por)
        exceso[u], n[u] = r["Exceso prom. %"], r["Operaciones"]
    exceso, n = pd.DataFrame(exceso), pd.DataFrame(n).fillna(0).astype(int)
    validos = exceso.where(n >= min_ops)
    mejor = pd.Series([fila.idxmax() if fila.notna().any() else None for _, fila in validos.iterrows()],
                      index=validos.index, dtype=object)
    out = pd.concat({"Exceso %": exceso, "Operaciones": n}, axis=1)
    out.columns = [f"{a} (RSI<{u})" for a, u in out.columns]
    out["Mejor umbral"] = mejor.map(lambda u: f"RSI < {u:g}" if pd.notna(u) else "pocas operaciones")
    return out
