"""
sectores.py
-----------
Rotación sectorial: cómo se mueven los sectores entre sí, en precio y en volumen.

Todo se calcula con pesos iguales (cada empresa pesa lo mismo dentro de su
sector), comparando contra el promedio de todo el universo ("mercado"):

  Retorno %          retorno promedio de las empresas del sector en el período
  vs mercado (pp)    retorno del sector menos el del mercado
  % al alza          amplitud: qué parte de las empresas del sector subió
  Vol. relativo      volumen en dólares del período / su promedio de los 20 anteriores
                     (12 en mensual)
  Part. volumen %    qué parte del volumen en dólares del mercado se transó en el sector
  Δ participación    participación actual menos su promedio de los 20 períodos anteriores
                     (pp): positivo = está entrando más dinero que lo habitual
  RS-Ratio / RS-Mom  gráfico de rotación relativa (RRG): fuerza relativa del sector
                     contra el mercado y su aceleración. Cuadrantes:
                       Liderando     (fuerte y acelerando)
                       Debilitándose (fuerte pero frenando)
                       Rezagado      (débil y frenando)
                       Mejorando     (débil pero acelerando)
                     El ciclo típico de una rotación recorre los cuadrantes en el
                     sentido de las agujas del reloj: Mejorando → Liderando →
                     Debilitándose → Rezagado.
"""

import pandas as pd

# períodos para el promedio de volumen (en mensual, un año)
VENTANA_VOL = {"Diaria": 20, "Semanal": 20, "Mensual": 12}
# (ventana del RS-Ratio, rezago del RS-Momentum)
RRG = {"Diaria": (20, 5), "Semanal": (10, 4), "Mensual": (6, 2)}
MOMENTUM = {"Diaria": {"1m": 21, "3m": 63}, "Semanal": {"1m": 4, "3m": 13}, "Mensual": {"1m": 1, "3m": 3}}
REMUESTREO = {"Semanal": "W-FRI", "Mensual": "ME"}   # la semana cierra el viernes; el mes, su último día


def paneles(precios, frecuencia):
    """(cierres, volumen en dólares) con una columna por ticker."""
    cierre = pd.DataFrame({t: df.set_index("Date")["Close"] for t, df in precios.items()}).sort_index()
    dolares = pd.DataFrame({t: df.set_index("Date")["Close"] * df.set_index("Date")["Volume"]
                            for t, df in precios.items()}).sort_index()
    if frecuencia in REMUESTREO:
        cierre = cierre.resample(REMUESTREO[frecuencia]).last()
        dolares = dolares.resample(REMUESTREO[frecuencia]).sum(min_count=1)
    return cierre, dolares


MIN_EMPRESAS = 5   # sectores más chicos se excluyen (con 1-2 empresas solo meten ruido)


def series(cierre, dolares, sector_de, frecuencia):
    """Series por sector (filas = fechas, columnas = sectores)."""
    sector_de = pd.Series(sector_de).reindex(cierre.columns)
    conteo = sector_de.value_counts()
    sector_de = sector_de[sector_de.isin(conteo[conteo >= MIN_EMPRESAS].index)]
    cierre, dolares = cierre[sector_de.index], dolares[sector_de.index]
    # retornos acotados: un split o un dato malo no debe mover todo un sector
    ret = cierre.pct_change(fill_method=None).clip(-0.5, 0.5)
    ret_sec = ret.T.groupby(sector_de).mean().T
    alza = (ret > 0).astype(float).where(ret.notna()).T.groupby(sector_de).mean().T * 100
    dv_sec = dolares.T.groupby(sector_de).sum(min_count=1).T
    ret_mkt = ret.mean(axis=1)

    part = dv_sec.div(dv_sec.sum(axis=1), axis=0) * 100
    v = VENTANA_VOL[frecuencia]
    prom_part = part.shift(1).rolling(v, min_periods=5).mean()
    prom_dv = dv_sec.shift(1).rolling(v, min_periods=5).mean()

    idx_sec = (1 + ret_sec.fillna(0)).cumprod()
    idx_mkt = (1 + ret_mkt.fillna(0)).cumprod()
    rs = idx_sec.div(idx_mkt, axis=0)
    return {
        "ret": ret_sec * 100,
        "rel": ret_sec.sub(ret_mkt, axis=0) * 100,
        "alza": alza,
        "vol_rel": dv_sec / prom_dv,
        "part": part,
        "d_part": part - prom_part,
        "rs": rs,
        "n": sector_de.value_counts(),
    }


def rrg(rs, frecuencia):
    """RS-Ratio y RS-Momentum (aproximación del gráfico de rotación relativa)."""
    ventana, rezago = RRG[frecuencia]
    ratio = 100 * rs / rs.rolling(ventana).mean()
    mom = 100 * ratio / ratio.shift(rezago)
    return ratio, mom


def cuadrante(ratio, mom):
    if pd.isna(ratio) or pd.isna(mom):
        return "Sin datos"
    if ratio >= 100:
        return "Liderando" if mom >= 100 else "Debilitándose"
    return "Mejorando" if mom >= 100 else "Rezagado"


def flujo(d_part):
    """Entrando / Saliendo según el signo de la Δ participación del último período, y
    cuántos períodos seguidos lleva así (un solo período puede ser ruido)."""
    v = d_part.dropna()
    if v.empty:
        return {"Flujo": "Sin datos", "Racha": None}
    signo = v.iloc[-1] > 0
    racha = 0
    for x in reversed(v.to_numpy()):
        if (x > 0) != signo:
            break
        racha += 1
    return {"Flujo": "Entrando" if signo else "Saliendo", "Racha": racha}


def tabla(s, ratio, mom, fecha, frecuencia):
    """Resumen de todos los sectores en una fecha."""
    m = MOMENTUM[frecuencia]
    i = s["rs"].index.get_loc(fecha)
    filas = []
    for sec in s["ret"].columns:
        rs = s["rs"][sec]
        filas.append({
            "Sector": sec,
            "Empresas": int(s["n"].get(sec, 0)),
            "Retorno %": s["ret"].at[fecha, sec],
            "vs mercado (pp)": s["rel"].at[fecha, sec],
            "% al alza": s["alza"].at[fecha, sec],
            "Vol. relativo": s["vol_rel"].at[fecha, sec],
            "Part. volumen %": s["part"].at[fecha, sec],
            "Δ participación (pp)": s["d_part"].at[fecha, sec],
            "Cuadrante": cuadrante(ratio.at[fecha, sec], mom.at[fecha, sec]),
            **flujo(s["d_part"][sec].iloc[:i + 1]),
            "Relativo 1m %": (rs.iloc[i] / rs.iloc[i - m["1m"]] - 1) * 100 if i >= m["1m"] else None,
            "Relativo 3m %": (rs.iloc[i] / rs.iloc[i - m["3m"]] - 1) * 100 if i >= m["3m"] else None,
        })
    return pd.DataFrame(filas).sort_values("vs mercado (pp)", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Riesgo vs refugio, momentum y factores macro
# ---------------------------------------------------------------------------

OFENSIVOS = ["Tecnología", "Industrial", "Materiales"]
DEFENSIVOS = ["Consumo básico", "Servicios públicos", "Salud"]
PLAZOS_MOMENTUM = {"1 mes": 21, "3 meses": 63, "6 meses": 126, "12 meses": 252}


def ofensivo_defensivo(ret):
    """Índice ofensivos / defensivos (pesos iguales) a partir de los retornos % por
    sector. Sube = el mercado prefiere riesgo; baja = busca refugio."""
    of = (1 + ret[[c for c in OFENSIVOS if c in ret]].mean(axis=1).fillna(0) / 100).cumprod()
    de = (1 + ret[[c for c in DEFENSIVOS if c in ret]].mean(axis=1).fillna(0) / 100).cumprod()
    return of / de


def momentum(rs):
    """Retorno relativo vs mercado (%) de cada sector en varios plazos, con datos diarios."""
    filas = []
    for sec in rs.columns:
        r = rs[sec].dropna()
        fila = {"Sector": sec}
        for nombre, n in PLAZOS_MOMENTUM.items():
            fila[nombre] = (r.iloc[-1] / r.iloc[-1 - n] - 1) * 100 if len(r) > n else None
        filas.append(fila)
    return pd.DataFrame(filas).sort_values("1 mes", ascending=False).reset_index(drop=True)


def sensibilidad_macro(rel, macro):
    """Correlación entre el retorno relativo semanal de cada sector y el cambio semanal
    de cada variable macro (tasas en puntos, el resto en %)."""
    rel_s = rel.resample("W-FRI").sum(min_count=1)
    cambios = {}
    for nombre, serie in macro.items():
        w = serie.resample("W-FRI").last()
        cambios[nombre] = w.diff() if nombre.startswith("Tasa") else w.pct_change(fill_method=None) * 100
    cambios = pd.DataFrame(cambios)
    datos = rel_s.join(cambios, how="inner").dropna()
    return datos[list(rel.columns)].apply(lambda c: datos[list(macro)].corrwith(c)).T
