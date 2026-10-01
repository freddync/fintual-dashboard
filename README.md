# Fintual Dashboard

Megacap + Large Cap (894 empresas) en Streamlit: RSI 5, MACD 6/13/5, fundamentales,
listado RSI 5 < 30 y seguimiento de empresas.

```
app.py          el dashboard
actualizar.py   descarga precios y/o fundamentales desde Yahoo
data/           precios, fundamentales y lista de empresas
.github/workflows/
  actualizar-precios.yml         cada hora, lun-vie, 9:00-20:00 NY
  actualizar-fundamentales.yml   día 1 de cada mes
```

## Datos

Las GitHub Actions actualizan todo solas y hacen commit; Streamlit Cloud toma los
cambios sin intervención. Para lanzarlas a mano: pestaña **Actions** → elegir la
Action → **Run workflow**.

## Seguimiento

En Streamlit Cloud la lista se guarda en un GitHub Gist privado, configurado en los
secrets de la app:

```toml
[seguimiento]
gist_id = "..."
token = "..."
```

Sin secrets (en tu computador) se guarda en `data/seguimiento.json`. Si copias los
mismos secrets a `.streamlit/secrets.toml`, el local y la nube comparten la lista.

## En tu computador

Doble click en `Dashboard.bat` (trae los datos de GitHub y abre el dashboard).
