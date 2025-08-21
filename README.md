# incidenciasCOVID
# Laboratorio 1.1 — COVID-19 (JHU)

Proyecto completo que cubre:
- **Parte 1–2**: EDA, tablas y gráficos con el daily report **2022-04-18** (JHU).
- **Parte 2 (avanzada)**: Métricas, IC de CFR, test de proporciones, outliers, gráfico de control.
- **Parte 3**: Series de tiempo por país, suavizado 7d, **SARIMA**/ETS, backtesting y bandas.
- **Parte 4**: Clustering (K-means) y **PCA** con variables de tasas/CFR/crecimiento.
- **Parte 5**: Dashboard en **Streamlit** con filtros, tabs y exportaciones.
- **Parte 6**: Propuesta de startup/innovación.

## Estructura
app/app.py
reports/estadistica.md
reports/startup.md
notebooks/laboratorio.ipynb
data/raw/04-18-2022.csv
data/processed/sample50.xlsx
requirements.txt
.gitignore
README.md

## Datos
- Daily report (18-04-2022): `data/raw/04-18-2022.csv`
- Para series de tiempo (Parte 3), el notebook descarga múltiples daily reports alrededor de esa fecha directamente desde el repo público de JHU.


