import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io

import plotly.express as px
import plotly.graph_objects as go

# ----------------------------
# Configuración base
# ----------------------------
st.set_page_config(page_title="Laboratorio 1.1 - COVID JHU", layout="wide")

@st.cache_data(show_spinner=False)
def load_daily(date_str: str) -> pd.DataFrame:
    """
    Carga un daily report de JHU por fecha 'MM-DD-YYYY'.
    """
    url = f"https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/{date_str}.csv"
    df = pd.read_csv(url)
    # Normaliza nombres esperados
    rename_map = {
        "Province/State": "Province_State",
        "Country/Region": "Country_Region",
        "Last Update": "Last_Update",
        "Latitude": "Lat",
        "Longitude": "Long_",
    }
    df = df.rename(columns=rename_map)
    # Asegurar columnas clave
    for col in ["Province_State","Country_Region","Confirmed","Deaths","Recovered","Active","Incident_Rate","Case_Fatality_Ratio"]:
        if col not in df.columns:
            df[col] = np.nan
    return df

@st.cache_data(show_spinner=False)
def load_range(end_date: str, days_back: int = 60) -> pd.DataFrame:
    """
    Descarga múltiples daily reports hacia atrás desde end_date (MM-DD-YYYY).
    Devuelve un DF long con índices por fecha y país.
    """
    end = datetime.strptime(end_date, "%m-%d-%Y")
    frames = []
    for i in range(days_back+1):
        d = end - timedelta(days=i)
        tag = d.strftime("%m-%d-%Y")
        try:
            df = load_daily(tag)
            df["date"] = d.date()
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out

def compute_country_agg(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("Country_Region", dropna=False).agg(
        Confirmed=("Confirmed","sum"),
        Deaths=("Deaths","sum"),
        Recovered=("Recovered","sum"),
        Active=("Active","sum"),
        Incident_Rate=("Incident_Rate","mean"),
        CFR_mean=("Case_Fatality_Ratio","mean")
    ).reset_index()
    # CFR empirico robusto
    g["CFR"] = np.where(g["Confirmed"]>0, g["Deaths"]/g["Confirmed"], np.nan)
    # Tasas por 100k: si no hay Population, aproximar Confirmed_100k por Incident_Rate
    if "Population" in df.columns and df["Population"].notna().any():
        pop = df.groupby("Country_Region")["Population"].sum().rename("Population").reset_index()
        g = g.merge(pop, on="Country_Region", how="left")
        g["Confirmed_100k"] = g["Confirmed"]/g["Population"]*1e5
        g["Deaths_100k"] = g["Deaths"]/g["Population"]*1e5
    else:
        g["Confirmed_100k"] = g["Incident_Rate"]
        g["Deaths_100k"] = g["CFR"]*g["Confirmed_100k"]
    return g

def export_df(df: pd.DataFrame, name: str) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("Filtros")
default_date = "04-18-2022"
date_str = st.sidebar.text_input("Fecha del daily (MM-DD-YYYY)", value=default_date)
days_back = st.sidebar.slider("Ventana histórica (días hacia atrás)", min_value=14, max_value=180, value=90, step=7)
min_confirmed = st.sidebar.number_input("Umbral mínimo de confirmados (país)", min_value=0, value=1000, step=500)

st.sidebar.divider()
st.sidebar.caption("Exportaciones")
export_selection = st.sidebar.multiselect("¿Qué exportar?", ["País (agg)","Crudo (daily)"], default=[])

# ----------------------------
# Carga de datos
# ----------------------------
with st.spinner("Descargando datos..."):
    df_today = load_daily(date_str)
    df_hist = load_range(date_str, days_back)
country_agg = compute_country_agg(df_today)
country_agg = country_agg[country_agg["Confirmed"]>=min_confirmed].sort_values("Confirmed", ascending=False)

countries = sorted(country_agg["Country_Region"].dropna().unique().tolist())
sel_countries = st.sidebar.multiselect("Países", countries, default=countries[:10])

# ----------------------------
# KPIs
# ----------------------------
st.title("Laboratorio 1.1 — Dashboard COVID (JHU)")
top = country_agg.head(10)
kpi1 = int(top["Confirmed"].sum())
kpi2 = int(top["Deaths"].sum())
kpi3 = float((top["Deaths"].sum()/max(top["Confirmed"].sum(),1))*100)

col1, col2, col3 = st.columns(3)
col1.metric("Confirmados (Top-10)", f"{kpi1:,}")
col2.metric("Fallecidos (Top-10)", f"{kpi2:,}")
col3.metric("CFR (Top-10) %", f"{kpi3:.2f}")

# ----------------------------
# Tabs
# ----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Visión general", "Estadística avanzada", "Modelado temporal", "Clustering y PCA", "Calidad de datos"
])

with tab1:
    st.subheader("Top-N por Confirmados y Fallecidos")
    colA, colB = st.columns(2)
    with colA:
        fig = px.bar(top, x="Country_Region", y="Confirmed", title="Top Confirmados")
        st.plotly_chart(fig, use_container_width=True)
    with colB:
        fig = px.bar(top.sort_values("Deaths", ascending=False), x="Country_Region", y="Deaths", title="Top Fallecidos")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Mapa (si hay lat/long por provincia)")
    if {"Lat","Long_","Confirmed"}.issubset(df_today.columns):
        geo = df_today.dropna(subset=["Lat","Long_"])
        fig = px.scatter_geo(
            geo, lat="Lat", lon="Long_", size="Confirmed",
            hover_name="Country_Region", color="Country_Region",
            title="Distribución geográfica (puntos por provincia/estado)"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hay columnas Lat/Long_ para mapa en este daily.")

with tab2:
    st.subheader("CFR y Tasas por 100k")
    st.dataframe(country_agg[["Country_Region","Confirmed","Deaths","CFR","Confirmed_100k","Deaths_100k"]].round(4))

    st.markdown("### Intervalos de Confianza (Wilson, 95%)")
    from math import sqrt
    z = 1.96
    ci = []
    for _, r in country_agg.iterrows():
        C, D = r["Confirmed"], r["Deaths"]
        if C>0:
            p = D/C
            denom = 1 + z**2/C
            center = p + z**2/(2*C)
            rad = z * sqrt(p*(1-p)/C + z**2/(4*C**2))
            lo = (center - rad)/denom
            hi = (center + rad)/denom
        else:
            lo=hi=np.nan
        ci.append((r["Country_Region"], p if C>0 else np.nan, lo, hi))
    df_ci = pd.DataFrame(ci, columns=["Country","CFR","CFR_L","CFR_U"]).dropna()
    st.dataframe(df_ci.sort_values("CFR", ascending=False).round(4))

    st.markdown("### Test de proporciones (CFR) entre dos países")
    c1, c2 = st.columns(2)
    with c1:
        a = st.selectbox("País A", countries, index=0)
    with c2:
        b = st.selectbox("País B", countries, index=1 if len(countries)>1 else 0)
    def prop_test(a, b):
        A = country_agg.loc[country_agg["Country_Region"]==a, ["Confirmed","Deaths"]].iloc[0]
        B = country_agg.loc[country_agg["Country_Region"]==b, ["Confirmed","Deaths"]].iloc[0]
        C1,D1 = int(A["Confirmed"]), int(A["Deaths"])
        C2,D2 = int(B["Confirmed"]), int(B["Deaths"])
        if C1==0 or C2==0:
            return np.nan, np.nan
        phat = (D1 + D2)/(C1 + C2)
        z = (D1/C1 - D2/C2)/np.sqrt(phat*(1-phat)*(1/C1 + 1/C2))
        from scipy.stats import norm
        pval = 2*(1 - norm.cdf(abs(z)))
        return z, pval
    zstat, pval = prop_test(a,b)
    st.write(f"**z = {zstat:.3f}, p-valor = {pval:.4f}**  → α=0.05 {'Rechaza H0' if pval<0.05 else 'No rechaza H0'}")

with tab3:
    st.subheader("Series de tiempo y pronóstico (14 días)")
    if df_hist.empty:
        st.warning("No se pudo construir la serie histórica. Ajusta la fecha o la ventana.")
    else:
        # Construir serie por país (agregada diaria)
        ts = (df_hist.groupby(["date","Country_Region"])["Deaths"].sum()
              .reset_index().sort_values("date"))
        target_country = st.selectbox("País para pronóstico", countries, index=0)
        y = ts.loc[ts["Country_Region"]==target_country, ["date","Deaths"]].set_index("date").asfreq("D").fillna(0)
        y["Deaths_7d"] = y["Deaths"].rolling(7).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y.index, y=y["Deaths"], name="Muertes"))
        fig.add_trace(go.Scatter(x=y.index, y=y["Deaths_7d"], name="Media 7d"))
        fig.update_layout(title=f"Serie diaria — {target_country}")
        st.plotly_chart(fig, use_container_width=True)

        # Modelo SARIMA simple con pmdarima (auto_arima)
        import pmdarima as pm
        try:
            model = pm.auto_arima(y["Deaths"], seasonal=False, stepwise=True, suppress_warnings=True)
            fc, confint = model.predict(n_periods=14, return_conf_int=True)
            idx_fc = pd.date_range(y.index.max() + pd.Timedelta(days=1), periods=14, freq="D")

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=y.index, y=y["Deaths"], name="Muertes"))
            fig2.add_trace(go.Scatter(x=idx_fc, y=fc, name="Pronóstico"))
            fig2.add_trace(go.Scatter(x=idx_fc, y=confint[:,0], name="IC 95% (inf)", line=dict(dash="dash")))
            fig2.add_trace(go.Scatter(x=idx_fc, y=confint[:,1], name="IC 95% (sup)", line=dict(dash="dash")))
            fig2.update_layout(title=f"Pronóstico 14 días — {target_country}")
            st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            st.info(f"No se pudo ajustar SARIMA: {e}")

        # Backtesting simple (últimos 14 días)
        if len(y) > 30:
            split = -14
            train, test = y["Deaths"][:split], y["Deaths"][split:]
            try:
                m = pm.auto_arima(train, seasonal=False, stepwise=True, suppress_warnings=True)
                pred = pd.Series(m.predict(n_periods=len(test)), index=test.index)
                mae = (test - pred).abs().mean()
                mape = ((test - pred).abs()/(test.replace(0, np.nan))).mean()*100
                st.write(f"**MAE:** {mae:.2f}  |  **MAPE:** {mape:.2f}%")
            except Exception as e:
                st.info(f"Backtesting no disponible: {e}")

with tab4:
    st.subheader("Clustering y PCA (país)")
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    X = country_agg[["Confirmed_100k","Deaths_100k","CFR"]].replace([np.inf,-np.inf], np.nan).dropna()
    base = country_agg.set_index("Country_Region").loc[X.index]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    k = st.slider("k (clusters)", 2, 6, 3)
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = km.fit_predict(Xs)

    pca = PCA(n_components=2, random_state=42)
    PC = pca.fit_transform(Xs)
    scat = pd.DataFrame({"PC1": PC[:,0], "PC2": PC[:,1], "cluster": labels, "Country": X.index})
    fig = px.scatter(scat, x="PC1", y="PC2", color=scat["cluster"].astype(str), hover_name="Country", title="PCA (PC1 vs PC2)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Perfiles (aprox.)**")
    st.write(scat.groupby("cluster")[["PC1","PC2"]].mean().round(2))

with tab5:
    st.subheader("Nulos e inconsistencias")
    nulls = df_today.isna().mean().sort_values(ascending=False)*100
    st.write("Porcentaje de nulos por columna:")
    st.dataframe(nulls.to_frame("Nulos_%").round(2))

    st.markdown("### Gráfico de control (3σ) de muertes diarias")
    if not df_hist.empty:
        country = st.selectbox("País", countries, index=0, key="ctrl_country")
        y = (df_hist.groupby(["date","Country_Region"])["Deaths"].sum()
             .reset_index().sort_values("date"))
        y = y.loc[y["Country_Region"]==country, ["date","Deaths"]].set_index("date").asfreq("D").fillna(0)
        roll = y["Deaths"].rolling(14)
        mu, sigma = roll.mean(), roll.std().replace(0, np.nan)
        cl = mu
        ucl = mu + 3*sigma
        lcl = (mu - 3*sigma).clip(lower=0)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y.index, y=y["Deaths"], name="Muertes"))
        fig.add_trace(go.Scatter(x=y.index, y=cl, name="CL"))
        fig.add_trace(go.Scatter(x=y.index, y=ucl, name="UCL", line=dict(dash="dash")))
        fig.add_trace(go.Scatter(x=y.index, y=lcl, name="LCL", line=dict(dash="dash")))
        fig.update_layout(title=f"Control chart — {country}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Sin historial para gráfico de control.")

# ----------------------------
# Exportaciones
# ----------------------------
if export_selection:
    st.subheader("Descargas")
    if "País (agg)" in export_selection:
        st.download_button("Descargar país (CSV)", data=export_df(country_agg, "country_agg.csv"),
                           file_name="country_agg.csv", mime="text/csv")
    if "Crudo (daily)" in export_selection:
        st.download_button("Descargar daily crudo (CSV)", data=export_df(df_today, "daily.csv"),
                           file_name="daily.csv", mime="text/csv")

