# Reporte de Estadística (Descriptiva y Avanzada)

## 1. Métricas clave por país
- **Confirmados (cum.)**
- **Fallecidos (cum.)**
- **CFR** = Fallecidos / Confirmados
- **Tasas por 100k**:
  - Confirmados_100k = (Confirmed / Population) * 100000
  - Deaths_100k = (Deaths / Population) * 100000
- Si `Population` no está disponible, se usa `Incident_Rate` como aproximación para Confirmados_100k.

## 2. Intervalos de confianza para CFR
- Se modela CFR como proporción binomial: p = D/C.
- IC (95%) Wilson:
  
  \[
  \hat{p}_\text{wilson} = \frac{p + z^2/(2C) \pm z \sqrt{\frac{p(1-p)}{C} + \frac{z^2}{4C^2}}}{1 + z^2/C}
  \]
  
  con \( z = 1.96 \) y \( C = \) Confirmados.

## 3. Test de hipótesis de proporciones (CFR entre dos países)
- **H0**: \( p_1 = p_2 \) (misma CFR)
- **H1**: \( p_1 \ne p_2 \)
- Estadístico z para comparación de dos proporciones con pool:
  
  \[
  \hat{p} = \frac{D_1 + D_2}{C_1 + C_2},\quad
  z = \frac{p_1 - p_2}{\sqrt{\hat{p}(1-\hat{p})(\frac{1}{C_1} + \frac{1}{C_2})}}
  \]

- Reportar p-valor y decisión con \(\alpha = 0.05\).

## 4. Detección de outliers
- **Z-score**: valores con |z| > 3.
- **IQR**: outliers por debajo de Q1 - 1.5·IQR o por encima de Q3 + 1.5·IQR.
- Aplicado a `Confirmed_100k`, `Deaths_100k`, `CFR`.

## 5. Gráfico de control (3σ) de muertes diarias
- Para una serie temporal de muertes diarias por país:
  - Media móvil y desviación (rolling).
  - Límites: CL = media, UCL = media + 3σ, LCL = max(media - 3σ, 0).
  - Señales de alerta: punto > UCL.

## 6. Resultados esperados
- Lista de países con CFR significativamente distinta.
- Países/fechas con anomalías (control chart).
- Outliers de tasas y CFR coherentes con realidades epidemiológicas (poblaciones pequeñas tienden a variabilidad mayor).

> La implementación está en `notebooks/laboratorio.ipynb` y visualizaciones integradas en `app/app.py`.

