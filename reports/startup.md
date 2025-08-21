# Propuesta de Startup: Sentinel-Salud

## Problema
Los gestores sanitarios carecen de **señales tempranas** y **paneles integrados** para decidir acciones frente a brotes y saturación de recursos.

## Solución
**Sentinel-Salud**: plataforma de **vigilancia y predicción** que integra datos públicos (JHU, MINSA/INS) y propios (clínicas, municipalidades), con:
- **Semáforos de riesgo** por distrito/provincia.
- **Alertas tempranas** con gráficos de control (3σ) y detectores de cambio.
- **Pronósticos 14 días** con intervalos.
- **API** para interoperabilidad (tableros existentes).

## Cliente y aliados
- **Cliente**: MINSA, INS, EsSalud, municipalidades, clínicas privadas.
- **Aliados**: universidades, telcos (datos de movilidad), Streamlit Cloud.

## MVP
- Dashboard con:
  - Filtros por fecha/área, KPIs (Confirmados, Fallecidos, CFR, tasas 100k).
  - Pestañas: Visión general, Estadística avanzada, Modelado, Clustering, Calidad de datos.
  - Exportación de datos y gráficos.

## North Star Metric
- **Tiempo de detección de anomalías** (TDA) desde que se produce hasta que se alerta.

## KPIs
- % brotes detectados a tiempo
- MAE/MAPE del forecast
- Tasa de falsos positivos de alertas
- Usuarios activos y retención

## Modelo de negocio
- **SaaS** por suscripción (por jurisdicción), escalas por población.
- **API premium** para integraciones.
- **Licenciamiento** on-prem para privados.

## Roadmap
1. **Vigilancia**: ingestión de datos, KPIs, semáforos.
2. **Predicción**: pronósticos y control charts con alertas.
3. **Interoperabilidad**: API y conectores (HL7/FHIR).
4. **Optimización**: asignación de recursos con aprendizaje automático.

## Diferenciadores
- Enfoque en **alertas accionables** (no sólo visualizaciones).
- **Validación estadística** integrada (tests, IC, control de calidad).
- **Despliegue rápido** en Streamlit Cloud.

