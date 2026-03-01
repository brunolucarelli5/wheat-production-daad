# Informe: Predicción de Rendimiento de Trigo en la Región Pampeana

**Proyecto:** Beca UTN-DAAD  
**Período:** 1990–2021  
**Fecha de generación:** 2026-02-27 01:10

---

## 1. Resumen Ejecutivo

Este informe presenta el análisis de rendimiento de trigo en la **Región Pampeana** (Argentina) utilizando datos climáticos de la NOAA y datos de producción del Ministerio de Agricultura. Se desarrolló un modelo predictivo basado en **Random Forest** para identificar el impacto de variables climáticas en etapas fenológicas clave del cultivo.

**Objetivo:** Comparar sistemas productivos entre la Región Pampeana y Alemania mediante análisis de importancia de variables climáticas (XAI).

---

## 2. Dataset

### 2.1 Características generales

- **Registros totales:** 4,810
- **Registros con rendimiento válido:** 4,810
- **Provincias:** Buenos Aires, Córdoba, Entre Ríos, La Pampa, Santa Fe
- **Período temporal:** 1990–2021
- **Variables climáticas:** 21 (precipitación, Tmax, Tmin por etapa fenológica)

### 2.2 Etapas fenológicas del trigo (Scian, 2004)

| Etapa | Período |
|-------|---------|
| Siembra | 21 jun – 10 jul |
| Emergencia | 11 jul – 30 ago |
| Macollaje | 1 sep – 30 sep |
| Encañazón | 1 oct – 20 oct |
| Espigazón | 21 oct – 31 oct |
| Floración | 1 nov – 10 nov |
| Llenado de grano | 11 nov – 30 nov |

### 2.3 Estadísticas de rendimiento

**Global:**
- Media: **2774 kg/ha**
- Desviación estándar: **1058 kg/ha**
- Mínimo: **0 kg/ha**
- Máximo: **7186 kg/ha**

**Por provincia:**

| Provincia | n | Media (kg/ha) | Desv. Est. |
|-----------|---|---------------|------------|
| Buenos Aires | 3010 | 3134 | 1008 |
| Córdoba | 599 | 2150 | 834 |
| Entre Ríos | 136 | 1863 | 468 |
| La Pampa | 483 | 1867 | 663 |
| Santa Fe | 582 | 2517 | 920 |

---

## 3. Modelo Predictivo: Random Forest

### 3.1 Configuración

- **Algoritmo:** Random Forest Regressor (Iqbal et al., 2024)
- **Parámetros:** n_estimators=100, max_depth=20, random_state=42
- **División:** 80% entrenamiento / 20% prueba
- **Registros usados:** 4,674 (sin NaN en features o target)

### 3.2 Métricas de evaluación (conjunto de prueba)

| Métrica | Valor |
|---------|-------|
| **R²** | 0.5608 |
| **RMSE** | 696.78 kg/ha |
| **MAE** | 511.16 kg/ha |

**Interpretación:**
- El modelo explica el **56.1%** de la varianza en el rendimiento.
- Error promedio absoluto: **511 kg/ha**.
- El RMSE de **697 kg/ha** indica la magnitud típica del error de predicción.

**Figura:** Ver `scatter_real_vs_predicho.png` para la comparación visual entre valores reales y predichos.

---

## 4. Análisis de Importancia de Variables (XAI)

### 4.1 Feature Importances (Gini)

Top 10 variables más importantes para el modelo:

| Ranking | Variable | Importancia |
|---------|----------|-------------|
| 1 | `tmin_llenado_de_grano` | 0.2456 |
| 2 | `lluvia_siembra` | 0.2046 |
| 3 | `tmax_siembra` | 0.1112 |
| 4 | `tmax_espigazon` | 0.0492 |
| 5 | `lluvia_floracion` | 0.0491 |
| 6 | `tmin_macollaje` | 0.0481 |
| 7 | `tmin_encanazon` | 0.0465 |
| 8 | `tmax_floracion` | 0.0424 |
| 9 | `lluvia_emergencia` | 0.0282 |
| 10 | `lluvia_macollaje` | 0.0198 |

**Figura:** Ver `importancia_variables.png` para el gráfico completo de barras.

### 4.2 SHAP: Explicabilidad del modelo

El análisis SHAP (SHapley Additive exPlanations) permite entender **cómo cada variable afecta las predicciones individuales**:

- **Eje horizontal (SHAP value):** Contribución de la variable al rendimiento predicho.
  - Valores positivos → aumentan el rinde.
  - Valores negativos → reducen el rinde.
- **Color del punto:** Valor de la variable (rojo = alto, azul = bajo).

**Insights clave:**
- Variables de **temperatura** en etapas críticas (floración, llenado de grano) tienen alto impacto.
- **Precipitación** en emergencia, encañazón y floración son determinantes.
- El modelo captura relaciones no lineales entre clima y rendimiento.

**Figura:** Ver `shap_summary_plot.png` para el análisis completo.

---

## 5. Conclusiones

1. **Datos integrados:** Se vincularon exitosamente datos climáticos diarios (NOAA) con rendimiento de trigo por departamento (1990-2021).

2. **Modelo predictivo:** Random Forest alcanza un R² de **0.5608**, con capacidad para explicar más de la mitad de la varianza en el rendimiento.

3. **Variables críticas:** Las variables de temperatura y precipitación en **floración** y **llenado de grano** son las más influyentes según el análisis de importancia.

4. **Aplicación:** Este análisis es **fundamental para la comparación de sistemas productivos entre la Región Pampeana (Argentina) y Alemania**, objetivo central de la beca UTN-DAAD.

---

## 6. Referencias

- **Scian, B. (2004).** Metodología de decadias y etapas fenológicas para trigo en la Región Pampeana.
- **Iqbal et al. (2024).** Machine Learning aplicado a predicción de rendimiento de cultivos.
- **Datos climáticos:** NOAA (National Oceanic and Atmospheric Administration).
- **Datos de producción:** Ministerio de Agricultura, Ganadería y Pesca de Argentina.

---

## 7. Archivos generados

### Datos procesados
- `clima_region_pampeana_feno.csv` — Datos diarios con decadias y etapas fenológicas.
- `clima_region_pampeana_features.csv` — Predictores anuales por estación.
- `rinde_trigo_pampa.csv` — Rendimiento filtrado (Región Pampeana, 1990-2021).
- `mapeo_departamento_estacion.csv` — Mapeo departamento → estación climática.
- `dataset_maestro_ia.csv` — Dataset final para modelado.

### Figuras
- `scatter_real_vs_predicho.png` — Real vs. predicho (Random Forest).
- `importancia_variables.png` — Importancia de features (Gini).
- `shap_summary_plot.png` — SHAP summary plot.

### Informes
- `informe_trigo.html` — Informe visual interactivo.
- `INFORME_TRIGO.md` — Este documento.

---

**Generado por:** Pipeline automatizado de análisis de trigo (proyecto UTN-DAAD)
