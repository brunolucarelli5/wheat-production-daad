# Metodología del Proyecto

## Resumen

Este proyecto implementa un pipeline de Machine Learning para predecir el rendimiento de trigo en la Región Pampeana de Argentina utilizando variables climáticas por etapa fenológica.

---

## Pipeline de Datos

### 1. Ingesta de Datos

**Fuentes:**
- **NOAA:** Datos climáticos diarios (precipitación, Tmax, Tmin) de estaciones meteorológicas.
- **Ministerio de Agricultura:** Datos de rendimiento de trigo por departamento (1990-2021).

**Script:** `src/data/make_dataset.py`

### 2. Mapeo de Etapas Fenológicas (Scian, 2004)

Se asigna cada fecha a:
- **Decadia:** Período de 10 días dentro del mes (1, 2 o 3).
- **Etapa fenológica:** Siembra, Emergencia, Macollaje, Encañazón, Espigazón, Floración, Llenado de grano.

**Fundamento:** Scian (2004) establece que las necesidades hídricas y térmicas del trigo varían según la etapa fenológica.

**Script:** `src/features/build_phenological_features.py`

### 3. Agregación de Predictores Climáticos

Por cada estación y año, se calcula:
- **Suma de precipitación** por etapa.
- **Promedio de Tmax y Tmin** por etapa.

Resultado: tabla pivotada con columnas como `lluvia_emergencia`, `tmax_floracion`, `tmin_macollaje`.

**Script:** `src/features/aggregate_climate.py`

### 4. Merge Rinde + Clima

Se vincula el rendimiento de trigo con las variables climáticas mediante:
1. Mapeo geográfico: Departamento → Estación climática más cercana.
2. Join por estación y año.

**Scripts:** `scripts/preparar_insumos_merge.py` + `scripts/merge_rinde_clima.py`

---

## Modelado

### Modelo Base: Random Forest (Iqbal et al., 2024)

**Configuración:**
- Algoritmo: `RandomForestRegressor`
- Hiperparámetros: `n_estimators=100`, `max_depth=20`
- Features: Variables climáticas por etapa (21 features)
- Target: Rendimiento (kg/ha)

**Script:** `src/models/train_model.py`

### Validación

#### Split Aleatorio (80/20)

**Resultados:**
- R² = 0.5608
- RMSE = 696.78 kg/ha
- MAE = 511.16 kg/ha

**Limitación:** Sobreestima la capacidad predictiva al mezclar años.

#### Validación Temporal (1990-2015 → 2016-2021)

**Rinde absoluto:**
- R² = -0.9549 (no generaliza)
- RMSE = 1.416,79 kg/ha
- MAE = 1.166,95 kg/ha

**Causa:** Tendencia tecnológica (mejora de variedades, prácticas) confunde al modelo.

**Script:** `src/models/validate_temporal.py`

### Detrending: Eliminación de Tendencia Tecnológica

**Método:**
1. Regresión lineal por departamento: `Rinde ~ Año`
2. Residuo: `Rinde_Detrended = Rinde real - Rinde tendencia`
3. Reentrenamiento sobre residuos.

**Resultados (validación temporal):**
- R² = -0.0378 (mejora de +96%)
- RMSE = 613.13 kg/ha (mejora de -57%)
- MAE = 496.49 kg/ha (mejora de -57%)

**Interpretación:** El modelo ahora captura **solo la señal climática**, sin confundirse con la mejora tecnológica.

**Script:** `src/models/train_detrended.py`

---

## Explicabilidad (XAI)

### Feature Importances (Gini)

Mide la importancia de cada variable en las divisiones del Random Forest.

### SHAP (SHapley Additive exPlanations)

Explica el impacto de cada variable en predicciones individuales:
- **Eje X:** Efecto SHAP sobre el rendimiento (kg/ha).
- **Color:** Valor de la variable (rojo = alto, azul = bajo).

**Insights:**
- Variables de temperatura en floración y llenado de grano tienen alto impacto.
- Precipitación en emergencia y encañazón son determinantes.

**Script:** `src/models/explain_model.py`

---

## Informes

### HTML Interactivo

Informe visual con gráficos embebidos, métricas y análisis.

**Script:** `src/visualization/generate_html_report.py`  
**Salida:** `reports/informe_trigo.html`

### Markdown

Informe escrito con tablas, métricas y referencias.

**Script:** `src/visualization/generate_markdown_report.py`  
**Salida:** `reports/INFORME_TRIGO.md`

---

## Referencias Bibliográficas

1. **Scian, B. (2004).** Metodología de decadias y etapas fenológicas para trigo en la Región Pampeana.
2. **Iqbal et al. (2024).** Machine Learning aplicado a predicción de rendimiento de cultivos.
3. **NOAA.** National Oceanic and Atmospheric Administration - Climate Data.
4. **Ministerio de Agricultura, Ganadería y Pesca de Argentina.** Datos de producción agrícola.

---

## Próximos Pasos

1. **Datos de Alemania:** Replicar el pipeline con datos climáticos y de rendimiento de Alemania.
2. **Comparación:** Analizar diferencias en importancia de variables entre regiones.
3. **Modelos avanzados:** Probar XGBoost, redes neuronales, modelos ensemble.
4. **Features adicionales:** Índices de sequía (SPEI, SPI), variables de suelo, tecnología agrícola.
