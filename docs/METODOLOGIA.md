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

### Modelo: Random Forest clima + suelo (Iqbal et al., 2024)

**Configuración:**
- Algoritmo: `RandomForestRegressor`
- Hiperparámetros: `n_estimators=100`, `max_depth=20`
- Features: Variables climáticas por etapa (precipitación, Tmax, Tmin) + variables edáficas (índice de productividad, profundidad efectiva, textura, drenaje).
- Target: `Rinde_Detrended` (kg/ha), es decir, rendimiento ajustado por tendencia tecnológica.

**Script principal:** `src/models/train_with_soil.py`

### Validación temporal

- Entrenamiento: 1990–2015
- Prueba: 2016–2021
- Métricas (R², RMSE, MAE sobre `Rinde_Detrended`) se reportan en consola y en los informes (`informe_trigo.html`, `INFORME_TRIGO.md`).

### Explicabilidad (XAI)

Se emplean dos enfoques complementarios:

- **Feature importances (Gini):** ranking global de variables clima + suelo.
- **SHAP (SHapley Additive exPlanations):** impacto de cada variable en predicciones individuales.

**Figuras generadas:**
- `importance.png` — Importancia de variables (clima + suelo).
- `shap.png` — SHAP summary plot (impacto en el rinde ajustado).

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
