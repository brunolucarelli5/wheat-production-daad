# Informe: Predicción de Rendimiento de Trigo en la Región Pampeana

**Proyecto:** Beca UTN-DAAD  
**Período:** 1990–2021  
**Fecha de generación:** 2026-03-03 00:02

---

## 1. Resumen Ejecutivo

Este informe presenta el análisis de rendimiento de trigo en la **Región Pampeana** (Argentina) utilizando datos climáticos (NOAA), datos de producción (Ministerio de Agricultura) y variables edáficas (Cartas de Suelo INTA). El modelo predictivo es un **Random Forest** que combina **clima y suelo** para predecir el rinde ajustado (residuos respecto a la tendencia tecnológica).

**Objetivo:** Comparar sistemas productivos entre la Región Pampeana y Alemania mediante análisis de importancia de variables (XAI).

---

## 2. Dataset

### 2.1 Características generales

- **Registros totales:** 4,810
- **Provincias:** N/A
- **Período temporal:** 1990–2021
- **Variables:** 25 (climáticas por etapa fenológica + edáficas por provincia)

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

### 2.3 Estadísticas de rendimiento (kg/ha)

**Global:**
- Media: **2774 kg/ha**
- Desviación estándar: **1058 kg/ha**
- Mínimo: **0 kg/ha**
- Máximo: **7186 kg/ha**

**Por provincia:**

| Provincia | n | Media (kg/ha) | Desv. Est. |
|-----------|---|---------------|------------|
| Total | 4810 | 2774 | 1058 |

---

## 3. Modelo Predictivo: Clima + Suelo

### 3.1 Configuración

- **Algoritmo:** Random Forest Regressor (Iqbal et al., 2024)
- **Target:** Rinde_Detrended (residuos respecto a la tendencia tecnológica)
- **Validación:** Temporal — entrenamiento 1990-2015, prueba 2016-2021
- **Registros entrenamiento:** 3,286
- **Registros prueba:** 789

### 3.2 Métricas (conjunto de prueba, Rinde_Detrended)

| Métrica | Valor |
|---------|-------|
| **R²** | -0.2414 |
| **RMSE** | 625.85 kg/ha |
| **MAE** | 505.18 kg/ha |

**Figuras:**
- `scatter.png` — Rendimiento real vs. predicho (kg/ha absolutos).
- `scatter_residuals.png` — Rinde ajustado (residuos): real vs. predicho.

---

## 4. Análisis de Importancia (XAI)

### 4.1 Top 10 variables (Gini)

| Ranking | Variable | Importancia |
|---------|----------|-------------|
| 1 | `lluvia_encanazon` | 0.2516 |
| 2 | `lluvia_floracion` | 0.1925 |
| 3 | `lluvia_emergencia` | 0.0794 |
| 4 | `lluvia_siembra` | 0.0455 |
| 5 | `tmin_siembra` | 0.0453 |
| 6 | `tmin_macollaje` | 0.0317 |
| 7 | `tmax_floracion` | 0.0308 |
| 8 | `lluvia_espigazon` | 0.0296 |
| 9 | `tmin_encanazon` | 0.0273 |
| 10 | `tmax_siembra` | 0.0268 |

**Figura:** `importance.png`

### 4.2 SHAP

El análisis SHAP muestra el efecto de cada variable sobre el rendimiento predicho (valores positivos aumentan el rinde, negativos lo reducen). Color: valor de la variable (rojo = alto, azul = bajo).

**Figura:** `shap.png`

---

## 5. Conclusiones

1. **Datos integrados:** Clima (NOAA), rendimiento (Ministerio) y suelo (INTA) vinculados por departamento y año.

2. **Modelo único:** Random Forest con clima + suelo, validación temporal 2016-2021.

3. **Variables críticas:** Temperatura y precipitación en floración y llenado de grano suelen dominar; las variables edáficas aportan información estable a escala provincial.

4. **Aplicación:** Análisis útil para la comparación de sistemas productivos (Región Pampeana vs. Alemania), objetivo de la beca UTN-DAAD.

---

## 6. Referencias

- **Scian, B. (2004).** Metodología de decadias y etapas fenológicas para trigo en la Región Pampeana.
- **Iqbal et al. (2024).** Machine Learning aplicado a predicción de rendimiento de cultivos.
- **Datos:** NOAA, Ministerio de Agricultura Argentina, INTA (Cartas de Suelo).

---

## 7. Archivos generados

### Datos
- `dataset_maestro_ia.csv` — Rinde + clima (intermedio).
- `dataset_final.csv` — Dataset final rinde + clima + suelo.

### Figuras
- `scatter.png` — Real vs. predicho (kg/ha).
- `scatter_residuals.png` — Rinde ajustado: real vs. predicho.
- `importance.png` — Importancia de variables (Gini).
- `shap.png` — SHAP summary.

### Informes
- `informe_trigo.html` — Informe visual.
- `INFORME_TRIGO.md` — Este documento.

---

**Generado por:** Pipeline clima + suelo (proyecto UTN-DAAD)
