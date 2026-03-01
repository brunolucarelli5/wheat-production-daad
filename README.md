# Predicción de Rendimiento de Trigo en la Región Pampeana

**Proyecto:** Beca UTN-DAAD  
**Objetivo:** Análisis comparativo de sistemas productivos de trigo entre la Región Pampeana (Argentina) y Alemania mediante Machine Learning y variables climáticas.

---

## Estructura del Proyecto

```
daad/
├── data/
│   ├── raw/                          # Datos originales (sin modificar)
│   │   ├── noaa/                     # Datos climáticos NOAA por estación
│   │   └── DATOS TRIGO - ARGENTINA - 1990 A 2025.csv
│   └── processed/                    # Datos procesados y transformados
│       ├── clima_region_pampeana.csv
│       ├── clima_region_pampeana_feno.csv
│       ├── clima_region_pampeana_features.csv
│       ├── rinde_trigo_pampa.csv
│       ├── mapeo_departamento_estacion.csv
│       ├── dataset_maestro_ia.csv
│       └── dataset_maestro_ia_detrended.csv
│
├── src/                              # Código fuente modular
│   ├── data/                         # Scripts de ingesta y limpieza
│   │   └── make_dataset.py           # Unificar CSVs de NOAA
│   ├── features/                     # Ingeniería de características
│   │   ├── build_phenological_features.py  # Mapeo de etapas fenológicas
│   │   └── aggregate_climate.py      # Agregación anual por estación
│   ├── models/                       # Entrenamiento y evaluación
│   │   ├── train_model.py            # Random Forest (split aleatorio)
│   │   ├── validate_temporal.py      # Validación temporal (1990-2015 vs 2016-2021)
│   │   ├── train_detrended.py        # Modelo sobre residuos (detrending)
│   │   └── explain_model.py          # XAI: importancia + SHAP
│   └── visualization/                # Generación de reportes
│       ├── generate_html_report.py   # Informe HTML interactivo
│       └── generate_markdown_report.py  # Informe Markdown
│
├── scripts/                          # Scripts legacy (mantener por compatibilidad)
│   ├── preparar_insumos_merge.py
│   ├── merge_rinde_clima.py
│   └── ejecutar_pipeline_completo.py
│
├── notebooks/                        # Jupyter notebooks para exploración
│
├── reports/                          # Informes y visualizaciones finales
│   ├── figures/                      # Gráficos generados
│   │   ├── scatter_real_vs_predicho.png
│   │   ├── importancia_variables.png
│   │   ├── shap_summary_plot.png
│   │   ├── mae_por_año_temporal.png
│   │   ├── scatter_temporal.png
│   │   ├── scatter_detrended_temporal.png
│   │   ├── mae_por_año_detrended.png
│   │   └── ejemplo_tendencia.png
│   ├── informe_trigo.html            # Informe visual interactivo
│   └── INFORME_TRIGO.md              # Informe escrito
│
├── models/                           # Modelos entrenados (serialized)
│
├── docs/                             # Documentación y papers
│   ├── papers/                       # Artículos científicos (Scian, Iqbal, etc.)
│   └── proyecto/                     # Documentos del proyecto
│
├── requirements.txt                  # Dependencias Python
├── README.md                         # Este archivo
└── .gitignore
```

---

## Instalación

### 1. Crear entorno virtual

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate  # Windows
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## Pipeline de Datos y Modelado

### Paso 1: Unificar datos climáticos NOAA

```bash
python src/data/make_dataset.py
```

**Salida:** `data/processed/clima_argentina_completo.csv`

### Paso 2: Mapear etapas fenológicas

```bash
python src/features/build_phenological_features.py
```

**Salida:** `data/processed/clima_region_pampeana_feno.csv` (con columnas `DECADIA` y `ETAPA_FENOLOGICA`)

### Paso 3: Agregar predictores climáticos

```bash
python src/features/aggregate_climate.py
```

**Salida:** `data/processed/clima_region_pampeana_features.csv` (predictores anuales por estación)

### Paso 4: Preparar rinde y mapeo

```bash
python scripts/preparar_insumos_merge.py
```

**Salida:**
- `data/processed/rinde_trigo_pampa.csv`
- `data/processed/mapeo_departamento_estacion.csv`

### Paso 5: Merge rinde + clima

```bash
python scripts/merge_rinde_clima.py
```

**Salida:** `data/processed/dataset_maestro_ia.csv`

### Paso 6: Entrenar modelo Random Forest

```bash
python src/models/train_model.py
```

**Salida:** Métricas + `reports/figures/scatter_real_vs_predicho.png`

### Paso 7: Validación temporal

```bash
python src/models/validate_temporal.py
```

**Salida:** Métricas temporales + figuras MAE por año

### Paso 8: Detrending y reentrenamiento

```bash
python src/models/train_detrended.py
```

**Salida:** `data/processed/dataset_maestro_ia_detrended.csv` + métricas mejoradas

### Paso 9: Análisis XAI (importancia + SHAP)

```bash
python src/models/explain_model.py
```

**Salida:** `reports/figures/importancia_variables.png` + `shap_summary_plot.png`

### Paso 10: Generar informes

```bash
python src/visualization/generate_html_report.py
python src/visualization/generate_markdown_report.py
```

**Salida:**
- `reports/informe_trigo.html`
- `reports/INFORME_TRIGO.md`

---

## Pipeline Completo (Un Solo Comando)

```bash
python scripts/ejecutar_pipeline_completo.py
```

Ejecuta todos los pasos secuencialmente y genera todos los archivos.

---

## Resultados Principales

### Modelo Random Forest (Split Aleatorio)

| Métrica | Valor |
|---------|-------|
| R² | 0.5608 |
| RMSE | 696.78 kg/ha |
| MAE | 511.16 kg/ha |

### Validación Temporal (1990-2015 → 2016-2021)

**Rinde absoluto:**
- R² = -0.9549 (no generaliza a años futuros)

**Rinde_Detrended (sin tendencia tecnológica):**
- R² = -0.0378
- RMSE = 613.13 kg/ha
- MAE = 496.49 kg/ha

**Mejora:** El detrending reduce el error en ~57% al eliminar el sesgo tecnológico.

---

## Etapas Fenológicas del Trigo (Scian, 2004)

| Etapa | Período | Código |
|-------|---------|--------|
| Siembra | 21 jun – 10 jul | jn3-jl1 |
| Emergencia | 11 jul – 30 ago | jl2-ag3 |
| Macollaje | 1 sep – 30 sep | se1-se3 |
| Encañazón | 1 oct – 20 oct | oc1-oc2 |
| Espigazón | 21 oct – 31 oct | oc3 |
| Floración | 1 nov – 10 nov | no1 |
| Llenado de grano | 11 nov – 30 nov | no2-no3 |

---

## Datos

### Fuentes

- **Clima:** NOAA (National Oceanic and Atmospheric Administration)
- **Producción:** Ministerio de Agricultura, Ganadería y Pesca de Argentina

### Región Pampeana

Provincias incluidas: Buenos Aires, Santa Fe, Córdoba, La Pampa, Entre Ríos.

---

## Referencias

- **Scian, B. (2004).** Metodología de decadias y etapas fenológicas para trigo.
- **Iqbal et al. (2024).** Machine Learning para predicción de rendimiento de cultivos.

---

## Contacto

Proyecto desarrollado en el marco de la beca UTN-DAAD para investigación en sistemas productivos agrícolas.
