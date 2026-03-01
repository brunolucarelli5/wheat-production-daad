# Guía Rápida

## Instalación

```bash
# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

## Ejecución Rápida

### Opción 1: Pipeline completo (un solo comando)

```bash
python scripts/ejecutar_pipeline_completo.py
```

Esto ejecuta todos los pasos y genera:
- Datos procesados en `data/processed/`
- Figuras en `reports/figures/`
- Informes en `reports/`

### Opción 2: Usando Makefile

```bash
# Ver comandos disponibles
make help

# Ejecutar pipeline completo
make all

# Ejecutar pasos individuales
make data      # Procesar datos
make features  # Generar features
make train     # Entrenar modelo
make validate  # Validación temporal
make explain   # Análisis XAI
make reports   # Generar informes
```

## Pasos Individuales

### 1. Mapear etapas fenológicas

```bash
python src/features/build_phenological_features.py
```

### 2. Agregar predictores climáticos

```bash
python src/features/aggregate_climate.py
```

### 3. Preparar y mergear datos

```bash
python scripts/preparar_insumos_merge.py
python scripts/merge_rinde_clima.py
```

### 4. Entrenar Random Forest

```bash
python src/models/train_model.py
```

### 5. Validación temporal

```bash
python src/models/validate_temporal.py
```

### 6. Detrending y reentrenamiento

```bash
python src/models/train_detrended.py
```

### 7. Análisis XAI

```bash
python src/models/explain_model.py
```

### 8. Generar informes

```bash
python src/visualization/generate_html_report.py
python src/visualization/generate_markdown_report.py
```

## Ver Resultados

### Informe HTML (visual)

```bash
firefox reports/informe_trigo.html
# o
xdg-open reports/informe_trigo.html
```

### Informe Markdown (texto)

```bash
cat reports/INFORME_TRIGO.md
# o abrirlo en cualquier editor
```

### Figuras

```bash
eog reports/figures/*.png
# o abrirlas individualmente
```

## Estructura de Archivos Generados

```
data/processed/
├── clima_region_pampeana_feno.csv      # Datos diarios con etapas
├── clima_region_pampeana_features.csv  # Predictores anuales
├── dataset_maestro_ia.csv              # Dataset para modelado
└── dataset_maestro_ia_detrended.csv    # Dataset con detrending

reports/
├── figures/
│   ├── scatter_real_vs_predicho.png
│   ├── importancia_variables.png
│   ├── shap_summary_plot.png
│   ├── mae_por_año_temporal.png
│   ├── scatter_detrended_temporal.png
│   └── ejemplo_tendencia.png
├── informe_trigo.html
└── INFORME_TRIGO.md
```

## Métricas Principales

### Split Aleatorio (80/20)
- R² = 0.5608
- RMSE = 696.78 kg/ha
- MAE = 511.16 kg/ha

### Validación Temporal (2016-2021)
**Rinde absoluto:**
- R² = -0.9549

**Rinde_Detrended (sin tendencia tecnológica):**
- R² = -0.0378
- RMSE = 613.13 kg/ha
- MAE = 496.49 kg/ha

## Troubleshooting

### Error: "No such file or directory"

Asegúrate de ejecutar los scripts desde la raíz del proyecto:

```bash
cd /home/blucarelli/dev/personal/daad
python src/models/train_model.py
```

### Error: "ModuleNotFoundError"

Los scripts en `src/` son independientes y no requieren imports cruzados. Ejecutalos directamente con Python.

### Limpiar archivos generados

```bash
make clean
```
