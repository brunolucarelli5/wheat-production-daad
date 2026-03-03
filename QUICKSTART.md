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

### 4. Integrar suelo y entrenar modelo clima + suelo

```bash
python src/features/integrate_soil.py
python src/models/train_with_soil.py
```

### 5. Generar informes

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
├── dataset_maestro_ia.csv              # Dataset rinde + clima
└── dataset_final.csv                   # Dataset rinde + clima + suelo

reports/
├── figures/
│   ├── scatter.png                     # Real vs. predicho (kg/ha, clima + suelo)
│   ├── scatter_residuals.png           # Rinde ajustado: real vs. predicho
│   ├── importance.png                  # Importancia de variables (clima + suelo)
│   └── shap.png                        # SHAP summary (clima + suelo)
├── informe_trigo.html
└── INFORME_TRIGO.md
```

## Troubleshooting

### Error: "No such file or directory"

Asegúrate de ejecutar los scripts desde la raíz del proyecto:

```bash
cd /home/blucarelli/dev/personal/daad
python src/models/train_with_soil.py
```

### Error: "ModuleNotFoundError"

Los scripts en `src/` son independientes y no requieren imports cruzados. Ejecútalos directamente con Python.

### Limpiar archivos generados

```bash
make clean
```
