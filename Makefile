.PHONY: help install clean data features train validate explain reports all

PYTHON := ./venv/bin/python
PIP := ./venv/bin/pip

help:
	@echo "Comandos disponibles:"
	@echo "  make install    - Instalar dependencias en venv"
	@echo "  make data       - Procesar datos crudos (NOAA + etapas fenológicas)"
	@echo "  make features   - Generar features climáticos anuales"
	@echo "  make train      - Entrenar modelo Random Forest (clima)"
	@echo "  make validate   - Validación temporal + detrending (clima)"
	@echo "  make soil       - Integrar suelo y entrenar modelo clima+suelo"
	@echo "  make explain    - Análisis XAI (importancia + SHAP, clima)"
	@echo "  make reports    - Generar informes HTML + Markdown"
	@echo "  make all        - Ejecutar pipeline completo"
	@echo "  make clean      - Limpiar archivos generados"

install:
	$(PIP) install -r requirements.txt

data:
	$(PYTHON) src/features/build_phenological_features.py

features:
	$(PYTHON) src/features/aggregate_climate.py
	$(PYTHON) scripts/preparar_insumos_merge.py
	$(PYTHON) scripts/merge_rinde_clima.py

train:
	$(PYTHON) src/models/train_model.py

validate:
	$(PYTHON) src/models/validate_temporal.py
	$(PYTHON) src/models/train_detrended.py

soil:
	$(PYTHON) src/features/integrate_soil.py
	$(PYTHON) src/models/train_with_soil.py

explain:
	$(PYTHON) src/models/explain_model.py

reports:
	$(PYTHON) src/visualization/generate_html_report.py
	$(PYTHON) src/visualization/generate_markdown_report.py

all:
	$(PYTHON) scripts/ejecutar_pipeline_completo.py

clean:
	rm -f data/processed/clima_region_pampeana_feno.csv
	rm -f data/processed/clima_region_pampeana_features.csv
	rm -f data/processed/dataset_maestro_ia*.csv
	rm -f data/processed/rinde_trigo_pampa.csv
	rm -f data/processed/mapeo_departamento_estacion.csv
	rm -f reports/figures/*.png
	rm -f reports/*.html reports/*.md
	@echo "Archivos generados eliminados"
