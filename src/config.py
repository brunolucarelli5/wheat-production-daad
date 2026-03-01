"""
Configuración centralizada del proyecto.
Parámetros, rutas y constantes usadas en todo el pipeline.
"""
from pathlib import Path

# Rutas base
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
REPORTS_FIGURES = REPORTS / "figures"
MODELS_DIR = ROOT / "models"

# Datos de entrada
NOAA_DIR = DATA_RAW / "noaa"
TRIGO_RAW = DATA_RAW / "DATOS TRIGO - ARGENTINA - 1990 A 2025.csv"

# Datos procesados
CLIMA_COMPLETO = DATA_PROCESSED / "clima_argentina_completo.csv"
CLIMA_PAMPEANA = DATA_PROCESSED / "clima_region_pampeana.csv"
CLIMA_FENO = DATA_PROCESSED / "clima_region_pampeana_feno.csv"
CLIMA_FEATURES = DATA_PROCESSED / "clima_region_pampeana_features.csv"
RINDE_PAMPA = DATA_PROCESSED / "rinde_trigo_pampa.csv"
MAPEO = DATA_PROCESSED / "mapeo_departamento_estacion.csv"
DATASET_MAESTRO = DATA_PROCESSED / "dataset_maestro_ia.csv"
DATASET_DETRENDED = DATA_PROCESSED / "dataset_maestro_ia_detrended.csv"

# Región Pampeana
PROVINCIAS_PAMPA = {
    "Buenos Aires",
    "Santa Fe",
    "Córdoba",
    "La Pampa",
    "Entre Ríos",
}

# Mapeo provincia → estación climática (una por provincia)
PROVINCIA_A_STATION = {
    "Buenos Aires": "AR000087692",
    "Santa Fe": "AR000087257",
    "Córdoba": "AR000087344",
    "La Pampa": "AR000087623",
    "Entre Ríos": "AR000000004",
}

# Parámetros temporales
RANGO_AÑO_MIN = 1990
RANGO_AÑO_MAX = 2021
AÑO_SPLIT_TEMPORAL = 2015

# Columnas
COL_RENDIMIENTO = "RENDIMIENTO - KG X HA"
COL_DETRENDED = "Rinde_Detrended"

# Hiperparámetros Random Forest
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 100
MAX_DEPTH = 20

# SHAP
SHAP_SAMPLE_SIZE = 500

# Etapas fenológicas (Scian, 2004)
ETAPAS_FENOLOGICAS = {
    "Siembra": {"inicio": (6, 21), "fin": (7, 10)},
    "Emergencia": {"inicio": (7, 11), "fin": (8, 30)},
    "Macollaje": {"inicio": (9, 1), "fin": (9, 30)},
    "Encañazón": {"inicio": (10, 1), "fin": (10, 20)},
    "Espigazón": {"inicio": (10, 21), "fin": (10, 31)},
    "Floración": {"inicio": (11, 1), "fin": (11, 10)},
    "Llenado de grano": {"inicio": (11, 11), "fin": (11, 30)},
}
