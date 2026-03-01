"""
Funciones de utilidad compartidas en todo el proyecto.
"""
import pandas as pd


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Retorna las columnas de features climáticos (lluvia_*, tmax_*, tmin_*).
    Excluye columnas de identificación y target.
    """
    exclude = {
        "PROVINCIA",
        "DEPARTAMENTO",
        "AÑO",
        "STATION_ID",
        "STATION",
        "NAME",
        "LATITUDE",
        "LONGITUDE",
        "ELEVATION",
        "RENDIMIENTO - KG X HA",
        "Rinde_Detrended",
        "Rinde_Tendencia",
    }
    return [
        c
        for c in df.columns
        if c not in exclude
        and (c.startswith("lluvia_") or c.startswith("tmax_") or c.startswith("tmin_"))
    ]


def normalize_stage_name(stage_name: str) -> str:
    """
    Normaliza el nombre de la etapa fenológica para usar en nombres de columnas:
    - pasa a minúsculas
    - reemplaza espacios por guiones bajos
    - remueve acentos simples
    """
    if not isinstance(stage_name, str):
        return "desconocido"
    nombre = stage_name.lower()
    reemplazos = {
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ú": "u",
        "ñ": "n",
    }
    for original, reemplazo in reemplazos.items():
        nombre = nombre.replace(original, reemplazo)
    nombre = nombre.replace(" ", "_")
    return nombre
