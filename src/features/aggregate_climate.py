from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent


def normalize_stage_name(stage_name: str) -> str:
    if not isinstance(stage_name, str):
        return "desconocido"
    nombre = stage_name.lower()
    reemplazos = {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ñ": "n"}
    for original, reemplazo in reemplazos.items():
        nombre = nombre.replace(original, reemplazo)
    nombre = nombre.replace(" ", "_")
    return nombre


def build_annual_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    A partir de un DataFrame diario con columnas:
    'STATION', 'DATE', 'PRCP', 'TMAX', 'TMIN', 'ETAPA_FENOLOGICA'
    genera predictores anuales por estación:
    - suma de PRCP por etapa
    - promedio de TMAX y TMIN por etapa
    - pivot wide: una fila por STATION-AÑO y columnas como:
      'lluvia_emergencia', 'tmax_floracion', 'tmin_macollaje', etc.
    Manejo de NaN:
    - agregaciones usan sum/mean (saltan NaN por defecto)
    - imputación posterior: por estación, se rellenan NaN con la media de la estación
    - filas con todos los predictores NaN se eliminan
    """
    df_trabajo = df.copy()
    df_trabajo["DATE"] = pd.to_datetime(df_trabajo["DATE"])
    df_trabajo["AÑO"] = df_trabajo["DATE"].dt.year
    # Nos quedamos solo con registros dentro de etapas fenológicas definidas
    mask_etapa_valida = df_trabajo["ETAPA_FENOLOGICA"] != "Fuera de ciclo"
    df_trabajo = df_trabajo.loc[mask_etapa_valida].copy()
    # Agregación por estación, año y etapa
    agrupado = (
        df_trabajo.groupby(["STATION", "AÑO", "ETAPA_FENOLOGICA"], observed=True)[
            ["PRCP", "TMAX", "TMIN"]
        ]
        .agg({"PRCP": "sum", "TMAX": "mean", "TMIN": "mean"})
        .rename(columns={"PRCP": "PRCP_sum", "TMAX": "TMAX_mean", "TMIN": "TMIN_mean"})
    )
    # Pasamos a formato ancho con columnas multi-índice (variable, etapa)
    tabla_pivot = agrupado.unstack("ETAPA_FENOLOGICA")
    # Aplanamos el multi-índice de columnas a nombres tipo 'lluvia_emergencia'
    columnas_nuevas = []
    for variable, etapa in tabla_pivot.columns:
        if variable == "PRCP_sum":
            prefijo = "lluvia"
        elif variable == "TMAX_mean":
            prefijo = "tmax"
        elif variable == "TMIN_mean":
            prefijo = "tmin"
        else:
            prefijo = variable.lower()
        etapa_normalizada = normalize_stage_name(etapa)
        nombre_columna = f"{prefijo}_{etapa_normalizada}"
        columnas_nuevas.append(nombre_columna)
    tabla_pivot.columns = columnas_nuevas
    tabla_pivot = tabla_pivot.reset_index()  # STATION, AÑO como columnas
    # Imputación de NaN por media de la estación
    columnas_predictores = [
        col for col in tabla_pivot.columns if col not in ["STATION", "AÑO"]
    ]
    # Convertimos a numérico por seguridad
    tabla_pivot[columnas_predictores] = tabla_pivot[columnas_predictores].apply(
        pd.to_numeric, errors="coerce"
    )
    # Para cada estación, rellenar NaN con la media de la estación en esa columna
    tabla_pivot[columnas_predictores] = (
        tabla_pivot.groupby("STATION")[columnas_predictores]
        .transform(lambda x: x.fillna(x.mean()))
        .astype(float)
    )
    # Eliminar filas donde todos los predictores sigan siendo NaN
    mask_todos_nan = tabla_pivot[columnas_predictores].isna().all(axis=1)
    tabla_limpia = tabla_pivot.loc[~mask_todos_nan].reset_index(drop=True)
    return tabla_limpia


def main() -> None:
    ruta_entrada = ROOT / "data" / "processed" / "clima_region_pampeana_feno.csv"
    ruta_salida = ROOT / "data" / "processed" / "clima_region_pampeana_features.csv"
    print(f"Leyendo datos diarios con etapas desde: {ruta_entrada}")
    df_diario = pd.read_csv(ruta_entrada)
    print(f"Registros diarios leídos: {len(df_diario)}")
    df_anual = build_annual_features(df_diario)
    print(f"Registros anuales por estación generados: {len(df_anual)}")
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    df_anual.to_csv(ruta_salida, index=False)
    print(f"Archivo de predictores anuales guardado en: {ruta_salida}")


if __name__ == "__main__":
    main()

