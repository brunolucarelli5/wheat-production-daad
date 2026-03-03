"""
Integración de variables edáficas (suelos INTA) al dataset maestro.
Preprocesa, agrega por provincia, calcula tendencia de rinde y une con dataset_maestro_ia.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


ROOT = Path(__file__).resolve().parent.parent.parent

PROVINCIAS_PAMPA = {
    "BUENOS AIRES",
    "CORDOBA",
    "SANTA FE",
    "LA PAMPA",
    "ENTRE RIOS",
}

COLUMNAS_SUELO = ["provincia", "ind_prod", "text_sups1", "drenaje_s1", "profund_s1"]


def preprocess_soil_data(df_suelo: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra Región Pampeana y selecciona columnas clave de suelo.
    """
    df = df_suelo.copy()
    df["provincia"] = df["provincia"].str.upper().str.strip()
    df_pampa = df[df["provincia"].isin(PROVINCIAS_PAMPA)][COLUMNAS_SUELO].copy()
    # Convertir ind_prod y profund_s1 a numérico
    df_pampa["ind_prod"] = pd.to_numeric(df_pampa["ind_prod"], errors="coerce")
    df_pampa["profund_s1"] = pd.to_numeric(df_pampa["profund_s1"], errors="coerce")
    return df_pampa


def aggregate_soil_by_province(df_suelo: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega variables de suelo por provincia:
    - ind_prod y profund_s1: promedio
    - text_sups1 y drenaje_s1: moda (valor más frecuente) + label encoding
    """
    agg_dict = {
        "ind_prod": "mean",
        "profund_s1": "mean",
        "text_sups1": lambda x: x.mode()[0] if len(x.mode()) > 0 else np.nan,
        "drenaje_s1": lambda x: x.mode()[0] if len(x.mode()) > 0 else np.nan,
    }
    df_agg = df_suelo.groupby("provincia").agg(agg_dict).reset_index()
    # Label encoding para categóricas
    le_textura = LabelEncoder()
    le_drenaje = LabelEncoder()
    df_agg["suelo_textura_encoded"] = le_textura.fit_transform(
        df_agg["text_sups1"].fillna("Desconocido")
    )
    df_agg["suelo_drenaje_encoded"] = le_drenaje.fit_transform(
        df_agg["drenaje_s1"].fillna("Desconocido")
    )
    # Renombrar columnas para claridad
    df_agg = df_agg.rename(
        columns={
            "ind_prod": "suelo_ind_prod",
            "profund_s1": "suelo_profundidad",
            "text_sups1": "suelo_textura",
            "drenaje_s1": "suelo_drenaje",
        }
    )
    return df_agg


def merge_soil_with_dataset(
    df_maestro: pd.DataFrame, df_suelo_agg: pd.DataFrame
) -> pd.DataFrame:
    """
    Une dataset maestro con variables de suelo por PROVINCIA.
    """
    df_maestro["PROVINCIA_UPPER"] = df_maestro["PROVINCIA"].str.upper().str.strip()
    df_merged = df_maestro.merge(
        df_suelo_agg, left_on="PROVINCIA_UPPER", right_on="provincia", how="left"
    )
    df_merged = df_merged.drop(columns=["PROVINCIA_UPPER", "provincia"])
    return df_merged


def add_trend_columns(df_maestro: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula tendencia lineal de rendimiento por (PROVINCIA, DEPARTAMENTO)
    y agrega columnas:
    - Rinde_Tendencia: valor esperado según la tendencia
    - Rinde_Detrended: residuo (real - tendencia)
    """

    def _fit_trend(group: pd.DataFrame) -> pd.DataFrame:
        años = group["AÑO"].astype(float).values
        rinde = group["RENDIMIENTO - KG X HA"].astype(float).values
        if len(group) < 2:
            tendencia = np.full_like(rinde, rinde.mean(), dtype=float)
        else:
            coef = np.polyfit(años, rinde, deg=1)
            tendencia = np.polyval(coef, años)
        group = group.copy()
        group["Rinde_Tendencia"] = tendencia
        group["Rinde_Detrended"] = group["RENDIMIENTO - KG X HA"] - tendencia
        return group

    return (
        df_maestro.groupby(["PROVINCIA", "DEPARTAMENTO"], group_keys=False)
        .apply(_fit_trend)
        .reset_index(drop=True)
    )


def main() -> None:
    raw = ROOT / "data" / "raw"
    processed = ROOT / "data" / "processed"
    ruta_suelo = raw / "carta-suelos-argentina.csv"
    ruta_maestro = processed / "dataset_maestro_ia.csv"
    ruta_salida = processed / "dataset_final.csv"

    print("Cargando carta-suelos-argentina.csv...")
    df_suelo_raw = pd.read_csv(ruta_suelo)
    print(f"  Registros totales: {len(df_suelo_raw)}")

    print("\nPreprocesando datos de suelo (Región Pampeana)...")
    df_suelo_pampa = preprocess_soil_data(df_suelo_raw)
    print(f"  Registros Región Pampeana: {len(df_suelo_pampa)}")

    print("\nAgregando por provincia (promedio ind_prod, moda textura/drenaje)...")
    df_suelo_agg = aggregate_soil_by_province(df_suelo_pampa)
    print(f"  Provincias con datos de suelo: {len(df_suelo_agg)}")
    print("\nVariables de suelo por provincia:")
    print(df_suelo_agg.to_string(index=False))

    print(f"\nCargando {ruta_maestro}...")
    df_maestro = pd.read_csv(ruta_maestro)
    print(f"  Registros antes del merge: {len(df_maestro)}")

    print("\nMergeando dataset maestro con variables de suelo...")
    df_merged = merge_soil_with_dataset(df_maestro, df_suelo_agg)
    print(f"  Registros después del merge: {len(df_merged)}")

    print("\nCalculando tendencia de rendimiento y rinde ajustado (Rinde_Detrended)...")
    df_final = add_trend_columns(df_merged)

    # Verificar NaN en variables de suelo
    soil_cols = [
        "suelo_ind_prod",
        "suelo_profundidad",
        "suelo_textura_encoded",
        "suelo_drenaje_encoded",
    ]
    print("\nPorcentaje de NaN en variables de suelo:")
    for col in soil_cols:
        if col in df_final.columns:
            pct = (df_final[col].isna().sum() / len(df_final)) * 100
            print(f"  {col}: {pct:.2f}%")

    processed.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(ruta_salida, index=False)
    print(f"\nDataset con variables de suelo guardado: {ruta_salida}")
    print(f"Columnas totales: {len(df_final.columns)}")


if __name__ == "__main__":
    main()
