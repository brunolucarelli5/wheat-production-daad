"""
Paso 5: Análisis de Importancia y Explicabilidad (XAI).
Identificación de qué variables climáticas por etapa fenológica impactan más en el rinde.
Este análisis es fundamental para la comparación de sistemas productivos entre
la Región Pampeana (Argentina) y Alemania (requerimiento beca UTN-DAAD).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent.parent
COL_RENDIMIENTO = "RENDIMIENTO - KG X HA"
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 100
MAX_DEPTH = 20
SHAP_SAMPLE_SIZE = 500


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {"PROVINCIA", "DEPARTAMENTO", "AÑO", "STATION_ID", COL_RENDIMIENTO}
    return [
        c
        for c in df.columns
        if c not in exclude
        and (c.startswith("lluvia_") or c.startswith("tmax_") or c.startswith("tmin_"))
    ]


def main() -> None:
    ruta_dataset = ROOT / "data" / "processed" / "dataset_maestro_ia.csv"
    ruta_importancia = ROOT / "reports" / "figures" / "importancia_variables.png"
    ruta_shap = ROOT / "reports" / "figures" / "shap_summary_plot.png"
    out_dir = ROOT / "reports" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Cargando dataset_maestro_ia.csv...")
    df = pd.read_csv(ruta_dataset)
    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df[COL_RENDIMIENTO]
    mask_ok = X.notna().all(axis=1) & y.notna()
    X = X.loc[mask_ok].copy()
    y = y.loc[mask_ok].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print("Entrenando Random Forest (misma configuración que Paso 4)...")
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    # 1) Gráfico de barras horizontal: feature_importances_ ordenado
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    names_ordered = [feature_cols[i] for i in order]
    values_ordered = importances[order]

    fig, ax = plt.subplots(figsize=(8, 10))
    ax.barh(range(len(names_ordered)), values_ordered, align="center")
    ax.set_yticks(range(len(names_ordered)))
    ax.set_yticklabels(names_ordered, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Importancia (Gini)")
    ax.set_title("Importancia de variables climáticas en el rendimiento de trigo (Random Forest)")
    plt.tight_layout()
    plt.savefig(ruta_importancia, dpi=150)
    plt.close()
    print(f"Gráfico de importancia guardado: {ruta_importancia}")

    # 2) SHAP Summary Plot (muestra para reducir tiempo)
    X_sample = X_train.sample(n=min(SHAP_SAMPLE_SIZE, len(X_train)), random_state=RANDOM_STATE)
    print("Calculando valores SHAP (muestra de entrenamiento)...")
    explainer = shap.TreeExplainer(model, X_sample)
    shap_values = explainer.shap_values(X_sample)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_cols, show=False)
    plt.title("SHAP Summary: impacto de variables climáticas en el rendimiento (kg/ha)")
    plt.tight_layout()
    plt.savefig(ruta_shap, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP summary plot guardado: {ruta_shap}")

    print("\n--- Interpretación breve (SHAP) ---")
    print("En el Summary Plot: cada punto es un registro. Eje X = efecto SHAP sobre el rendimiento.")
    print("Valores SHAP > 0 aumentan el rinde predicho; < 0 lo reducen.")
    print("Variables como tmax_floracion o lluvia por etapa muestran cómo la temperatura y la")
    print("precipitación en momentos fenológicos clave afectan positiva o negativamente el rinde.")
    print("\nEste análisis de importancia es fundamental para la comparación de sistemas productivos")
    print("entre la Región Pampeana y Alemania (beca UTN-DAAD).")


if __name__ == "__main__":
    main()
