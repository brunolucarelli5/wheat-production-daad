"""
Paso 4: Entrenamiento y Evaluación (Iqbal et al., 2024).
Modelo Random Forest para predicción de rendimiento de trigo.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent.parent
COL_RENDIMIENTO = "RENDIMIENTO - KG X HA"
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 100
MAX_DEPTH = 20


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
    ruta_figura = ROOT / "reports" / "figures" / "scatter_real_vs_predicho.png"

    print("Cargando dataset_maestro_ia.csv...")
    df = pd.read_csv(ruta_dataset)

    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df[COL_RENDIMIENTO]

    # Eliminar filas con NaN en features o target para el entrenamiento
    mask_ok = X.notna().all(axis=1) & y.notna()
    X = X.loc[mask_ok].copy()
    y = y.loc[mask_ok].copy()
    print(f"Filas usadas (sin NaN en X o y): {len(X)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    print(f"\nEntrenamiento: n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH}")
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print("\n--- Métricas (conjunto de prueba) ---")
    print(f"R2:   {r2:.4f}")
    print(f"RMSE: {rmse:.2f} kg/ha")
    print(f"MAE:  {mae:.2f} kg/ha")

    # Gráfico dispersión: real vs predicho
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, alpha=0.5, edgecolors="k", linewidth=0.3)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="y = x")
    ax.set_xlabel("Rendimiento real (kg/ha)")
    ax.set_ylabel("Rendimiento predicho (kg/ha)")
    ax.set_title("Valores reales vs. predichos (Random Forest)")
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    ruta_figura.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(ruta_figura, dpi=150)
    plt.close()
    print(f"\nGráfico guardado: {ruta_figura}")


if __name__ == "__main__":
    main()
