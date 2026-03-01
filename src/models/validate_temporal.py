"""
Validación Temporal (Time-Series Split) para Random Forest.
Split: 1990-2015 (train) vs 2016-2021 (test) para simular predicción de años futuros.
Análisis de error por año para identificar años problemáticos (ej. sequías).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent.parent
COL_RENDIMIENTO = "RENDIMIENTO - KG X HA"
RANDOM_STATE = 42
N_ESTIMATORS = 100
MAX_DEPTH = 20
AÑO_SPLIT_TEMPORAL = 2015


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
    ruta_mae_por_año = ROOT / "reports" / "figures" / "mae_por_año_temporal.png"
    ruta_scatter_temporal = ROOT / "reports" / "figures" / "scatter_temporal.png"

    print("Cargando dataset_maestro_ia.csv...")
    df = pd.read_csv(ruta_dataset)
    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df[COL_RENDIMIENTO]
    años = df["AÑO"]

    mask_ok = X.notna().all(axis=1) & y.notna()
    X_clean = X.loc[mask_ok].copy()
    y_clean = y.loc[mask_ok].copy()
    años_clean = años.loc[mask_ok].copy()

    # Split temporal: 1990-2015 (train) vs 2016-2021 (test)
    mask_train = años_clean <= AÑO_SPLIT_TEMPORAL
    mask_test = años_clean > AÑO_SPLIT_TEMPORAL

    X_train = X_clean.loc[mask_train]
    y_train = y_clean.loc[mask_train]
    X_test = X_clean.loc[mask_test]
    y_test = y_clean.loc[mask_test]
    años_test = años_clean.loc[mask_test]

    print(f"\nSplit temporal:")
    print(f"  Entrenamiento: 1990-{AÑO_SPLIT_TEMPORAL} ({len(X_train)} registros)")
    print(f"  Prueba: {AÑO_SPLIT_TEMPORAL+1}-2021 ({len(X_test)} registros)")

    # Entrenar modelo
    print(f"\nEntrenando Random Forest (n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH})...")
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    # Evaluar en test
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print("\n--- Métricas (validación temporal 2016-2021) ---")
    print(f"R2:   {r2:.4f}")
    print(f"RMSE: {rmse:.2f} kg/ha")
    print(f"MAE:  {mae:.2f} kg/ha")

    # Análisis de error por año
    df_test = pd.DataFrame({"año": años_test, "real": y_test, "pred": y_pred})
    df_test["error_abs"] = (df_test["real"] - df_test["pred"]).abs()
    mae_por_año = df_test.groupby("año")["error_abs"].mean().sort_index()

    print("\n--- MAE por año (conjunto de prueba) ---")
    for año, mae_val in mae_por_año.items():
        print(f"  {int(año)}: {mae_val:.2f} kg/ha")

    # Gráfico 1: MAE por año
    fig1, ax1 = plt.subplots(figsize=(9, 5))
    años_list = mae_por_año.index.astype(int)
    ax1.bar(años_list, mae_por_año.values, color="#7cb342", alpha=0.8, edgecolor="black")
    ax1.set_xlabel("Año")
    ax1.set_ylabel("MAE (kg/ha)")
    ax1.set_title("Error Absoluto Medio por año (validación temporal 2016-2021)")
    ax1.set_xticks(años_list)
    ax1.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(ruta_mae_por_año, dpi=150)
    plt.close()
    print(f"\nGráfico MAE por año guardado: {ruta_mae_por_año}")

    # Gráfico 2: Scatter real vs predicho (temporal)
    fig2, ax2 = plt.subplots(figsize=(7, 7))
    ax2.scatter(y_test, y_pred, alpha=0.5, edgecolors="k", linewidth=0.3, color="#2d5a27")
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="y = x")
    ax2.set_xlabel("Rendimiento real (kg/ha)")
    ax2.set_ylabel("Rendimiento predicho (kg/ha)")
    ax2.set_title("Real vs. Predicho (validación temporal 2016-2021)")
    ax2.legend()
    ax2.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(ruta_scatter_temporal, dpi=150)
    plt.close()
    print(f"Scatter temporal guardado: {ruta_scatter_temporal}")

    # Comparación con split aleatorio (valores del Paso 4)
    print("\n" + "="*70)
    print("COMPARACIÓN: Split aleatorio (Paso 4) vs. Validación temporal")
    print("="*70)
    print("\nSplit aleatorio (80/20, random_state=42):")
    print("  R2:   0.5608")
    print("  RMSE: 696.78 kg/ha")
    print("  MAE:  511.16 kg/ha")
    print("\nValidación temporal (train: 1990-2015, test: 2016-2021):")
    print(f"  R2:   {r2:.4f}")
    print(f"  RMSE: {rmse:.2f} kg/ha")
    print(f"  MAE:  {mae:.2f} kg/ha")
    print("\n--- Interpretación ---")
    if r2 >= 0.50:
        print("✓ El modelo mantiene capacidad predictiva en años nunca vistos.")
    else:
        print("⚠ El modelo pierde capacidad predictiva en años futuros (R2 < 0.50).")
    print("  La validación temporal es más realista para series temporales.")
    print("  Años con MAE alto pueden corresponder a eventos extremos (sequías, etc.).")


if __name__ == "__main__":
    main()
