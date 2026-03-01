"""
Detrending: Elimina la tendencia tecnológica del rendimiento mediante regresión lineal por departamento.
Genera 'Rinde_Detrended' = Rinde real - Rinde tendencia (residuo).
Reentrena Random Forest sobre Rinde_Detrended con validación temporal.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent.parent
COL_RENDIMIENTO = "RENDIMIENTO - KG X HA"
COL_DETRENDED = "Rinde_Detrended"
RANDOM_STATE = 42
N_ESTIMATORS = 100
MAX_DEPTH = 20
AÑO_SPLIT_TEMPORAL = 2015


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {
        "PROVINCIA",
        "DEPARTAMENTO",
        "AÑO",
        "STATION_ID",
        COL_RENDIMIENTO,
        COL_DETRENDED,
        "Rinde_Tendencia",
    }
    return [
        c
        for c in df.columns
        if c not in exclude
        and (c.startswith("lluvia_") or c.startswith("tmax_") or c.startswith("tmin_"))
    ]


def calculate_detrended_yield(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica regresión lineal por DEPARTAMENTO para calcular tendencia temporal.
    Genera columnas: 'Rinde_Tendencia' y 'Rinde_Detrended' (residuo).
    """
    df_result = df.copy()
    df_result["Rinde_Tendencia"] = np.nan
    df_result[COL_DETRENDED] = np.nan

    for departamento in df_result["DEPARTAMENTO"].unique():
        mask = df_result["DEPARTAMENTO"] == departamento
        subset = df_result.loc[mask, ["AÑO", COL_RENDIMIENTO]].dropna()

        if len(subset) < 3:
            continue

        X_time = subset["AÑO"].values.reshape(-1, 1)
        y_rinde = subset[COL_RENDIMIENTO].values

        lr = LinearRegression()
        lr.fit(X_time, y_rinde)
        tendencia = lr.predict(X_time)

        df_result.loc[subset.index, "Rinde_Tendencia"] = tendencia
        df_result.loc[subset.index, COL_DETRENDED] = y_rinde - tendencia

    return df_result


def main() -> None:
    ruta_dataset = ROOT / "data" / "processed" / "dataset_maestro_ia.csv"
    ruta_dataset_detrended = ROOT / "data" / "processed" / "dataset_maestro_ia_detrended.csv"
    ruta_scatter_detrended = ROOT / "reports" / "figures" / "scatter_detrended_temporal.png"
    ruta_mae_año_detrended = ROOT / "reports" / "figures" / "mae_por_año_detrended.png"
    ruta_tendencia_ejemplo = ROOT / "reports" / "figures" / "ejemplo_tendencia.png"

    print("Cargando dataset_maestro_ia.csv...")
    df = pd.read_csv(ruta_dataset)

    print("Calculando tendencia por departamento (regresión lineal)...")
    df_detrended = calculate_detrended_yield(df)

    # Guardar dataset con columnas nuevas
    df_detrended.to_csv(ruta_dataset_detrended, index=False)
    print(f"Dataset con detrending guardado: {ruta_dataset_detrended}")

    # Ejemplo visual: tendencia en un departamento
    ejemplo_dep = df_detrended["DEPARTAMENTO"].value_counts().idxmax()
    subset_ej = df_detrended[df_detrended["DEPARTAMENTO"] == ejemplo_dep].dropna(
        subset=[COL_RENDIMIENTO, "Rinde_Tendencia"]
    )
    fig_ej, ax_ej = plt.subplots(figsize=(9, 4))
    ax_ej.scatter(subset_ej["AÑO"], subset_ej[COL_RENDIMIENTO], label="Rinde real", alpha=0.6)
    ax_ej.plot(
        subset_ej["AÑO"],
        subset_ej["Rinde_Tendencia"],
        color="red",
        linewidth=2,
        label="Tendencia (regresión lineal)",
    )
    ax_ej.set_xlabel("Año")
    ax_ej.set_ylabel("Rendimiento (kg/ha)")
    ax_ej.set_title(f"Ejemplo de detrending: {ejemplo_dep}")
    ax_ej.legend()
    ax_ej.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(ruta_tendencia_ejemplo, dpi=150)
    plt.close()
    print(f"Ejemplo de tendencia guardado: {ruta_tendencia_ejemplo}")

    # Preparar datos para modelado (sin NaN en features y target detrended)
    feature_cols = get_feature_columns(df_detrended)
    X = df_detrended[feature_cols]
    y = df_detrended[COL_DETRENDED]
    años = df_detrended["AÑO"]

    mask_ok = X.notna().all(axis=1) & y.notna()
    X_clean = X.loc[mask_ok].copy()
    y_clean = y.loc[mask_ok].copy()
    años_clean = años.loc[mask_ok].copy()

    # Split temporal
    mask_train = años_clean <= AÑO_SPLIT_TEMPORAL
    mask_test = años_clean > AÑO_SPLIT_TEMPORAL
    X_train = X_clean.loc[mask_train]
    y_train = y_clean.loc[mask_train]
    X_test = X_clean.loc[mask_test]
    y_test = y_clean.loc[mask_test]
    años_test = años_clean.loc[mask_test]

    print(f"\nSplit temporal (sobre Rinde_Detrended):")
    print(f"  Entrenamiento: 1990-{AÑO_SPLIT_TEMPORAL} ({len(X_train)} registros)")
    print(f"  Prueba: {AÑO_SPLIT_TEMPORAL+1}-2021 ({len(X_test)} registros)")

    # Entrenar modelo sobre residuos
    print(f"\nEntrenando Random Forest sobre Rinde_Detrended...")
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

    print("\n--- Métricas (Rinde_Detrended, validación temporal 2016-2021) ---")
    print(f"R2:   {r2:.4f}")
    print(f"RMSE: {rmse:.2f} kg/ha")
    print(f"MAE:  {mae:.2f} kg/ha")

    # MAE por año
    df_test = pd.DataFrame({"año": años_test, "real": y_test, "pred": y_pred})
    df_test["error_abs"] = (df_test["real"] - df_test["pred"]).abs()
    mae_por_año = df_test.groupby("año")["error_abs"].mean().sort_index()

    print("\n--- MAE por año (Rinde_Detrended) ---")
    for año, mae_val in mae_por_año.items():
        print(f"  {int(año)}: {mae_val:.2f} kg/ha")

    # Gráfico 1: MAE por año (detrended)
    fig1, ax1 = plt.subplots(figsize=(9, 5))
    años_list = mae_por_año.index.astype(int)
    ax1.bar(años_list, mae_por_año.values, color="#2d5a27", alpha=0.8, edgecolor="black")
    ax1.set_xlabel("Año")
    ax1.set_ylabel("MAE (kg/ha)")
    ax1.set_title("MAE por año (Rinde_Detrended, validación temporal)")
    ax1.set_xticks(años_list)
    ax1.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(ruta_mae_año_detrended, dpi=150)
    plt.close()
    print(f"\nGráfico MAE por año (detrended) guardado: {ruta_mae_año_detrended}")

    # Gráfico 2: Scatter detrended
    fig2, ax2 = plt.subplots(figsize=(7, 7))
    ax2.scatter(y_test, y_pred, alpha=0.5, edgecolors="k", linewidth=0.3, color="#7cb342")
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="y = x")
    ax2.set_xlabel("Rinde_Detrended real (kg/ha)")
    ax2.set_ylabel("Rinde_Detrended predicho (kg/ha)")
    ax2.set_title("Real vs. Predicho (Rinde_Detrended, temporal 2016-2021)")
    ax2.legend()
    ax2.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(ruta_scatter_detrended, dpi=150)
    plt.close()
    print(f"Scatter detrended guardado: {ruta_scatter_detrended}")

    # Comparación
    print("\n" + "="*70)
    print("COMPARACIÓN: Rinde absoluto vs. Rinde_Detrended (validación temporal)")
    print("="*70)
    print("\nRinde absoluto (con tendencia tecnológica):")
    print("  R2:   -0.9549")
    print("  RMSE: 1416.79 kg/ha")
    print("  MAE:  1166.95 kg/ha")
    print("\nRinde_Detrended (sin tendencia tecnológica):")
    print(f"  R2:   {r2:.4f}")
    print(f"  RMSE: {rmse:.2f} kg/ha")
    print(f"  MAE:  {mae:.2f} kg/ha")
    print("\n--- Interpretación ---")
    if r2 > -0.95:
        print("✓ El detrending mejoró significativamente la capacidad predictiva.")
        print("  El modelo ahora captura la señal climática sin el sesgo tecnológico.")
    else:
        print("⚠ El detrending no mejoró las métricas; revisar otras causas.")
    print("\nEl análisis sobre residuos permite aislar el efecto del clima en el rendimiento,")
    print("fundamental para comparar sistemas productivos entre regiones (Pampa vs. Alemania).")


if __name__ == "__main__":
    main()
