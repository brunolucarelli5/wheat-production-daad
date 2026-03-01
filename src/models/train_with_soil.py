"""
Entrenamiento Random Forest con variables climáticas + edáficas.
Validación temporal (1990-2015 → 2016-2021) sobre Rinde_Detrended.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parent.parent.parent
COL_TARGET = "Rinde_Detrended"
RANDOM_STATE = 42
N_ESTIMATORS = 100
MAX_DEPTH = 20
AÑO_SPLIT = 2015
SHAP_SAMPLE_SIZE = 500


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Retorna columnas de features: climáticas + edáficas.
    Excluye identificadores y targets.
    """
    exclude = {
        "PROVINCIA",
        "DEPARTAMENTO",
        "AÑO",
        "STATION_ID",
        "RENDIMIENTO - KG X HA",
        "Rinde_Detrended",
        "Rinde_Tendencia",
        "suelo_textura",
        "suelo_drenaje",
    }
    return [
        c
        for c in df.columns
        if c not in exclude
        and (
            c.startswith("lluvia_")
            or c.startswith("tmax_")
            or c.startswith("tmin_")
            or c.startswith("suelo_")
        )
    ]


def main() -> None:
    processed = ROOT / "data" / "processed"
    figures = ROOT / "reports" / "figures"
    ruta_dataset = processed / "dataset_maestro_ia_con_suelo.csv"
    ruta_scatter = figures / "scatter_con_suelo.png"
    ruta_importancia = figures / "importancia_con_suelo.png"
    ruta_shap = figures / "shap_con_suelo.png"
    ruta_comparacion = figures / "comparacion_metricas.png"

    print("Cargando dataset con variables de suelo...")
    df = pd.read_csv(ruta_dataset)
    feature_cols = get_feature_columns(df)
    print(f"Features totales: {len(feature_cols)}")
    clima_cols = [c for c in feature_cols if not c.startswith("suelo_")]
    suelo_cols = [c for c in feature_cols if c.startswith("suelo_")]
    print(f"  Climáticas: {len(clima_cols)}")
    print(f"  Edáficas: {len(suelo_cols)} → {suelo_cols}")

    X = df[feature_cols]
    y = df[COL_TARGET]
    años = df["AÑO"]

    # Eliminar filas con NaN en features o target
    mask_ok = X.notna().all(axis=1) & y.notna()
    X_clean = X.loc[mask_ok].copy()
    y_clean = y.loc[mask_ok].copy()
    años_clean = años.loc[mask_ok].copy()
    print(f"\nRegistros válidos (sin NaN): {len(X_clean)}")

    # Split temporal
    mask_train = años_clean <= AÑO_SPLIT
    mask_test = años_clean > AÑO_SPLIT
    X_train = X_clean.loc[mask_train]
    y_train = y_clean.loc[mask_train]
    X_test = X_clean.loc[mask_test]
    y_test = y_clean.loc[mask_test]
    años_test = años_clean.loc[mask_test]

    print(f"\nSplit temporal:")
    print(f"  Entrenamiento: 1990-{AÑO_SPLIT} ({len(X_train)} registros)")
    print(f"  Prueba: {AÑO_SPLIT+1}-2021 ({len(X_test)} registros)")

    # Entrenar modelo
    print(f"\nEntrenando Random Forest (clima + suelo)...")
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)

    # Evaluar
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print("\n" + "="*70)
    print("MÉTRICAS: Modelo con clima + suelo (validación temporal 2016-2021)")
    print("="*70)
    print(f"R2:   {r2:.4f}")
    print(f"RMSE: {rmse:.2f} kg/ha")
    print(f"MAE:  {mae:.2f} kg/ha")

    # Comparación con modelo solo clima
    print("\n" + "="*70)
    print("COMPARACIÓN: Solo clima vs. Clima + suelo")
    print("="*70)
    print("\nSolo clima (Rinde_Detrended, temporal):")
    print("  R2:   -0.0378")
    print("  RMSE: 613.13 kg/ha")
    print("  MAE:  496.49 kg/ha")
    print("\nClima + suelo (Rinde_Detrended, temporal):")
    print(f"  R2:   {r2:.4f}")
    print(f"  RMSE: {rmse:.2f} kg/ha")
    print(f"  MAE:  {mae:.2f} kg/ha")

    mejora_r2 = r2 - (-0.0378)
    mejora_mae = 496.49 - mae
    print(f"\nMejora R2: {mejora_r2:+.4f}")
    print(f"Mejora MAE: {mejora_mae:+.2f} kg/ha")

    # Scatter plot
    figures.mkdir(parents=True, exist_ok=True)
    fig1, ax1 = plt.subplots(figsize=(7, 7))
    ax1.scatter(y_test, y_pred, alpha=0.5, edgecolors="k", linewidth=0.3, color="#2d5a27")
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="y = x")
    ax1.set_xlabel("Rinde_Detrended real (kg/ha)")
    ax1.set_ylabel("Rinde_Detrended predicho (kg/ha)")
    ax1.set_title("Real vs. Predicho (Clima + Suelo, temporal 2016-2021)")
    ax1.legend()
    ax1.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(ruta_scatter, dpi=150)
    plt.close()
    print(f"\nScatter guardado: {ruta_scatter}")

    # Importancia de variables
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    names_ordered = [feature_cols[i] for i in order]
    values_ordered = importances[order]

    print("\n--- Top 15 variables más importantes ---")
    for i in range(min(15, len(names_ordered))):
        print(f"  {i+1:2}. {names_ordered[i]:30} {values_ordered[i]:.4f}")

    # Identificar posición de variables de suelo
    print("\n--- Ranking de variables edáficas ---")
    for soil_col in suelo_cols:
        if soil_col in names_ordered:
            rank = names_ordered.index(soil_col) + 1
            imp = importances[feature_cols.index(soil_col)]
            print(f"  {soil_col:30} Ranking: {rank:2} | Importancia: {imp:.4f}")

    fig2, ax2 = plt.subplots(figsize=(8, 10))
    ax2.barh(range(len(names_ordered)), values_ordered, align="center")
    ax2.set_yticks(range(len(names_ordered)))
    ax2.set_yticklabels(names_ordered, fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel("Importancia (Gini)")
    ax2.set_title("Importancia de variables (Clima + Suelo)")
    plt.tight_layout()
    plt.savefig(ruta_importancia, dpi=150)
    plt.close()
    print(f"\nGráfico de importancia guardado: {ruta_importancia}")

    # SHAP
    X_sample = X_train.sample(n=min(SHAP_SAMPLE_SIZE, len(X_train)), random_state=RANDOM_STATE)
    print(f"\nCalculando valores SHAP (muestra de {len(X_sample)} registros)...")
    explainer = shap.TreeExplainer(model, X_sample)
    shap_values = explainer.shap_values(X_sample)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_cols, show=False)
    plt.title("SHAP: Clima + Suelo → Rinde_Detrended")
    plt.tight_layout()
    plt.savefig(ruta_shap, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP summary guardado: {ruta_shap}")

    # Gráfico comparación de métricas
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    categorias = ["Solo clima", "Clima + suelo"]
    r2_vals = [-0.0378, r2]
    mae_vals = [496.49, mae]
    x = np.arange(len(categorias))
    width = 0.35
    ax3.bar(x - width/2, r2_vals, width, label="R²", color="#7cb342", alpha=0.8)
    ax3_twin = ax3.twinx()
    ax3_twin.bar(x + width/2, mae_vals, width, label="MAE (kg/ha)", color="#2d5a27", alpha=0.8)
    ax3.set_ylabel("R²")
    ax3_twin.set_ylabel("MAE (kg/ha)")
    ax3.set_xticks(x)
    ax3.set_xticklabels(categorias)
    ax3.set_title("Comparación: Solo clima vs. Clima + suelo")
    ax3.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax3.legend(loc="upper left")
    ax3_twin.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(ruta_comparacion, dpi=150)
    plt.close()
    print(f"Gráfico de comparación guardado: {ruta_comparacion}")

    print("\n--- Análisis crítico ---")
    if mejora_r2 > 0.05:
        print("✓ Las variables de suelo mejoraron significativamente el R².")
    elif mejora_r2 > 0:
        print("→ Las variables de suelo mejoraron ligeramente el R².")
    else:
        print("✗ Las variables de suelo no mejoraron el R².")

    if mejora_mae > 20:
        print("✓ Las variables de suelo redujeron significativamente el MAE.")
    elif mejora_mae > 0:
        print("→ Las variables de suelo redujeron ligeramente el MAE.")
    else:
        print("✗ Las variables de suelo no redujeron el MAE.")

    # Verificar si suelo_ind_prod está en top 10
    top_10 = names_ordered[:10]
    if "suelo_ind_prod" in top_10:
        rank = names_ordered.index("suelo_ind_prod") + 1
        print(f"\n✓ suelo_ind_prod está en el TOP 10 (ranking {rank}).")
    else:
        print("\n→ suelo_ind_prod NO está en el TOP 10 de importancia.")
        print("  Las variables climáticas (ej. lluvia_floracion) siguen siendo más determinantes.")


if __name__ == "__main__":
    main()
