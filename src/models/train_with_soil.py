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


def stage_to_spanish(stage_key: str) -> str:
    mapping = {
        "siembra": "Siembra",
        "emergencia": "Emergencia",
        "macollaje": "Macollaje",
        "encanazon": "Encañazón",
        "espigazon": "Espigazón",
        "floracion": "Floración",
        "llenado_de_grano": "Llenado de grano",
    }
    return mapping.get(stage_key, stage_key.replace("_", " ").capitalize())


def pretty_feature_name(feature_name: str) -> str:
    if feature_name.startswith("lluvia_"):
        stage_key = feature_name.replace("lluvia_", "", 1)
        return f"Precipitación acumulada en {stage_to_spanish(stage_key)} (mm)"
    if feature_name.startswith("tmax_"):
        stage_key = feature_name.replace("tmax_", "", 1)
        return f"Temperatura máxima media en {stage_to_spanish(stage_key)} (°C)"
    if feature_name.startswith("tmin_"):
        stage_key = feature_name.replace("tmin_", "", 1)
        return f"Temperatura mínima media en {stage_to_spanish(stage_key)} (°C)"
    soil_map = {
        "suelo_ind_prod": "Índice de productividad del suelo (promedio provincial)",
        "suelo_profundidad": "Profundidad efectiva del suelo (cm, promedio provincial)",
        "suelo_textura_encoded": "Textura del suelo (codificada, moda provincial)",
        "suelo_drenaje_encoded": "Drenaje del suelo (codificado, moda provincial)",
    }
    return soil_map.get(feature_name, feature_name)


def prepare_features(df: pd.DataFrame) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str]
]:
    """
    Aplica limpieza básica y construye matrices de features y target.

    Devuelve: df_clean, X_clean, y_clean, años_clean, feature_cols
    """
    feature_cols = get_feature_columns(df)
    clima_cols = [col for col in feature_cols if not col.startswith("suelo_")]
    suelo_cols = [col for col in feature_cols if col.startswith("suelo_")]
    print(
        f"Features: {len(feature_cols)} "
        f"(clima={len(clima_cols)}, suelo={len(suelo_cols)})"
    )

    X = df[feature_cols]
    y = df[COL_TARGET]
    años = df["AÑO"]

    mask_ok = X.notna().all(axis=1) & y.notna()
    df_clean = df.loc[mask_ok].copy()
    X_clean = X.loc[mask_ok].copy()
    y_clean = y.loc[mask_ok].copy()
    años_clean = años.loc[mask_ok].copy()
    print(f"Registros válidos (sin NaN): {len(X_clean)}")

    return df_clean, X_clean, y_clean, años_clean, feature_cols


def temporal_train_test_split(
    X: pd.DataFrame, y: pd.Series, años: pd.Series
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Divide en train / test usando un split temporal fijo en AÑO_SPLIT.
    Devuelve matrices y máscaras de train / test.
    """
    mask_train = años <= AÑO_SPLIT
    mask_test = años > AÑO_SPLIT
    X_train = X.loc[mask_train]
    y_train = y.loc[mask_train]
    X_test = X.loc[mask_test]
    y_test = y.loc[mask_test]

    print(
        f"Split temporal: train=1990-{AÑO_SPLIT} ({len(X_train)}), "
        f"test={AÑO_SPLIT + 1}-2021 ({len(X_test)})"
    )

    return X_train, y_train, X_test, y_test, mask_train, mask_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.Series
) -> tuple[np.ndarray, float, float, float]:
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))

    print("\n" + "=" * 70)
    print("MÉTRICAS: Modelo con clima + suelo (validación temporal 2016-2021)")
    print("=" * 70)
    print(f"R2:   {r2:.4f}")
    print(f"RMSE: {rmse:.2f} kg/ha")
    print(f"MAE:  {mae:.2f} kg/ha")

    return y_pred, r2, rmse, mae


def plot_residual_scatter(
    y_test: pd.Series,
    y_pred: np.ndarray,
    ruta_salida: Path,
) -> None:
    """Scatter de rinde ajustado (residuos) real vs. predicho."""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(
        y_test,
        y_pred,
        alpha=0.6,
        edgecolors="none",
        s=30,
        color="#2d5a27",
    )
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="y = x")
    ax.set_xlabel("Rinde ajustado real (kg/ha)")
    ax.set_ylabel("Rinde ajustado predicho (kg/ha)")
    ax.set_title(
        "Rinde ajustado: valores reales vs. predichos\n"
        "Modelo clima + suelo (validación temporal 2016–2021)"
    )
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_aspect("equal", adjustable="box")
    plt.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.12)
    plt.savefig(ruta_salida, dpi=150)
    plt.close()


def plot_absolute_scatter(
    df_test: pd.DataFrame,
    y_pred: np.ndarray,
    ruta_salida: Path,
) -> None:
    """Scatter de rendimiento absoluto real vs. calculado (kg/ha)."""
    y_real_abs = df_test["RENDIMIENTO - KG X HA"].values
    y_pred_abs = y_pred + df_test["Rinde_Tendencia"].values

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(
        y_real_abs,
        y_pred_abs,
        alpha=0.6,
        edgecolors="none",
        s=30,
        color="#2d5a27",
    )
    min_abs = min(y_real_abs.min(), y_pred_abs.min())
    max_abs = max(y_real_abs.max(), y_pred_abs.max())
    ax.plot([min_abs, max_abs], [min_abs, max_abs], "r--", lw=2, label="y = x")
    ax.set_xlabel("Rendimiento real (kg/ha)")
    ax.set_ylabel("Rendimiento predicho por el modelo (kg/ha)")
    ax.set_title(
        "Rendimiento real vs. calculado por el modelo\n"
        "Clima + suelo, validación temporal 2016–2021"
    )
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_aspect("equal", adjustable="box")
    plt.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.12)
    plt.savefig(ruta_salida, dpi=150)
    plt.close()


def compute_feature_importance(
    model: RandomForestRegressor, feature_cols: list[str]
) -> tuple[list[str], np.ndarray]:
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    names_ordered = [pretty_feature_name(feature_cols[i]) for i in order]
    values_ordered = importances[order]
    return names_ordered, values_ordered


def plot_feature_importance(
    names_ordered: list[str],
    values_ordered: np.ndarray,
    ruta_importancia: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.barh(range(len(names_ordered)), values_ordered, align="center")
    ax.set_yticks(range(len(names_ordered)))
    ax.set_yticklabels(names_ordered, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Importancia (Gini)")
    ax.set_title("Importancia de variables (Clima + Suelo)")
    plt.subplots_adjust(left=0.46, right=0.98, top=0.94, bottom=0.06)
    plt.savefig(ruta_importancia, dpi=150)
    plt.close()


def compute_and_plot_shap(
    model: RandomForestRegressor,
    X_train: pd.DataFrame,
    feature_cols: list[str],
    ruta_shap: Path,
) -> None:
    X_sample = X_train.sample(
        n=min(SHAP_SAMPLE_SIZE, len(X_train)), random_state=RANDOM_STATE
    )
    explainer = shap.TreeExplainer(model, X_sample)
    shap_values = explainer.shap_values(X_sample)

    X_sample_pretty = X_sample.rename(
        columns={col: pretty_feature_name(col) for col in feature_cols}
    )
    shap.summary_plot(shap_values, X_sample_pretty, show=False, plot_size=(14, 9))
    plt.title("Análisis de importancia de variables en el rendimiento de trigo")
    plt.tight_layout()
    plt.savefig(ruta_shap, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    processed = ROOT / "data" / "processed"
    figures = ROOT / "reports" / "figures"
    ruta_dataset = processed / "dataset_final.csv"
    ruta_scatter = figures / "scatter_residuals.png"
    ruta_scatter_absoluto = figures / "scatter.png"
    ruta_importancia = figures / "importance.png"
    ruta_shap = figures / "shap.png"

    print("Cargando dataset con variables de suelo...")
    df = pd.read_csv(ruta_dataset)

    (
        df_clean,
        X_clean,
        y_clean,
        años_clean,
        feature_cols,
    ) = prepare_features(df)

    X_train, y_train, X_test, y_test, _, mask_test = temporal_train_test_split(
        X_clean, y_clean, años_clean
    )

    model = train_model(X_train, y_train)
    y_pred, r2, rmse, mae = evaluate_model(model, X_test, y_test)

    figures.mkdir(parents=True, exist_ok=True)
    plot_residual_scatter(y_test=y_test, y_pred=y_pred, ruta_salida=ruta_scatter)

    df_test = df_clean.loc[mask_test]
    plot_absolute_scatter(
        df_test=df_test, y_pred=y_pred, ruta_salida=ruta_scatter_absoluto
    )

    names_ordered, values_ordered = compute_feature_importance(
        model=model, feature_cols=feature_cols
    )
    plot_feature_importance(
        names_ordered=names_ordered,
        values_ordered=values_ordered,
        ruta_importancia=ruta_importancia,
    )

    compute_and_plot_shap(
        model=model,
        X_train=X_train,
        feature_cols=feature_cols,
        ruta_shap=ruta_shap,
    )

    print("\nFiguras generadas:")
    print(f"  - {ruta_scatter_absoluto}")
    print(f"  - {ruta_scatter}")
    print(f"  - {ruta_importancia}")
    print(f"  - {ruta_shap}")


if __name__ == "__main__":
    main()
