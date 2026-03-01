"""
Genera un informe HTML visual con gráficos a partir del dataset maestro y figuras del pipeline.
Informe atractivo para presentación del proyecto (trigo, clima, modelo, XAI).
"""
import base64
import io
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

# Colores: tonos verdes/trigo para tema agrícola
COLOR_PRIMARY = "#2d5a27"
COLOR_SECONDARY = "#7cb342"
COLOR_ACCENT = "#c0ca33"
COLOR_BG = "#faf8f5"
COLOR_CARD = "#ffffff"


def fig_to_base64(fig: plt.Figure, dpi: int = 120) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def png_file_to_base64(path: Path) -> str:
    data = path.read_bytes()
    return base64.b64encode(data).decode("utf-8")


def main() -> None:
    ruta_dataset = ROOT / "data" / "processed" / "dataset_maestro_ia.csv"
    ruta_informe = ROOT / "reports" / "informe_trigo.html"

    print("Cargando dataset...")
    df = pd.read_csv(ruta_dataset)
    df = df.dropna(subset=[COL_RENDIMIENTO])
    feature_cols = get_feature_columns(df)
    lluvia_cols = [c for c in feature_cols if c.startswith("lluvia_")]

    # --- Gráfico 1: Rendimiento por provincia (boxplot) ---
    plt.rc("axes", grid=True)
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    provincias = df["PROVINCIA"].unique()
    data_by_prov = [df[df["PROVINCIA"] == p][COL_RENDIMIENTO].dropna() for p in provincias]
    bp = ax1.boxplot(data_by_prov, tick_labels=provincias, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor(COLOR_SECONDARY)
        patch.set_alpha(0.7)
    ax1.set_ylabel("Rendimiento (kg/ha)")
    ax1.set_title("Distribución del rendimiento de trigo por provincia (Región Pampeana)")
    ax1.tick_params(axis="x", rotation=15)
    b64_boxplot = fig_to_base64(fig1)
    plt.close()

    # --- Gráfico 2: Rendimiento medio por año ---
    fig2, ax2 = plt.subplots(figsize=(9, 4))
    yearly = df.groupby("AÑO")[COL_RENDIMIENTO].mean()
    ax2.fill_between(yearly.index, yearly.values, alpha=0.4, color=COLOR_PRIMARY)
    ax2.plot(yearly.index, yearly.values, color=COLOR_PRIMARY, linewidth=2)
    ax2.set_xlabel("Año")
    ax2.set_ylabel("Rendimiento medio (kg/ha)")
    ax2.set_title("Evolución del rendimiento medio de trigo (1990-2021)")
    b64_trend = fig_to_base64(fig2)
    plt.close()

    # --- Gráfico 3: Precipitación media por etapa fenológica ---
    etapas = [c.replace("lluvia_", "").replace("_", " ") for c in lluvia_cols]
    medias = df[lluvia_cols].mean()
    fig3, ax3 = plt.subplots(figsize=(9, 4))
    bars = ax3.barh(etapas, medias.values, color=COLOR_SECONDARY, alpha=0.8)
    ax3.set_xlabel("Precipitación acumulada media (mm)")
    ax3.set_title("Precipitación por etapa fenológica del trigo")
    b64_lluvia = fig_to_base64(fig3)
    plt.close()

    # --- Métricas del modelo (recalcular para el informe) ---
    X = df[feature_cols]
    y = df[COL_RENDIMIENTO]
    mask_ok = X.notna().all(axis=1) & y.notna()
    X_clean = X.loc[mask_ok]
    y_clean = y.loc[mask_ok]
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    # --- Incrustar figuras existentes ---
    figures = ROOT / "reports" / "figures"
    scatter_path = figures / "scatter_real_vs_predicho.png"
    importancia_path = figures / "importancia_variables.png"
    shap_path = figures / "shap_summary_plot.png"
    mae_temporal_path = figures / "mae_por_año_temporal.png"
    scatter_temporal_path = figures / "scatter_temporal.png"
    mae_detrended_path = figures / "mae_por_año_detrended.png"
    scatter_detrended_path = figures / "scatter_detrended_temporal.png"
    scatter_soil_path = figures / "scatter_con_suelo.png"
    importancia_soil_path = figures / "importancia_con_suelo.png"
    shap_soil_path = figures / "shap_con_suelo.png"
    comparacion_path = figures / "comparacion_metricas.png"

    b64_scatter = png_file_to_base64(scatter_path) if scatter_path.exists() else ""
    b64_importancia = png_file_to_base64(importancia_path) if importancia_path.exists() else ""
    b64_shap = png_file_to_base64(shap_path) if shap_path.exists() else ""
    b64_mae_temporal = png_file_to_base64(mae_temporal_path) if mae_temporal_path.exists() else ""
    b64_scatter_temporal = png_file_to_base64(scatter_temporal_path) if scatter_temporal_path.exists() else ""
    b64_mae_detrended = png_file_to_base64(mae_detrended_path) if mae_detrended_path.exists() else ""
    b64_scatter_detrended = png_file_to_base64(scatter_detrended_path) if scatter_detrended_path.exists() else ""
    b64_scatter_soil = png_file_to_base64(scatter_soil_path) if scatter_soil_path.exists() else ""
    b64_importancia_soil = png_file_to_base64(importancia_soil_path) if importancia_soil_path.exists() else ""
    b64_shap_soil = png_file_to_base64(shap_soil_path) if shap_soil_path.exists() else ""
    b64_comparacion = png_file_to_base64(comparacion_path) if comparacion_path.exists() else ""

    # --- Construir HTML ---
    n_reg = len(df)
    n_prov = df["PROVINCIA"].nunique()
    años = f"{int(df['AÑO'].min())}–{int(df['AÑO'].max())}"

    # --- Resumen de suelo (si existe dataset con suelo) ---
    soil_summary_html = ""
    ruta_dataset_suelo = ROOT / "data" / "processed" / "dataset_maestro_ia_con_suelo.csv"
    if ruta_dataset_suelo.exists():
        df_soil = pd.read_csv(ruta_dataset_suelo)
        soil_cols = ["suelo_ind_prod", "suelo_profundidad"]
        if all(col in df_soil.columns for col in soil_cols):
            resumen_suelo = (
                df_soil.groupby("PROVINCIA")[soil_cols]
                .mean()
                .reset_index()
                .sort_values("PROVINCIA")
            )
            rows = []
            for _, row in resumen_suelo.iterrows():
                rows.append(
                    f"<tr><td>{row['PROVINCIA']}</td>"
                    f"<td>{row['suelo_ind_prod']:.1f}</td>"
                    f"<td>{row['suelo_profundidad']:.1f}</td></tr>"
                )
            soil_summary_html = (
                "<table style='width:100%; border-collapse:collapse; margin-top:1rem;'>"
                "<thead><tr>"
                "<th style='text-align:left; border-bottom:1px solid #ccc;'>Provincia</th>"
                "<th style='text-align:right; border-bottom:1px solid #ccc;'>Índice de productividad</th>"
                "<th style='text-align:right; border-bottom:1px solid #ccc;'>Profundidad (cm)</th>"
                "</tr></thead><tbody>"
                + "".join(rows)
                + "</tbody></table>"
            )

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Informe: Trigo y Clima en la Región Pampeana</title>
  <style>
    :root {{ --primary: {COLOR_PRIMARY}; --secondary: {COLOR_SECONDARY}; --accent: {COLOR_ACCENT}; --bg: {COLOR_BG}; --card: {COLOR_CARD}; }}
    * {{ box-sizing: border-box; }}
    body {{ font-family: 'Segoe UI', system-ui, sans-serif; margin: 0; background: var(--bg); color: #333; line-height: 1.6; }}
    .container {{ max-width: 960px; margin: 0 auto; padding: 2rem; }}
    header {{ text-align: center; padding: 2rem 0; border-bottom: 3px solid var(--primary); }}
    header h1 {{ color: var(--primary); font-size: 1.8rem; margin: 0; }}
    header p {{ color: #666; margin: 0.5rem 0 0; }}
    .card {{ background: var(--card); border-radius: 12px; padding: 1.5rem; margin: 1.5rem 0; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
    .card h2 {{ color: var(--primary); font-size: 1.25rem; margin-top: 0; border-bottom: 2px solid var(--secondary); padding-bottom: 0.5rem; }}
    .stats {{ display: flex; flex-wrap: wrap; gap: 1rem; margin: 1rem 0; }}
    .stat {{ background: linear-gradient(135deg, var(--secondary), var(--accent)); color: white; padding: 1rem 1.5rem; border-radius: 8px; min-width: 140px; }}
    .stat strong {{ display: block; font-size: 1.5rem; }}
    .stat span {{ font-size: 0.9rem; opacity: 0.95; }}
    img.report-img {{ max-width: 100%; height: auto; border-radius: 8px; display: block; margin: 1rem auto; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 1rem; }}
    .metric {{ text-align: center; padding: 1rem; background: #f5f5f5; border-radius: 8px; }}
    .metric strong {{ color: var(--primary); font-size: 1.3rem; }}
    table {{ font-size: 0.9rem; }}
    th, td {{ padding: 0.35rem 0.5rem; }}
    footer {{ text-align: center; padding: 2rem; color: #888; font-size: 0.9rem; }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>Trigo y Clima en la Región Pampeana</h1>
      <p>Análisis de rendimiento, variables climáticas por etapa fenológica y modelo predictivo (Random Forest + XAI)</p>
    </header>

    <section class="card">
      <h2>Resumen del dataset</h2>
      <div class="stats">
        <div class="stat"><strong>{n_reg:,}</strong><span>Registros</span></div>
        <div class="stat"><strong>{n_prov}</strong><span>Provincias</span></div>
        <div class="stat"><strong>{años}</strong><span>Período</span></div>
      </div>
      <p>Datos de rendimiento de trigo (kg/ha) vinculados a variables climáticas (precipitación, Tmax, Tmin) por etapa fenológica (Siembra, Emergencia, Macollaje, Encañazón, Espigazón, Floración, Llenado de grano).</p>
    </section>

    <section class="card">
      <h2>Rendimiento por provincia</h2>
      <p>Distribución del rendimiento en las provincias pampeanas.</p>
      <img src="data:image/png;base64,{b64_boxplot}" alt="Rendimiento por provincia" class="report-img">
    </section>

    <section class="card">
      <h2>Evolución temporal del rendimiento</h2>
      <p>Rendimiento medio anual en el período 1990-2021.</p>
      <img src="data:image/png;base64,{b64_trend}" alt="Rendimiento por año" class="report-img">
    </section>

    <section class="card">
      <h2>Precipitación por etapa fenológica</h2>
      <p>Precipitación acumulada media (mm) en cada etapa del ciclo del trigo.</p>
      <img src="data:image/png;base64,{b64_lluvia}" alt="Lluvia por etapa" class="report-img">
    </section>

    <section class="card">
      <h2>Modelo: Random Forest (Iqbal et al., 2024)</h2>
      <p>Predicción de rendimiento a partir de variables climáticas por etapa. División 80% entrenamiento / 20% prueba.</p>
      <div class="metrics">
        <div class="metric"><strong>R²</strong><br>{r2:.4f}</div>
        <div class="metric"><strong>RMSE</strong><br>{rmse:.0f} kg/ha</div>
        <div class="metric"><strong>MAE</strong><br>{mae:.0f} kg/ha</div>
      </div>
      <p><strong>Real vs. predicho (conjunto de prueba):</strong></p>
      <img src="data:image/png;base64,{b64_scatter}" alt="Real vs predicho" class="report-img">
    </section>

    <section class="card">
      <h2>Importancia de variables (XAI)</h2>
      <p>Importancia Gini de las variables climáticas en el Random Forest.</p>
      <img src="data:image/png;base64,{b64_importancia}" alt="Importancia variables" class="report-img">
    </section>

    <section class="card">
      <h2>SHAP: impacto en el rendimiento</h2>
      <p>Efecto de cada variable sobre el rendimiento predicho (valores SHAP). Rojo = valor alto de la variable; azul = valor bajo. Eje horizontal: contribución al rendimiento (kg/ha).</p>
      <img src="data:image/png;base64,{b64_shap}" alt="SHAP summary" class="report-img">
    </section>

    <section class="card">
      <h2>Validación temporal y detrending</h2>
      <p>Validación más realista usando años futuros (2016-2021) y eliminación de la tendencia tecnológica del rinde.</p>
      <p>Los gráficos muestran el error por año y la dispersión real vs. predicho antes y después del detrending.</p>
      <img src="data:image/png;base64,{b64_mae_temporal}" alt="MAE por año (rinde absoluto)" class="report-img">
      <img src="data:image/png;base64,{b64_scatter_temporal}" alt="Real vs predicho (rinde absoluto, temporal)" class="report-img">
      <img src="data:image/png;base64,{b64_mae_detrended}" alt="MAE por año (Rinde_Detrended)" class="report-img">
      <img src="data:image/png;base64,{b64_scatter_detrended}" alt="Real vs predicho (Rinde_Detrended, temporal)" class="report-img">
      <p style="font-size:0.9rem; color:#555;">
        El detrending mejora sustancialmente las métricas al aislar la señal climática (reducción del error y R² más estable),
        permitiendo comparar mejor la respuesta del cultivo entre regiones o sistemas productivos.
      </p>
    </section>

    <section class="card">
      <h2>Suelo y modelo clima + suelo</h2>
      <p>Variables edáficas derivadas de las Cartas de Suelo del INTA (índice de productividad, profundidad efectiva, textura y drenaje).</p>
      {soil_summary_html}
      <p style="margin-top:1rem;">
        A continuación se muestran los resultados del modelo que incorpora tanto clima como suelo, junto con la comparación
        frente al modelo sólo climático.
      </p>
      <img src="data:image/png;base64,{b64_scatter_soil}" alt="Real vs predicho (clima + suelo)" class="report-img">
      <img src="data:image/png;base64,{b64_importancia_soil}" alt="Importancia (clima + suelo)" class="report-img">
      <img src="data:image/png;base64,{b64_shap_soil}" alt="SHAP (clima + suelo)" class="report-img">
      <img src="data:image/png;base64,{b64_comparacion}" alt="Comparación métricas: solo clima vs clima + suelo" class="report-img">
      <p style="font-size:0.9rem; color:#555;">
        En este caso, las variables de suelo aportan información estable (estática) y aparecen con menor importancia relativa
        que las variables climáticas de etapas críticas (por ejemplo, lluvia en encañazón y floración). Esto sugiere que, a escala
        departamental y en la Región Pampeana, la variabilidad interanual del rinde está dominada por el clima.
      </p>
    </section>

    <footer>
      Proyecto UTN-DAAD · Datos: NOAA, producción trigo Argentina · Etapas fenológicas: Scian (2004) · Modelo: Iqbal et al. (2024)
    </footer>
  </div>
</body>
</html>
"""

    ruta_informe.parent.mkdir(parents=True, exist_ok=True)
    ruta_informe.write_text(html, encoding="utf-8")
    print(f"Informe guardado: {ruta_informe}")


if __name__ == "__main__":
    main()
