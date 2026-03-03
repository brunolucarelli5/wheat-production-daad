"""
Genera un informe en Markdown con métricas del modelo clima + suelo y referencias a figuras.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent.parent
COL_RENDIMIENTO = "RENDIMIENTO - KG X HA"
COL_TARGET = "Rinde_Detrended"
RANDOM_STATE = 42
N_ESTIMATORS = 100
MAX_DEPTH = 20
AÑO_SPLIT = 2015


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Columnas climáticas + edáficas para el modelo."""
    exclude = {
        "PROVINCIA",
        "DEPARTAMENTO",
        "AÑO",
        "STATION_ID",
        COL_RENDIMIENTO,
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
    ruta_dataset = ROOT / "data" / "processed" / "dataset_final.csv"
    ruta_informe_md = ROOT / "reports" / "INFORME_TRIGO.md"

    if not ruta_dataset.exists():
        print(f"ERROR: No existe {ruta_dataset}. Ejecute antes: make soil")
        return

    print("Cargando dataset con suelo...")
    df = pd.read_csv(ruta_dataset)
    df = df.dropna(subset=[COL_RENDIMIENTO, COL_TARGET])
    feature_cols = get_feature_columns(df)

    n_total = len(df)
    has_provincia = "PROVINCIA" in df.columns
    if has_provincia:
        provincias = sorted(df["PROVINCIA"].unique())
    else:
        provincias = []
    años = f"{int(df['AÑO'].min())}–{int(df['AÑO'].max())}"
    rinde_mean = df[COL_RENDIMIENTO].mean()
    rinde_std = df[COL_RENDIMIENTO].std()
    rinde_min = df[COL_RENDIMIENTO].min()
    rinde_max = df[COL_RENDIMIENTO].max()

    # Split temporal (igual que train_with_soil)
    X = df[feature_cols]
    y = df[COL_TARGET]
    años_serie = df["AÑO"]
    mask_ok = X.notna().all(axis=1) & y.notna()
    X_clean = X.loc[mask_ok]
    y_clean = y.loc[mask_ok]
    años_clean = años_serie.loc[mask_ok]
    mask_train = años_clean <= AÑO_SPLIT
    mask_test = años_clean > AÑO_SPLIT
    X_train, X_test = X_clean.loc[mask_train], X_clean.loc[mask_test]
    y_train, y_test = y_clean.loc[mask_train], y_clean.loc[mask_test]
    n_train = len(X_train)
    n_test = len(X_test)

    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:10]
    top_features = [(feature_cols[i], importances[i]) for i in order]

    stats_prov = []
    if has_provincia:
        for prov in provincias:
            subset = df[df["PROVINCIA"] == prov][COL_RENDIMIENTO]
            stats_prov.append(
                {
                    "Provincia": prov,
                    "n": len(subset),
                    "Media (kg/ha)": f"{subset.mean():.0f}",
                    "Desv. Est.": f"{subset.std():.0f}",
                }
            )
    else:
        stats_prov.append(
            {
                "Provincia": "Total",
                "n": n_total,
                "Media (kg/ha)": f"{rinde_mean:.0f}",
                "Desv. Est.": f"{rinde_std:.0f}",
            }
        )

    provincias_str = ", ".join(provincias) if provincias else "N/A"

    md_content = f"""# Informe: Predicción de Rendimiento de Trigo en la Región Pampeana

**Proyecto:** Beca UTN-DAAD  
**Período:** {años}  
**Fecha de generación:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

---

## 1. Resumen Ejecutivo

Este informe presenta el análisis de rendimiento de trigo en la **Región Pampeana** (Argentina) utilizando datos climáticos (NOAA), datos de producción (Ministerio de Agricultura) y variables edáficas (Cartas de Suelo INTA). El modelo predictivo es un **Random Forest** que combina **clima y suelo** para predecir el rinde ajustado (residuos respecto a la tendencia tecnológica).

**Objetivo:** Comparar sistemas productivos entre la Región Pampeana y Alemania mediante análisis de importancia de variables (XAI).

---

## 2. Dataset

### 2.1 Características generales

- **Registros totales:** {n_total:,}
- **Provincias:** {provincias_str}
- **Período temporal:** {años}
- **Variables:** {len(feature_cols)} (climáticas por etapa fenológica + edáficas por provincia)

### 2.2 Etapas fenológicas del trigo (Scian, 2004)

| Etapa | Período |
|-------|---------|
| Siembra | 21 jun – 10 jul |
| Emergencia | 11 jul – 30 ago |
| Macollaje | 1 sep – 30 sep |
| Encañazón | 1 oct – 20 oct |
| Espigazón | 21 oct – 31 oct |
| Floración | 1 nov – 10 nov |
| Llenado de grano | 11 nov – 30 nov |

### 2.3 Estadísticas de rendimiento (kg/ha)

**Global:**
- Media: **{rinde_mean:.0f} kg/ha**
- Desviación estándar: **{rinde_std:.0f} kg/ha**
- Mínimo: **{rinde_min:.0f} kg/ha**
- Máximo: **{rinde_max:.0f} kg/ha**

**Por provincia:**

| Provincia | n | Media (kg/ha) | Desv. Est. |
|-----------|---|---------------|------------|
"""

    for stat in stats_prov:
        md_content += f"| {stat['Provincia']} | {stat['n']} | {stat['Media (kg/ha)']} | {stat['Desv. Est.']} |\n"

    md_content += f"""
---

## 3. Modelo Predictivo: Clima + Suelo

### 3.1 Configuración

- **Algoritmo:** Random Forest Regressor (Iqbal et al., 2024)
- **Target:** Rinde_Detrended (residuos respecto a la tendencia tecnológica)
- **Validación:** Temporal — entrenamiento 1990-{AÑO_SPLIT}, prueba {AÑO_SPLIT+1}-2021
- **Registros entrenamiento:** {n_train:,}
- **Registros prueba:** {n_test:,}

### 3.2 Métricas (conjunto de prueba, Rinde_Detrended)

| Métrica | Valor |
|---------|-------|
| **R²** | {r2:.4f} |
| **RMSE** | {rmse:.2f} kg/ha |
| **MAE** | {mae:.2f} kg/ha |

**Figuras:**
- `scatter.png` — Rendimiento real vs. predicho (kg/ha absolutos).
- `scatter_residuals.png` — Rinde ajustado (residuos): real vs. predicho.

---

## 4. Análisis de Importancia (XAI)

### 4.1 Top 10 variables (Gini)

| Ranking | Variable | Importancia |
|---------|----------|-------------|
"""

    for idx, (feat, imp) in enumerate(top_features, 1):
        md_content += f"| {idx} | `{feat}` | {imp:.4f} |\n"

    md_content += """
**Figura:** `importance.png`

### 4.2 SHAP

El análisis SHAP muestra el efecto de cada variable sobre el rendimiento predicho (valores positivos aumentan el rinde, negativos lo reducen). Color: valor de la variable (rojo = alto, azul = bajo).

**Figura:** `shap.png`

---

## 5. Conclusiones

1. **Datos integrados:** Clima (NOAA), rendimiento (Ministerio) y suelo (INTA) vinculados por departamento y año.

2. **Modelo único:** Random Forest con clima + suelo, validación temporal 2016-2021.

3. **Variables críticas:** Temperatura y precipitación en floración y llenado de grano suelen dominar; las variables edáficas aportan información estable a escala provincial.

4. **Aplicación:** Análisis útil para la comparación de sistemas productivos (Región Pampeana vs. Alemania), objetivo de la beca UTN-DAAD.

---

## 6. Referencias

- **Scian, B. (2004).** Metodología de decadias y etapas fenológicas para trigo en la Región Pampeana.
- **Iqbal et al. (2024).** Machine Learning aplicado a predicción de rendimiento de cultivos.
- **Datos:** NOAA, Ministerio de Agricultura Argentina, INTA (Cartas de Suelo).

---

## 7. Archivos generados

### Datos
- `dataset_maestro_ia.csv` — Rinde + clima (intermedio).
- `dataset_final.csv` — Dataset final rinde + clima + suelo.

### Figuras
- `scatter.png` — Real vs. predicho (kg/ha).
- `scatter_residuals.png` — Rinde ajustado: real vs. predicho.
- `importance.png` — Importancia de variables (Gini).
- `shap.png` — SHAP summary.

### Informes
- `informe_trigo.html` — Informe visual.
- `INFORME_TRIGO.md` — Este documento.

---

**Generado por:** Pipeline clima + suelo (proyecto UTN-DAAD)
"""

    ruta_informe_md.parent.mkdir(parents=True, exist_ok=True)
    ruta_informe_md.write_text(md_content, encoding="utf-8")
    print(f"Informe Markdown guardado: {ruta_informe_md}")


if __name__ == "__main__":
    main()
