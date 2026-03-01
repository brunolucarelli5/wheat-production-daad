"""
Pipeline completo: ejecuta todos los pasos desde mapeo de etapas hasta XAI.
Genera imágenes, CSV procesados e informes (HTML + Markdown).
"""
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
PYTHON = sys.executable


def run_script(script_path: str, description: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {description}")
    print(f"{'='*70}")
    full_path = ROOT / script_path
    result = subprocess.run([PYTHON, str(full_path)], cwd=ROOT)
    if result.returncode != 0:
        print(f"ERROR: {script_path} falló con código {result.returncode}")
        sys.exit(1)


def main() -> None:
    print("PIPELINE COMPLETO: Trigo + Clima → Modelo + XAI + Informes")
    print(f"Python: {PYTHON}")
    print(f"Root: {ROOT}")

    # Paso 1: Mapeo de fechas y etapas fenológicas
    run_script("src/features/build_phenological_features.py", "PASO 1: Mapeo de decadias y etapas fenológicas")

    # Paso 2: Agregación de predictores anuales por estación
    run_script("src/features/aggregate_climate.py", "PASO 2: Agregación de predictores climáticos")

    # Paso 3: Preparar insumos (rinde + mapeo) y merge
    run_script("scripts/preparar_insumos_merge.py", "PASO 3a: Preparar rinde y mapeo")
    run_script("scripts/merge_rinde_clima.py", "PASO 3b: Merge rinde + clima → dataset maestro")

    # Paso 4: Entrenamiento Random Forest
    run_script("src/models/train_model.py", "PASO 4: Entrenamiento Random Forest y métricas")

    # Paso 5: Análisis XAI (importancia + SHAP)
    run_script("src/models/explain_model.py", "PASO 5: Análisis de importancia y XAI (SHAP)")

    # Paso 6: Validación temporal
    run_script("src/models/validate_temporal.py", "PASO 6: Validación temporal (2016-2021)")

    # Paso 7: Detrending y reentrenamiento
    run_script("src/models/train_detrended.py", "PASO 7: Detrending y modelo sobre residuos")

    # Paso 8: Integración de variables de suelo
    run_script("src/features/integrate_soil.py", "PASO 8: Integración de variables de suelo (INTA)")

    # Paso 9: Modelo con clima + suelo
    run_script("src/models/train_with_soil.py", "PASO 9: Random Forest con clima + suelo (validación temporal)")

    # Informe HTML
    run_script("src/visualization/generate_html_report.py", "INFORME: Generación HTML con gráficos")

    # Informe Markdown
    run_script("src/visualization/generate_markdown_report.py", "INFORME: Generación Markdown con métricas")

    print(f"\n{'='*70}")
    print("  PIPELINE COMPLETADO")
    print(f"{'='*70}")
    print("\nArchivos generados:")
    print("  Datos procesados:")
    print("    - data/processed/clima_region_pampeana_feno.csv")
    print("    - data/processed/clima_region_pampeana_features.csv")
    print("    - data/processed/dataset_maestro_ia.csv")
    print("    - data/processed/dataset_maestro_ia_detrended.csv")
    print("    - data/processed/dataset_maestro_ia_con_suelo.csv")
    print("  Figuras:")
    print("    - reports/figures/scatter_real_vs_predicho.png")
    print("    - reports/figures/importancia_variables.png")
    print("    - reports/figures/shap_summary_plot.png")
    print("    - reports/figures/mae_por_año_temporal.png")
    print("    - reports/figures/scatter_detrended_temporal.png")
    print("    - reports/figures/ejemplo_tendencia.png")
    print("    - reports/figures/scatter_con_suelo.png")
    print("    - reports/figures/importancia_con_suelo.png")
    print("    - reports/figures/shap_con_suelo.png")
    print("    - reports/figures/comparacion_metricas.png")
    print("  Informes:")
    print("    - reports/informe_trigo.html")
    print("    - reports/INFORME_TRIGO.md")


if __name__ == "__main__":
    main()
