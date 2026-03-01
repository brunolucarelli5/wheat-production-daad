"""
Paso 3: Unión de Rinde y Clima (The Big Merge).
Vincular rendimiento de trigo con variables climáticas usando mapeo geográfico.
"""
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

RANGO_AÑO_MIN = 1990
RANGO_AÑO_MAX = 2021
COL_RENDIMIENTO = "RENDIMIENTO - KG X HA"


def build_dataset_maestro(
    rinde: pd.DataFrame,
    mapeo: pd.DataFrame,
    clima: pd.DataFrame,
) -> pd.DataFrame:
    """
    Une rinde con mapeo (DEPARTAMENTO, PROVINCIA) y luego con clima (STATION_ID=STATION, AÑO).
    Filtra años 1990-2021 y elimina filas con rendimiento faltante.
    """
    # Merge rinde + mapeo por DEPARTAMENTO y PROVINCIA
    rinde_mapeo = rinde.merge(
        mapeo,
        on=["DEPARTAMENTO", "PROVINCIA"],
        how="inner",
    )
    # Merge con clima por STATION_ID = STATION y AÑO
    clima_renamed = clima.rename(columns={"STATION": "STATION_ID"})
    maestro = rinde_mapeo.merge(
        clima_renamed,
        on=["STATION_ID", "AÑO"],
        how="inner",
    )
    # Rango 1990-2021
    maestro = maestro[
        (maestro["AÑO"] >= RANGO_AÑO_MIN) & (maestro["AÑO"] <= RANGO_AÑO_MAX)
    ]
    # Eliminar filas con rendimiento faltante
    maestro = maestro.dropna(subset=[COL_RENDIMIENTO])
    return maestro.reset_index(drop=True)


def main() -> None:
    processed = ROOT / "data" / "processed"
    ruta_rinde = processed / "rinde_trigo_pampa.csv"
    ruta_mapeo = processed / "mapeo_departamento_estacion.csv"
    ruta_clima = processed / "clima_region_pampeana_features.csv"
    ruta_salida = processed / "dataset_maestro_ia.csv"

    print("Leyendo rinde_trigo_pampa.csv...")
    rinde = pd.read_csv(ruta_rinde)
    print("Leyendo mapeo_departamento_estacion.csv...")
    mapeo = pd.read_csv(ruta_mapeo)
    print("Leyendo clima_region_pampeana_features.csv...")
    clima = pd.read_csv(ruta_clima)

    maestro = build_dataset_maestro(rinde, mapeo, clima)
    processed.mkdir(parents=True, exist_ok=True)
    maestro.to_csv(ruta_salida, index=False)

    print(f"\nArchivo guardado: {ruta_salida}")
    print(f"Total de registros: {len(maestro)}")
    print("\nPrimeras 5 filas:")
    print(maestro.head().to_string(index=False))


if __name__ == "__main__":
    main()
