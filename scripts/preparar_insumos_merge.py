"""
Genera rinde_trigo_pampa.csv y mapeo_departamento_estacion.csv a partir de datos existentes
para poder ejecutar el merge (Paso 3).
"""
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

# Región Pampeana (provincias típicas para trigo)
PROVINCIAS_PAMPA = {
    "Buenos Aires",
    "Santa Fe",
    "Córdoba",
    "La Pampa",
    "Entre Ríos",
}

# Una estación de clima por provincia (primera que aparece en clima para esa provincia)
PROVINCIA_A_STATION = {
    "Buenos Aires": "AR000087692",
    "Santa Fe": "AR000087257",
    "Córdoba": "AR000087344",
    "La Pampa": "AR000087623",
    "Entre Ríos": "AR000000004",
}


def main() -> None:
    processed = ROOT / "data" / "processed"
    raw = ROOT / "data" / "raw"

    # 1) Rinde Pampa: desde DATOS TRIGO, filtrado por provincia y año 1990-2021
    ruta_trigo = raw / "DATOS TRIGO - ARGENTINA - 1990 A 2025.csv"
    print(f"Leyendo {ruta_trigo}...")
    trigo = pd.read_csv(ruta_trigo, sep=";", encoding="latin-1")
    rinde = trigo[
        (trigo["PROVINCIA"].isin(PROVINCIAS_PAMPA))
        & (trigo["AÑO"] >= 1990)
        & (trigo["AÑO"] <= 2021)
    ][["PROVINCIA", "DEPARTAMENTO", "AÑO", "RENDIMIENTO - KG X HA"]].copy()
    ruta_rinde = processed / "rinde_trigo_pampa.csv"
    rinde.to_csv(ruta_rinde, index=False)
    print(f"Guardado {ruta_rinde} ({len(rinde)} filas).")

    # 2) Mapeo: cada (PROVINCIA, DEPARTAMENTO) -> STATION_ID (una estación por provincia)
    mapeo = (
        rinde[["PROVINCIA", "DEPARTAMENTO"]]
        .drop_duplicates()
        .assign(
            STATION_ID=lambda df: df["PROVINCIA"].map(PROVINCIA_A_STATION)
        )
    )
    mapeo = mapeo.dropna(subset=["STATION_ID"])
    ruta_mapeo = processed / "mapeo_departamento_estacion.csv"
    mapeo.to_csv(ruta_mapeo, index=False)
    print(f"Guardado {ruta_mapeo} ({len(mapeo)} filas).")


if __name__ == "__main__":
    main()
