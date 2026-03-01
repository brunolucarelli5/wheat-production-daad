import glob
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent

# 1. Definir el patrón de búsqueda (CSV de NOAA en data/raw/noaa)
archivos_csv = sorted(glob.glob(str(ROOT / "data" / "raw" / "noaa" / "*.csv"))) 

# Opcional: Filtrar solo los archivos que vienen de la NOAA (ej. los que tienen números)
# archivos_csv = [f for f in archivos_csv if f.startswith('4243')] 

print(f"Se encontraron {len(archivos_csv)} archivos para unir.")

# 2. Leer cada archivo y guardarlo en una lista
lista_df = []
for archivo in archivos_csv:
    # Leemos el archivo. Nota: checkeá si el separador es coma (,) o punto y coma (;)
    df = pd.read_csv(archivo)
    lista_df.append(df)

# 3. Concatenar todos los DataFrames en uno solo
df_total = pd.concat(lista_df, ignore_index=True)

# 4. (Opcional) Ordenar por fecha para que quede prolijo
if 'DATE' in df_total.columns:
    df_total['DATE'] = pd.to_datetime(df_total['DATE'])
    df_total = df_total.sort_values(by=['STATION', 'DATE'])

# 5. Guardar el resultado final
path_salida = ROOT / "data" / "processed" / "clima_argentina_completo.csv"
path_salida.parent.mkdir(parents=True, exist_ok=True)
df_total.to_csv(path_salida, index=False)

print(f"¡Hecho! El archivo unificado se guardó como: {path_salida}")
print(f"Total de registros: {len(df_total)}")