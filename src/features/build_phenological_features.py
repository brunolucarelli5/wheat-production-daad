from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent


def get_decada(day: int) -> int:
    if day <= 10:
        return 1
    if day <= 20:
        return 2
    return 3


def get_etapa_fenologica(month: int, day: int) -> str:
    # Siembra: 21 jun al 10 jul (jn3-jl1)
    if (month == 6 and day >= 21) or (month == 7 and day <= 10):
        return "Siembra"
    # Emergencia: 11 jul al 30 ago (jl2-ag3)
    if (month == 7 and day >= 11) or (month == 8 and day <= 30):
        return "Emergencia"
    # Macollaje: septiembre completo (se1-se3)
    if month == 9:
        return "Macollaje"
    # Encañazón: 1 oct al 20 oct (oc1-oc2)
    if month == 10 and day <= 20:
        return "Encañazón"
    # Espigazón: 21 oct al 31 oct (oc3)
    if month == 10 and day >= 21:
        return "Espigazón"
    # Floración: 1 nov al 10 nov (no1)
    if month == 11 and day <= 10:
        return "Floración"
    # Llenado de grano: 11 nov al 30 nov (no2-no3)
    if month == 11 and 11 <= day <= 30:
        return "Llenado de grano"
    return "Fuera de ciclo"


def add_decada_y_etapa(df: pd.DataFrame) -> pd.DataFrame:
    """
    Espera columnas: 'STATION', 'DATE', 'PRCP', 'TMAX', 'TMIN'.
    Devuelve el DataFrame con columnas nuevas: 'DECADIA' y 'ETAPA_FENOLOGICA'.
    """
    df_resultado = df.copy()
    df_resultado["DATE"] = pd.to_datetime(df_resultado["DATE"])
    df_resultado["month"] = df_resultado["DATE"].dt.month
    df_resultado["day"] = df_resultado["DATE"].dt.day
    df_resultado["DECADIA"] = df_resultado["day"].apply(get_decada)
    df_resultado["ETAPA_FENOLOGICA"] = df_resultado.apply(
        lambda fila: get_etapa_fenologica(fila["month"], fila["day"]),
        axis=1,
    )
    df_resultado = df_resultado.drop(columns=["month", "day"])
    return df_resultado


def main() -> None:
    ruta_entrada = ROOT / "data" / "processed" / "clima_region_pampeana.csv"
    ruta_salida = ROOT / "data" / "processed" / "clima_region_pampeana_feno.csv"
    print(f"Leyendo datos desde: {ruta_entrada}")
    df_clima = pd.read_csv(ruta_entrada)
    print(f"Registros leídos: {len(df_clima)}")
    df_con_etapas = add_decada_y_etapa(df_clima)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    df_con_etapas.to_csv(ruta_salida, index=False)
    print(f"Archivo con etapas fenológicas guardado en: {ruta_salida}")
    print(f"Total de registros procesados: {len(df_con_etapas)}")


if __name__ == "__main__":
    main()

