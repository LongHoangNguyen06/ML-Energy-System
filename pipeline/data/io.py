import os
from datetime import datetime

import pandas as pd

CAPACITY_FORMAT = "%d.%m.%y"
STANDARD_FORMAT = "%d.%m.%y %H:%M"
WEATHER_FORMAT = "%Y-%m-%d %H:%M:%S"
DATE_COLUMNS = ["Date from", "Date to"]
DATE_COLUMNS_WEATHER = ["forecast_origin", "time"]


def load_data(CONF):
    """
    Load all data from the data folder.
    Args
    ----
    CONF : DotMap
        Configuration object.
    Returns
    -------
    Installed_Capacity_Germany : pd.DataFrame
        Installed Installed_Capacity_Germany of power plants in Germany.
    Prices_Europe : pd.DataFrame
        Electricity Prices_Europe in Europe.
    Realised_Supply_Germany : pd.DataFrame
        Realised_Supply_Germany of electricity in Germany.
    Realised_Demand_Germany : pd.DataFrame
        Realised_Demand_Germany of electricity in Germany.
    Weather_Data_Germany : pd.DataFrame
        Weather data in Germany.
    """
    ROOT_DIR = CONF.data.raw_data_dir

    path = os.path.join(ROOT_DIR, "Installed_Capacity_Germany.csv")
    Installed_Capacity_Germany = pd.read_csv(
        path,
        sep=";",
        thousands=".",
        decimal=",",
        na_values=["-"],
        parse_dates=DATE_COLUMNS,
        date_parser=lambda x: datetime.strptime(x.strip(), CAPACITY_FORMAT),
    )
    print(f"Loaded Installed_Capacity_Germany from '{path}' successfully.")

    path = os.path.join(ROOT_DIR, "Prices_Europe.csv")
    Prices_Europe = pd.read_csv(
        path,
        sep=";",
        thousands=".",
        decimal=",",
        na_values=["-"],
        parse_dates=DATE_COLUMNS,
        date_parser=lambda x: datetime.strptime(x.strip(), STANDARD_FORMAT),
    )
    print(f"Loaded Prices_Europe from '{path}' successfully.")

    path = os.path.join(ROOT_DIR, "Realised_Supply_Germany.csv")
    Realised_Supply_Germany = pd.read_csv(
        path,
        sep=";",
        thousands=".",
        decimal=",",
        na_values=["-"],
        parse_dates=DATE_COLUMNS,
        date_parser=lambda x: datetime.strptime(x.strip(), STANDARD_FORMAT),
    )
    print(f"Loaded Realised_Supply_Germany from '{path}' successfully.")

    path = os.path.join(ROOT_DIR, "Reaslised_Demand_Germany.csv")
    Realised_Demand_Germany = pd.read_csv(
        path,
        sep=";",
        thousands=".",
        decimal=",",
        na_values=["-"],
        parse_dates=DATE_COLUMNS,
        date_parser=lambda x: datetime.strptime(x.strip(), STANDARD_FORMAT),
    )
    print(f"Loaded Realised_Demand_Germany from '{path}' successfully.")

    path = os.path.join(ROOT_DIR, "Weather_Data_Germany.csv")
    Weather_Data_Germany = pd.read_csv(
        path,
        sep=",",
        na_values=["-"],
        parse_dates=DATE_COLUMNS_WEATHER,
        date_parser=lambda x: datetime.strptime(x.strip(), WEATHER_FORMAT),
    )
    print(f"Loaded Weather_Data_Germany from '{path}' successfully.")

    path = os.path.join(ROOT_DIR, "Weather_Data_Germany_2022.csv")
    Weather_Data_Germany_2022 = pd.read_csv(
        path,
        sep=",",
        na_values=["-"],
        parse_dates=["time"],
        date_parser=lambda x: datetime.strptime(x.strip(), WEATHER_FORMAT),
    )
    Weather_Data_Germany_2022["forecast_origin"] = pd.to_datetime(
        Weather_Data_Germany_2022["forecast_origin"]
    )
    print(f"Loaded Weather_Data_Germany_2022 from '{path}' successfully.")
    return (
        Installed_Capacity_Germany,
        Prices_Europe,
        Realised_Supply_Germany,
        Realised_Demand_Germany,
        Weather_Data_Germany,
        Weather_Data_Germany_2022,
    )


def load_final_df(CONF):
    return pd.read_csv(
        os.path.join(CONF.data.preprocessed_data_dir, "df.csv"),
        parse_dates=DATE_COLUMNS,
    )
