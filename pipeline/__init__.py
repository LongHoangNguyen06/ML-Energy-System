import os

import pandas as pd


def load_raw_data(CONF):
    """
    Load all data from the data folder.
    Args
    ----
    CONF : DotMap
        Configuration object.

    Returns
    -------
    capacity : pd.DataFrame
        Installed capacity of power plants in Germany.
    prices : pd.DataFrame
        Electricity prices in Europe.
    supply : pd.DataFrame
        Realised supply of electricity in Germany.
    demand : pd.DataFrame
        Realised demand of electricity in Germany.
    weather : pd.DataFrame
        Weather data in Germany.
    """
    ROOT_DIR = os.path.join(
        CONF.data_processing.data_dir, CONF.data_processing.raw_data_dir
    )
    capacity = pd.read_csv(
        f"{ROOT_DIR}/Installed_Capacity_Germany.csv",
        sep=";",
        thousands=".",
        decimal=",",
        na_values=["-"],
        parse_dates=["Date from", "Date to"],
    )
    prices = pd.read_csv(
        f"{ROOT_DIR}/Prices_Europe.csv",
        sep=";",
        decimal=",",
        na_values=["-"],
        parse_dates=["Date from", "Date to"],
    )
    supply = pd.read_csv(
        f"{ROOT_DIR}/Realised_Supply_Germany.csv",
        sep=";",
        thousands=".",
        decimal=",",
        na_values=["-"],
        parse_dates=["Date from", "Date to"],
    )
    demand = pd.read_csv(
        f"{ROOT_DIR}/Realised_Demand_Germany.csv",
        sep=";",
        thousands=".",
        decimal=",",
        na_values=["-"],
        parse_dates=["Date from", "Date to"],
    )
    weather = pd.read_csv(
        f"{ROOT_DIR}/Weather_Data_Germany.csv",
        sep=",",
        decimal=",",
        na_values=["-"],
        parse_dates=["forecast_origin", "time"],
    )
    return capacity, prices, supply, demand, weather
