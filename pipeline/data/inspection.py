import os

import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport

from pipeline.data import io


def save_data_inspection(
    Installed_Capacity_Germany,
    Prices_Europe,
    Realised_Supply_Germany,
    Realised_Demand_Germany,
    Weather_Data_Germany,
    CONF,
    data_type="raw",
    Weather_Data_Germany_2022=None,
):
    """
    Save data inspection reports for all data.
    Args
    ----
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
    Weather_Data_Germany_2022 : pd.DataFrame
        Weather data in Germany for 2022.
    CONF : DotMap
        Configuration object.
    data_type : str
        Type of data to load. Options are:
        - raw
        - preprocessed
    """

    if data_type == "raw":
        ROOT_DIR = CONF.data.raw_inspection_dir
    elif data_type == "preprocessed":
        ROOT_DIR = CONF.data.preprocessed_data_inspection_dir
    else:
        raise ValueError("data_type must be either 'raw' or 'preprocessed'")

    ProfileReport(
        Installed_Capacity_Germany,
        title=f"{data_type} Installed_Capacity_Germany",
        progress_bar=False,
    ).to_file(
        os.path.join(
            ROOT_DIR,
            "Installed_Capacity_Germany.html",
        )
    )

    ProfileReport(
        Prices_Europe, title=f"{data_type} Prices_Europe", progress_bar=False
    ).to_file(
        os.path.join(
            ROOT_DIR,
            "Prices_Europe.html",
        )
    )

    ProfileReport(
        Realised_Supply_Germany,
        title=f"{data_type} Realised_Supply_Germany",
        progress_bar=False,
    ).to_file(
        os.path.join(
            ROOT_DIR,
            "Realised_Supply_Germany.html",
        )
    )

    ProfileReport(
        Realised_Demand_Germany,
        title=f"{data_type} Realised_Demand_Germany",
        progress_bar=False,
    ).to_file(
        os.path.join(
            ROOT_DIR,
            "Realised_Demand_Germany.html",
        )
    )

    ProfileReport(
        Weather_Data_Germany,
        title=f"{data_type} Weather_Data_Germany",
        progress_bar=False,
        minimal=True,  # noqa: E501
    ).to_file(
        os.path.join(
            ROOT_DIR,
            "Weather_Data_Germany.html",
        )
    )

    if Weather_Data_Germany_2022 is not None:
        ProfileReport(
            Weather_Data_Germany_2022,
            title=f"{data_type} Weather_Data_Germany_2022",
            progress_bar=False,
            minimal=True,  # noqa: E501
        ).to_file(
            os.path.join(
                ROOT_DIR,
                "Weather_Data_Germany_2022.html",
            )
        )
    print(f"Saved {data_type} data inspection successfully.")


def date_range_and_resolution(
    df: pd.DataFrame, date_columns: list[str], expected_time_delta: str = None
):
    """
    Calculate the date range and resolution of the data.
    Args
    ----
    df : pd.DataFrame
        DataFrame to inspect.
    date_columns : list
        List of date columns in the DataFrame.
    expected_time_delta : str
        The expected time difference between consecutive dates, specified as a
        pandas timedelta string (e.g., '1H', '30T').
    """

    for date_column in date_columns:
        # Convert the column to datetime if not already
        df[date_column] = pd.to_datetime(df[date_column])

        # Calculate minimum and maximum date
        min_date = df[date_column].min()
        max_date = df[date_column].max()

        # Calculate differences between consecutive dates and find the most common
        date_diff = df[date_column].diff().dt.total_seconds()  # this is in seconds
        resolution = date_diff.mode()[0]  # The most common difference in seconds
        print(f"Min {date_column}: {min_date}")
        print(f"Max {date_column}: {max_date}")
        print(f"Resolution {date_column}: {pd.to_timedelta(resolution, unit='s')}")
        print(
            f"Total number of rows in the DataFrame: {len(df)}"
        )  # Print the total number of rows

        # Calculate and analyze differences between consecutive dates
        df = df.sort_values(by=date_column)
        date_diff = df[date_column].diff().dt.total_seconds()

        # Check for monotonic increase
        if not df[date_column].is_monotonic_increasing:
            print(f"Warning: {date_column} is not strictly monotonically increasing.")

        if expected_time_delta:
            expected_delta_seconds = pd.to_timedelta(
                expected_time_delta
            ).total_seconds()
            # Check if all time deltas match the expected time delta
            if not all(date_diff.dropna() == expected_delta_seconds):
                print(
                    f"Warning: Time delta in {date_column} not match {expected_delta_seconds}s."
                )
                diff = (
                    df["Date to"].diff().dt.total_seconds().dropna()
                    / expected_delta_seconds
                )
                print(np.where(diff != 1.0)[0])


def date_range_and_resolution_dfs(
    Installed_Capacity_Germany,
    Prices_Europe,
    Realised_Supply_Germany,
    Realised_Demand_Germany,
    Weather_Data_Germany,
    processed=False,
):
    print("# Installed Capacity Germany")
    date_range_and_resolution(Installed_Capacity_Germany, io.DATE_COLUMNS)
    print("\n")

    print("# Prices Europe")
    if processed:
        date_range_and_resolution(Prices_Europe, io.DATE_COLUMNS, "1h")
    else:
        date_range_and_resolution(Prices_Europe, io.DATE_COLUMNS, "1h")
    print("\n")

    print("# Realised Supply Germany")
    if processed:
        date_range_and_resolution(Realised_Supply_Germany, io.DATE_COLUMNS, "1h")
    else:
        date_range_and_resolution(Realised_Supply_Germany, io.DATE_COLUMNS, "15m")
    print("\n")

    print("# Realised Demand Germany")
    if processed:
        date_range_and_resolution(Realised_Demand_Germany, io.DATE_COLUMNS, "1h")
    else:
        date_range_and_resolution(Realised_Demand_Germany, io.DATE_COLUMNS, "15m")
    print("\n")

    print("# Weather Data Germany")
    date_range_and_resolution(Weather_Data_Germany, io.DATE_COLUMNS_WEATHER[1:])
