import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

from pipeline.data import io


class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self, constant=1):
        self.constant = constant

    def fit(self, X, y=None):
        # Custom logic for fitting, if necessary
        return self

    def transform(self, X):
        return X / self.constant

    def inverse_transform(self, X):
        return X * self.constant


def process_na_values(data: pd.DataFrame, CONF):
    """
    Process missing values in the data.
    Args
    ----
    data : pd.DataFrame
        Data to process.
    CONF : DotMap
        Configuration object.

    Returns
    -------
    data : pd.DataFrame
        Data with missing values processed.
    """
    if CONF.data.na_values == "drop_rows":
        data = data.dropna(axis=0, how="any")
    elif CONF.data.na_values == "drop_columns":
        data = data.dropna(axis=1, how="any")
    elif CONF.data.na_values == "fillna":
        data = data.fillna(
            0
        )  # TODO: Implement a more sophisticated way to fill missing values
    assert not data.isnull().values.any()
    return data


# Define a function to split the data
def split_data(df: pd.DataFrame, column_name: str):
    """
    Split the data into train, validation, and test sets based
    on the year in the specified column.
    Args
    ----
    df : pd.DataFrame
        Data to split.
    column_name : str
        Name of the column containing the date information.
    Returns
    -------
    df : pd.DataFrame
        Data with additional columns indicating the split.
    """
    # Create binary columns for train, val, and test
    df["train"] = df[column_name].dt.year.isin([2019, 2020])
    df["val"] = df[column_name].dt.year == 2021
    df["test"] = df[column_name].dt.year > 2021

    return df


def normalize_data(df, ignore_features=list[str], constant=None):
    """
    Scale the data using StandardScaler or custom constants.

    Args:
    ----
    df : pd.DataFrame
        DataFrame to scale.
    ignore_features : list
        List of features to ignore during scaling.
    constant : float, optional
        customized normalizing constants for each feature.

    Returns:
    -------
    df : pd.DataFrame
        Scaled DataFrame.
    scalers : dict
        Dictionary containing the scalers used for each feature.
    """

    def apply_scaling(df, feature):
        if constant:
            scaler = CustomScaler(constant=constant)
        else:
            scaler = StandardScaler()
        train_df = df[df["train"]]
        train_feature = train_df[[feature]]
        scaler.fit(train_feature)
        scaled_features = scaler.transform(df[[feature]])
        df[feature] = scaled_features
        return df, scaler

    # Apply scaling to all datasets
    flag_colums = ["train", "val", "test", "index"]
    scalers = dict()
    feature_columns = df.columns.tolist()
    for feature in feature_columns:
        if feature not in (ignore_features + flag_colums):
            df, scaler = apply_scaling(df, feature)
            scalers[feature] = scaler
    return df, scalers


def aggregate_weather_data(df, column, copy=True):
    """
    Aggregate weather data by the specified column.
    Args
    ----
    df : pd.DataFrame
        DataFrame containing weather data.
    column : str
        Column to group by.
    copy : bool
        Whether to copy the DataFrame before processing.
    Returns
    -------
    df : pd.DataFrame
        DataFrame with weather data aggregated by the specified column.
    """
    if copy:
        df = df.copy()
    aggregations = {
        "cdir": ["min", "max", "mean"],
        "z": ["min", "max", "mean"],
        "msl": ["min", "max", "mean"],
        "blh": ["min", "max", "mean"],
        "tcc": ["min", "max", "mean"],
        "u10": ["min", "max", "mean"],
        "v10": ["min", "max", "mean"],
        "t2m": ["min", "max", "mean"],
        "ssr": ["min", "max", "mean"],
        "tsr": ["min", "max", "mean"],
        "sund": ["min", "max", "mean"],
        "tp": ["min", "max", "mean"],
        "fsr": ["min", "max", "mean"],
        "u100": ["min", "max", "mean"],
        "v100": ["min", "max", "mean"],
    }
    df = df.groupby(column).agg(aggregations)
    df.columns = ["_".join(col).strip() for col in df.columns.values]
    df.reset_index(inplace=True)
    return df


def patch_time_saving(df: pd.DataFrame):
    """
    Patch the data to removing time shifting problems...
    Args
    ----
    df : pd.DataFrame
        DataFrame to patch.
    Returns
    -------
    df : pd.DataFrame
        Patched DataFrame.
    """

    def prepare_data(df):
        before = len(df)
        # Drop duplicates and sort to prevent issues in dropping non-ascending rows
        df = df.drop_duplicates().sort_values(by=["Date from", "Date to"])
        # Drop rows where 'Date from' or 'Date to' is not greater than the previous row
        df = df[df["Date from"].diff().dt.total_seconds() > 0]
        df = df[df["Date to"].diff().dt.total_seconds() > 0]
        after = len(df)
        print(f"Removed {before - after} rows")
        return df

    def create_full_time_df(df, start_date, end_date, time_delta):
        """
        Creates a DataFrame with a complete time range from start_date to end_date, initializing columns
        based on an input DataFrame.

        Args:
        df : pd.DataFrame
            The input DataFrame from which to derive the column structure.
        start_date : str
            The start date of the time range.
        end_date : str
            The end date of the fort columns.
        time_delta : str
            The frequency of the time range (e.g., '1H' for one hour).

        Returns:
        pd.DataFrame
            A new DataFrame with the specified time range and columns initialized.
        """
        # Create the date range for the full time DataFrame
        date_range = pd.date_range(start=start_date, end=end_date, freq=time_delta)

        # Initialize the DataFrame with date columns
        df_full = pd.DataFrame(
            {"Date from": date_range[:-1], "Date to": date_range[1:]}
        )

        # Initialize other columns based on the input DataFrame
        for column in df.columns:
            if column not in ["Date from", "Date to"]:
                df_full[column] = (
                    pd.NA
                )  # Initialize with pandas NA for appropriate handling of missing types
        return df_full

    def merge_and_fill(df1, df2):
        """
        Merges two DataFrames based on 'Date from' and 'Date to' columns and applies forward
        filling to handle missing values.
        Ensures that `df2` maintains its structure, only updating with data from `df1` where dates match.

        Args:
        df1 (pd.DataFrame): Source DataFrame with observed data.
        df2 (pd.DataFrame): Target DataFrame with a predefined time interval and initialized columns.

        Returns:
        pd.DataFrame: Updated version of `df2` with missing values filled from `df1`.

        Details:
        - Performs a left join to keep all rows in `df2` and updates its data from `df1` where
        their 'Date from' and 'Date to' match.
        - Forward fills missing values in `df2` to ensure no data gaps.
        """
        df1 = df1.copy()
        df2 = df2.copy()
        df1["Date from"] = pd.to_datetime(df1["Date from"])
        df1["Date to"] = pd.to_datetime(df1["Date to"])
        df2["Date from"] = pd.to_datetime(df2["Date from"])
        df2["Date to"] = pd.to_datetime(df2["Date to"])

        # Merge with an indicator to track where data came from
        merged_df = df2.merge(
            df1, on=["Date from", "Date to"], how="left", suffixes=("", "_from_df1")
        )

        # Update df2 columns with df1 where available
        for col in df1.columns:
            if col not in ["Date from", "Date to"]:  # Avoid key columns
                # Use data from df1 if available, otherwise retain df2 data
                merged_df[col] = merged_df[col + "_from_df1"]
                merged_df.drop(col + "_from_df1", axis=1, inplace=True)
        merged_df.ffill(inplace=True)
        merged_df.bfill(inplace=True)
        return merged_df

    ########################################################
    # Patch the data to removing time shifting problems...
    ########################################################
    df1 = prepare_data(df)
    df2 = create_full_time_df(df, "2019-01-01 00:00:00", "2023-01-01 00:00:00", "1H")
    df = merge_and_fill(df1, df2)

    df = df[df["Date to"] != df["Date to"].max()]
    return df


def german2greenwich(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the German time to Greenwich time.
    Args
    ----
    df : pd.DataFrame
        DataFrame containing the German time.
    Returns
    -------
    df : pd.DataFrame
        DataFrame with the Greenwich time.
    """
    df["Date from"] = (
        df["Date from"]
        .dt.tz_localize("Europe/Berlin", ambiguous="NaT")
        .dt.tz_convert("GMT")
        .dt.tz_localize(None)
    )
    df["Date to"] = (
        df["Date to"]
        .dt.tz_localize("Europe/Berlin", ambiguous="NaT")
        .dt.tz_convert("GMT")
        .dt.tz_localize(None)
    )
    return df


def split_dfs(
    Processed_Installed_Capacity_Germany: pd.DataFrame,
    Processed_Prices_Europe: pd.DataFrame,
    Processed_Realised_Supply_Germany: pd.DataFrame,
    Processed_Realised_Demand_Germany: pd.DataFrame,
    Processed_Weather_Data_Germany: pd.DataFrame,
) -> None:
    Processed_Installed_Capacity_Germany = split_data(
        df=Processed_Installed_Capacity_Germany, column_name=io.DATE_COLUMNS[-1]
    )
    Processed_Prices_Europe = split_data(
        df=Processed_Prices_Europe, column_name=io.DATE_COLUMNS[-1]
    )
    Processed_Realised_Supply_Germany = split_data(
        df=Processed_Realised_Supply_Germany, column_name=io.DATE_COLUMNS[-1]
    )
    Processed_Realised_Demand_Germany = split_data(
        df=Processed_Realised_Demand_Germany, column_name=io.DATE_COLUMNS[-1]
    )
    Processed_Weather_Data_Germany = split_data(
        df=Processed_Weather_Data_Germany, column_name=io.DATE_COLUMNS_WEATHER[0]
    )
    return (
        Processed_Installed_Capacity_Germany,
        Processed_Prices_Europe,
        Processed_Realised_Supply_Germany,
        Processed_Realised_Demand_Germany,
        Processed_Weather_Data_Germany,
    )


def remove_train_val_test_cols(df) -> pd.DataFrame:
    return df.drop(columns=["train", "val", "test"], axis=1)
