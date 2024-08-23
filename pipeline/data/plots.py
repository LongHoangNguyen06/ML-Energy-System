import math
import os

import matplotlib.pyplot as plt

from pipeline.data import io


def plot_df(
    df,
    df_name,
    CONF,
    date_col=io.DATE_COLUMNS[0],
    drop_date_cols=io.DATE_COLUMNS,
    processed_data=True,
    figsize=(30, 15),
):
    """Plot all columns of a DataFrame in separate
    subplots within a single figure and save the plot
    to a directory.
    Args:
        df (pd.DataFrame): DataFrame to plot.
        df_name (str): Name of the DataFrame.
        CONF (config.Config): Configuration object.
        date_cols (list, optional): List of date columns. Defaults to io.DATE_COLUMNS.
        drop_date_cols (list, optional): List of date columns to drop. Defaults to io.DATE_COLUMNS.
        processed_data (bool, optional): Whether the data is processed. Defaults to True.
    """
    num_columns = (
        len(df.columns) - 2
    )  # Exclude 'Date to' from the count if it's not plotted
    nrows = 4
    ncols = math.ceil(num_columns / nrows)

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=figsize, constrained_layout=True
    )
    fig.suptitle("Overview of All Columns")

    if processed_data:
        plot_dir = os.path.join(CONF.data.data_dir, "processed_data_figures")
    else:
        plot_dir = os.path.join(CONF.data.data_dir, "raw_data_figures")
    os.makedirs(plot_dir, exist_ok=True)

    axes = axes.flatten()

    # Extract 'Date from' for use as x-axis
    x_dates = df[date_col].to_numpy()

    # Plot each column except 'Date from' and 'Date to'
    for idx, column in enumerate(df.columns.drop(drop_date_cols)):
        y_data = df[column].to_numpy()
        ax = axes[idx]
        ax.plot(x_dates, y_data, label=column)
        ax.set_title(column, fontsize=10)
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend()

    # Hide unused subplots
    unused = len(df.columns) - 2
    for ax in axes[unused:]:
        ax.axis("off")

    plt.savefig(os.path.join(plot_dir, f"{df_name.replace('/', '-')}.png"))
    plt.close()
