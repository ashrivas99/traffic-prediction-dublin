import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from constants import COLORS, NEW_FEATURES, SITES_LIST

warnings.filterwarnings("ignore")


def plot_graphs(df):
    """
    Plots the graphs for the traffic over different parameters
    """
    plt.figure(figsize=(20, 4), facecolor="#627D78")
    time_series_plt = sns.lineplot(
        x=df["End_Time"],
        y=df["Average_volume_of_all_detectors"],
        ci=None,
        data=df,
        hue="Site",
        palette=COLORS,
    )
    time_series_plt.set_title("Traffic On Sites")
    time_series_plt.set_ylabel("Number of cars")
    time_series_plt.set_xlabel("Date")
    time_series_plt.legend(
        [SITES_LIST[i] for i in range(len(SITES_LIST))], loc="upper right"
    )
    time_series_plt.figure.savefig("Plots/time_series.png")

    df["Month"] = df["End_Time"].dt.month
    df["Date_no"] = df["End_Time"].dt.day
    df["Hour"] = df["End_Time"].dt.hour
    df["Day"] = df.End_Time.dt.strftime("%A")
    df.head()

    for feature in NEW_FEATURES:
        plt.figure(figsize=(20, 4), facecolor="#627D78")
        feature_plot = sns.lineplot(
            x=df[feature],
            y="Average_volume_of_all_detectors",
            ci=None,
            data=df,
            hue="Site",
            palette=COLORS,
        )
        feature_plot.set_xlabel(feature)
        feature_plot.legend(
            [SITES_LIST[i] for i in range(len(SITES_LIST))],
            bbox_to_anchor=(1.05, 1),
            loc="upper right",
        )
        feature_plot.figure.savefig(f"Plots/{feature}.png")

    plt.figure(figsize=(20, 4), facecolor="#627D78")
    cars_per_month = sns.countplot(data=df, x=df["Month"], palette=None)
    cars_per_month.set_title("Count Of Traffic On Sites Over Months")
    cars_per_month.set_ylabel("Number of cars")
    cars_per_month.set_xlabel("Month")
    cars_per_month.figure.savefig("Plots/count_of_traffic_on_sites_over_months.png")


if __name__ == "__main__":

    df = pd.read_csv("Datasets/preprocessed-dataset/preproc_classification_data.csv")
    df["End_Time"] = pd.to_datetime(df["End_Time"])
    df_graphs = df.copy()
    df_graphs = df[df["Site"].isin(SITES_LIST)]
    plot_graphs(df_graphs)
