import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_squared_error,
    roc_curve,
)
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier

from constants import SITES_LIST

warnings.filterwarnings("ignore")


def cityCenterSiteMetadata():
    df_site_info = pd.read_csv(
        "Datasets/traffic_volumes_site_metadata_jan_jun_2020/its_scats_sites_aug-2020.csv"
    )
    df_site_info = df_site_info[df_site_info["SiteID"].isin(SITES_LIST)]
    return df_site_info


def selected_sites_df(df):
    sites_df_dict = {}
    sites_df_list = list()
    df_selected_sites = df[df["Site"].isin(SITES_LIST)]

    for count, site in enumerate(SITES_LIST):
        site_id = df.Site == site
        df_site = df[site_id]
        sites_df_dict[count] = df_site
        sites_df_list.append(df_site)

    return df_selected_sites, sites_df_dict, sites_df_list


def visualize_site_data(timestamps_in_days, y_avg_vol_cars):
    # plot extracted data
    plt.rc("font", size=18)
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.figure(figsize=(8, 8), dpi=80)
    plt.scatter(timestamps_in_days, y_avg_vol_cars, color="blue", marker=".")
    plt.show()


def visualizeClassifications(df):
    df_class_one = df[df["classification_output"] == -1]
    df_class_two = df[df["classification_output"] == 1]
    precipitation_class_one = df_class_one.iloc[:, 4]
    avg_vol_cars_class_one = df_class_one.iloc[:, 3]
    precipitation_class_two = df_class_two.iloc[:, 4]
    avg_vol_cars_class_two = df_class_two.iloc[:, 3]
    plt.rc("font", size=18)
    plt.rc("font", size=18)
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.scatter(
        avg_vol_cars_class_one, precipitation_class_one, color="green", marker="."
    )
    plt.scatter(
        avg_vol_cars_class_two, precipitation_class_two, color="blue", marker="+"
    )
    plt.xlabel("avg_vol_cars")
    plt.ylabel("precipitation")
    plt.legend(["Target Value = -1", "Target Value = +1"], bbox_to_anchor=(1.00, 1.15))
    plt.show()


def plot_predictions(
    plot,
    y_pred,
    timestamps_in_days,
    y_avg_vol_cars,
    end_time_in_days,
    time_sampling_interval,
):
    plt.scatter(timestamps_in_days, y_avg_vol_cars, color="black")
    plt.scatter(end_time_in_days, y_pred, color="blue")
    plt.xlabel("time (days)")
    plt.ylabel("avg volume of cars")
    plt.legend(["training data", "predictions"], loc="upper right")
    day = math.floor(24 * 60 * 60 / time_sampling_interval)  # number of samples per day
    # plt.xlim((4 * 10, 4 * 10 + 4))
    plt.show()


def plot_3d_graph(df):
    precipitation = df.iloc[:, 4]
    avg_vol_cars = df.iloc[:, 3]
    end_time = pd.array((pd.DatetimeIndex(df.iloc[:, 0])).astype(np.int64)) / 1000000000
    timestamps_in_days = (end_time - end_time[0]) / 60 / 60 / 24
    plt.rc("font", size=18)
    plt.rcParams["figure.constrained_layout.use"] = True
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(precipitation, timestamps_in_days, avg_vol_cars)
    ax.set_title("Data Points")
    ax.set_xlabel("Precipitation", labelpad=20.0)
    ax.set_ylabel("Days", labelpad=20.0)
    ax.set_zlabel("Average Volume of Cars", labelpad=20.0)
    ax.legend(["Data points"], loc="best")
    plt.show()


def q_step_ahead_preds(
    q,
    dd,
    lag,
    plot,
    y_avg_vol_cars,
    y_precipitation,
    timestamps_in_days,
    time_sampling_interval,
):
    # dd is trend or seasonality; lag is number of points; q is step size
    stride = 1
    X_avg_vol_cars = y_avg_vol_cars[0 : y_avg_vol_cars.size - q - lag * dd : stride]
    X_precipitation = y_precipitation[0 : y_avg_vol_cars.size - q - lag * dd : stride]

    input_features_XX = np.column_stack((X_avg_vol_cars, X_precipitation))
    for i in range(1, lag):
        X_avg_vol_cars = y_avg_vol_cars[
            i * dd : y_avg_vol_cars.size - q - (lag - i) * dd : stride
        ]
        X_precipitation = y_precipitation[
            i * dd : y_avg_vol_cars.size - q - (lag - i) * dd : stride
        ]
        input_features_XX = np.column_stack(
            (input_features_XX, X_avg_vol_cars, X_precipitation)
        )

    yy = y_avg_vol_cars[lag * dd + q :: stride]
    end_time_in_days = timestamps_in_days[lag * dd + q :: stride]

    train, test = train_test_split(np.arange(0, yy.size), test_size=0.2)
    model = Ridge(fit_intercept=False).fit(input_features_XX[train], yy[train])
    print(model.intercept_, model.coef_)

    if plot:
        y_pred = model.predict(input_features_XX)
        plt.scatter(timestamps_in_days, y_avg_vol_cars, color="black")
        plt.scatter(end_time_in_days, y_pred, color="blue")
        plt.xlabel("time (days)")
        plt.ylabel("#volume of cars")
        plt.legend(["training data", "predictions"], loc="upper right")
        day = math.floor(
            24 * 60 * 60 / time_sampling_interval
        )  # number of samples per day
        plt.xlim(((lag * dd + q) / day, (lag * dd + q) / day + 2))
        plt.show()


def experiment_1(
    y_avg_vol_cars, y_precipitation, timestamps_in_days, time_sampling_interval
):
    # prediction using short-term trend
    plot = True
    q_step_ahead_preds(
        q=10,
        dd=1,
        lag=3,
        plot=plot,
        y_avg_vol_cars=y_avg_vol_cars,
        y_precipitation=y_precipitation,
        timestamps_in_days=timestamps_in_days,
        time_sampling_interval=time_sampling_interval,
    )

    # # prediction using daily seasonality
    d = math.floor(24 * 60 * 60 / time_sampling_interval)  # number of samples per day
    q_step_ahead_preds(
        q=d,
        dd=d,
        lag=3,
        plot=plot,
        y_avg_vol_cars=y_avg_vol_cars,
        y_precipitation=y_precipitation,
        timestamps_in_days=timestamps_in_days,
        time_sampling_interval=time_sampling_interval,
    )

    # # # prediction using weekly seasonality
    w = math.floor(
        7 * 24 * 60 * 60 / time_sampling_interval
    )  # number of samples per week
    q_step_ahead_preds(
        q=w,
        dd=w,
        lag=3,
        plot=plot,
        y_avg_vol_cars=y_avg_vol_cars,
        y_precipitation=y_precipitation,
        timestamps_in_days=timestamps_in_days,
        time_sampling_interval=time_sampling_interval,
    )


def featureEngineering(
    time_sampling_interval,
    y_avg_vol_cars,
    y_precipitation,
    y_classification,
    timestamps_in_days,
    q,
    lag,
    stride,
):
    d = math.floor(24 * 60 * 60 / time_sampling_interval)  # number of samples per day
    w = math.floor(
        7 * 24 * 60 * 60 / time_sampling_interval
    )  # number of samples per week
    len = y_avg_vol_cars.size - w - lag * w - q

    X_avg_vol_cars = y_avg_vol_cars[q : q + len : stride]
    X_precipitation = y_precipitation[q : q + len : stride]
    features_XX = np.column_stack((X_avg_vol_cars, X_precipitation))

    for i in range(1, lag):
        X_avg_vol_cars = y_avg_vol_cars[i * w + q : i * w + q + len : stride]
        X_precipitation = y_precipitation[i * w + q : i * w + q + len : stride]
        features_XX = np.column_stack((features_XX, X_avg_vol_cars, X_precipitation))

    for i in range(0, lag):
        X_avg_vol_cars = y_avg_vol_cars[(i * d) + q : (i * d) + q + len : stride]
        X_precipitation = y_precipitation[(i * d) + q : (i * d) + q + len : stride]
        features_XX = np.column_stack((features_XX, X_avg_vol_cars, X_precipitation))

    for i in range(0, lag):
        X_avg_vol_cars = y_avg_vol_cars[i : i + len : stride]
        X_precipitation = y_precipitation[i : i + len : stride]
        features_XX = np.column_stack((features_XX, X_avg_vol_cars, X_precipitation))

    yy_regression = y_avg_vol_cars[lag * w + w + q : lag * w + w + q + len : stride]
    yy_classification = y_classification[
        lag * w + w + q : lag * w + w + q + len : stride
    ]
    end_time_in_days = timestamps_in_days[
        lag * w + w + q : lag * w + w + q + len : stride
    ]

    return features_XX, yy_regression, yy_classification, end_time_in_days


def lag_cross_validation(
    time_sampling_interval,
    y_avg_vol_cars,
    y_precipitation,
    y_classification,
    timestamps_in_days,
):
    mean_error_log_reg = []
    std_error_log_reg = []
    mean_error_kNN = []
    std_error_kNN = []
    mean_error_decisionTree = []
    std_error_decisionTree = []
    mean_error_ridge = []
    std_error_ridge = []
    mean_error_lasso = []
    std_error_lasso = []

    q = 3
    stride = 1
    lag_range = list(range(1, 15))

    for lag in lag_range:
        (
            cross_val_XX,
            cross_val_yy_regression,
            cross_val_yy_classification,
            end_time_in_days,
        ) = featureEngineering(
            time_sampling_interval,
            y_avg_vol_cars,
            y_precipitation,
            y_classification,
            timestamps_in_days,
            q,
            lag,
            stride,
        )

        model_classification = LogisticRegression(
            penalty="l2", solver="lbfgs", C=1, max_iter=100000
        )
        model_kNN = KNeighborsClassifier(weights="uniform")
        model_decisionTree = DecisionTreeClassifier(max_depth=1)
        model_ridge = Ridge(max_iter=100000)
        model_lasso = Lasso(max_iter=100000)

        scores_log_reg = cross_val_score(
            model_classification,
            cross_val_XX,
            cross_val_yy_classification,
            cv=5,
            scoring="f1",
        )
        scores_kNN = cross_val_score(
            model_kNN, cross_val_XX, cross_val_yy_classification, cv=5, scoring="f1"
        )
        scores_decisionTree = cross_val_score(
            model_decisionTree,
            cross_val_XX,
            cross_val_yy_classification,
            cv=5,
            scoring="f1",
        )
        scores_ridge = cross_val_score(
            model_ridge, cross_val_XX, cross_val_yy_regression, cv=5, scoring="r2"
        )
        scores_lasso = cross_val_score(
            model_lasso, cross_val_XX, cross_val_yy_regression, cv=5, scoring="r2"
        )

        mean_error_log_reg.append(np.array(scores_log_reg).mean())
        std_error_log_reg.append(np.array(scores_log_reg).std())
        mean_error_kNN.append(np.array(scores_kNN).mean())
        std_error_kNN.append(np.array(scores_kNN).std())
        mean_error_decisionTree.append(np.array(scores_decisionTree).mean())
        std_error_decisionTree.append(np.array(scores_decisionTree).std())
        mean_error_lasso.append(np.array(scores_lasso).mean())
        std_error_lasso.append(np.array(scores_lasso).std())
        mean_error_ridge.append(np.array(scores_ridge).mean())
        std_error_ridge.append(np.array(scores_ridge).std())

    plt.rc("font", size=18)
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.errorbar(lag_range, mean_error_log_reg, yerr=std_error_log_reg, linewidth=3)
    plt.xlabel("lag")
    plt.ylabel("F1 Score")
    plt.title("Logistic Regression Cross Validation Results: Lag")
    plt.show()

    plt.rc("font", size=18)
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.errorbar(lag_range, mean_error_kNN, yerr=std_error_kNN, linewidth=3)
    plt.xlabel("lag")
    plt.ylabel("F1 Score")
    plt.title("kNN Cross Validation Results: Lag")
    plt.show()

    plt.rc("font", size=18)
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.errorbar(
        lag_range, mean_error_decisionTree, yerr=std_error_decisionTree, linewidth=3
    )
    plt.xlabel("lag")
    plt.ylabel("F1 Score")
    plt.title("Decision Trees Cross Validation Results: Lag")
    plt.show()

    plt.rc("font", size=18)
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.errorbar(lag_range, mean_error_lasso, yerr=std_error_lasso, linewidth=3)
    plt.xlabel("lag")
    plt.ylabel("r2")
    plt.title("Lasso Regression Cross Validation Results: Lag")
    plt.show()

    plt.rc("font", size=18)
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.errorbar(lag_range, mean_error_ridge, yerr=std_error_ridge, linewidth=3)
    plt.xlabel("lag")
    plt.ylabel("r2")
    plt.title("Ridge Regression Cross Validation Results: Lag")
    plt.show()


def LogRegPolynomialOrderCrossValidation(X, y_classification):
    mean_error = []
    std_error = []
    q_range = [1, 2, 3, 4, 5]
    for q in q_range:
        Xpoly = PolynomialFeatures(q).fit_transform(X)
        model = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=1000000)
        scores = cross_val_score(model, Xpoly, y_classification, cv=5, scoring="f1")
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())
    plt.rc("font", size=18)
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.errorbar(q_range, mean_error, yerr=std_error, linewidth=3)
    plt.xlabel("q")
    plt.ylabel("F1 Score")
    plt.title("Logistic Regression Cross Validation Results: Polynomial Feature q")
    plt.show()


# Using 5 fold cross validation to find the optimal C value for Lasso regression
def LassoPolynomialOrderCrossValidation(X, y_regression):
    kf = KFold(n_splits=5)
    mean_error = []
    std_error = []
    q_range = [1, 2, 3, 4, 5]
    for q in q_range:
        Xpoly = PolynomialFeatures(q).fit_transform(X)
        lasso_model = Lasso()
        temp = []
        for train, test in kf.split(X):
            lasso_model.fit(Xpoly[train], y_regression[train])
            ypred = lasso_model.predict(Xpoly[test])
            temp.append(mean_squared_error(y_regression[test], ypred))
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())
    plt.rc("font", size=18)
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.errorbar(q_range, mean_error, yerr=std_error, linewidth=3)
    plt.xlabel("q")
    plt.ylabel("Mean square error")
    plt.title("Lasso Regression Cross Validation Results: Polynomial Feature q")
    plt.show()


def RidgePolynomialOrderCrossValidation(X, y_regression):
    kf = KFold(n_splits=5)
    mean_error = []
    std_error = []
    q_range = [1, 2, 3, 4, 5]
    for q in q_range:
        Xpoly = PolynomialFeatures(q).fit_transform(X)
        ridge_model = Ridge()
        temp = []
        for train, test in kf.split(X):
            ridge_model.fit(Xpoly[train], y_regression[train])
            ypred = ridge_model.predict(Xpoly[test])
            temp.append(mean_squared_error(y_regression[test], ypred))
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())
    plt.rc("font", size=18)
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.errorbar(q_range, mean_error, yerr=std_error, linewidth=3)
    plt.xlabel("q")
    plt.ylabel("Mean square error")
    plt.title("Ridge Regression Cross Validation Results: Polynomial Feature q")
    plt.show()


def LogRegCvalueCrossValidation(X, y_classification):
    mean_error = []
    std_error = []
    Ci_range = [0.01, 0.1, 1, 5, 10, 25, 50, 100]
    for Ci in Ci_range:
        model = LogisticRegression(penalty="l2", solver="lbfgs", C=Ci, max_iter=1000000)
        scores = cross_val_score(
            model,
            X,
            y_classification,
            cv=5,
            scoring="f1",
        )
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())
    plt.rc("font", size=18)
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.errorbar(Ci_range, mean_error, yerr=std_error, linewidth=3)
    plt.xlabel("Ci")
    plt.ylabel("F1 Score")
    plt.title("Logistic Regression Cross Validation Results: Penalty Parameter Ci")
    plt.show()


# Using 5 fold cross validation to find the optimal C value for Lasso regression
def LassoRegressionCrossValidation(X, y_regression):
    mean_error = []
    std_error = []
    Ci_range = [0.1, 0.5, 1, 5, 10, 50, 100, 500]
    for Ci in Ci_range:
        lasso_model = Lasso(alpha=1 / (Ci))
        temp = []
        kf = KFold(n_splits=5)
        for train, test in kf.split(X):
            lasso_model.fit(X[train], y_regression[train])
            ypred = lasso_model.predict(X[test])
            temp.append(mean_squared_error(y_regression[test], ypred))
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())
    plt.rc("font", size=18)
    plt.rcParams["figure.constrained_layout.use"] = True
    mean_error = np.array(mean_error)
    std_error = np.array(std_error)
    plt.errorbar(Ci_range, mean_error, yerr=std_error)
    plt.xlabel("Ci")
    plt.ylabel("Mean square error")
    plt.title("Lasso Regression Cross Validation for wide range of Ci")
    plt.xlim((0, 200))
    plt.show()


# Using 5 fold cross validation to find the optimal C value for Ridge regression
def RidgeRegressionCrossValidation(X, y_regression):
    mean_error = []
    std_error = []
    Ci_range = [0.1, 0.5, 1, 5, 10, 50, 100, 500]
    for Ci in Ci_range:
        ridge_model = Lasso(alpha=1 / (Ci), max_iter=100000)
        temp = []
        kf = KFold(n_splits=5)
        for train, test in kf.split(X):
            ridge_model.fit(X[train], y_regression[train])
            ypred = ridge_model.predict(X[test])
            temp.append(mean_squared_error(y_regression[test], ypred))
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())
    plt.rc("font", size=18)
    plt.rcParams["figure.constrained_layout.use"] = True
    mean_error = np.array(mean_error)
    std_error = np.array(std_error)
    plt.errorbar(Ci_range, mean_error, yerr=std_error)
    plt.xlabel("Ci")
    plt.ylabel("Mean square error")
    plt.title("Ridge Regression Cross Validation for wide range of Ci")
    plt.xlim((0, 200))
    plt.show()


def kNN_k_value_finder(X, y_classification):
    mean_error = []
    std_error = []
    k_range = list(range(1, 31))
    for k in k_range:
        model = KNeighborsClassifier(n_neighbors=k, weights="uniform")
        scores = cross_val_score(model, X, y_classification, cv=5, scoring="f1")
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())
    plt.rc("font", size=18)
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.errorbar(k_range, mean_error, yerr=std_error, linewidth=3)
    plt.xlabel("k")
    plt.ylabel("F1 Score")
    plt.title("kNN Cross Validation Results: k Value")
    plt.show()


def decision_tree_depth_value_finder(X, y_classification):
    mean_error = []
    std_error = []
    depth_range = list(range(1, 31))
    for depth in depth_range:
        model = DecisionTreeClassifier(max_depth=depth)
        scores = cross_val_score(model, X, y_classification, cv=5, scoring="f1")
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())
    plt.rc("font", size=18)
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.errorbar(depth_range, mean_error, yerr=std_error, linewidth=3)
    plt.xlabel("max_depth")
    plt.ylabel("F1 Score")
    plt.title("Decision Tree Classifier Cross Validation Results: Tree Depth Value")
    plt.show()


def main():
    df = pd.read_csv("Datasets/preprocessed-dataset/preproc_classification_data.csv")
    df_site_info_ccity = cityCenterSiteMetadata()
    df, selected_sites_df_dict, selected_sites_df_list = selected_sites_df(df)

    SITE_1 = df.Site == 628
    df_site_1 = df[SITE_1]
    df_site_1.set_index("End_Time")

    # convert date/time to unix timestamp in sec
    all_timestamps_in_sec = (
        pd.array((pd.DatetimeIndex(df_site_1.iloc[:, 0])).astype(np.int64)) / 1000000000
    )
    time_sampling_interval = all_timestamps_in_sec[1] - all_timestamps_in_sec[0]
    print("data sampling interval is %d secs" % time_sampling_interval)

    timestamps_in_days = (
        (all_timestamps_in_sec - all_timestamps_in_sec[0]) / 60 / 60 / 24
    )  # convert timestamp to days
    y_avg_vol_cars = np.extract(all_timestamps_in_sec, df_site_1.iloc[:, 3]).astype(
        np.int64
    )
    y_precipitation = np.extract(all_timestamps_in_sec, df_site_1.iloc[:, 4]).astype(
        np.int64
    )
    y_classification = np.extract(all_timestamps_in_sec, df_site_1.iloc[:, 6]).astype(
        np.int64
    )

    visualize_site_data(timestamps_in_days, y_avg_vol_cars)
    plot_3d_graph(df_site_1)
    experiment_1(
        y_avg_vol_cars, y_precipitation, timestamps_in_days, time_sampling_interval
    )
    lag_cross_validation(
        time_sampling_interval,
        y_avg_vol_cars,
        y_precipitation,
        y_classification,
        timestamps_in_days,
    )

    # Taking 4 as most optimal value
    lag_value = int(
        input("Please choose the desired Lag value for your time series models:    ")
    )
    # lag_value = 4

    # putting it together
    q = 10
    lag = lag_value
    stride = 1
    XX, yy_regression, yy_classification, end_time_in_days = featureEngineering(
        time_sampling_interval,
        y_avg_vol_cars,
        y_precipitation,
        y_classification,
        timestamps_in_days,
        q,
        lag,
        stride,
    )

    scaler = MinMaxScaler()
    XX_scaled = scaler.fit_transform(XX)

    LogRegPolynomialOrderCrossValidation(XX_scaled, yy_classification)
    q_log_reg = int(
        input(
            "Please choose the desired polynomial order for Logistic Regression 'q' value:    "
        )
    )

    if q_log_reg == 1:
        XX_poly_log_reg = XX
    else:
        XX_poly_log_reg = PolynomialFeatures(q_log_reg).fit_transform(XX_scaled)

    LassoPolynomialOrderCrossValidation(XX_scaled, yy_regression)
    q_lasso_reg = int(
        input(
            "Please choose the desired polynomial order for Lasso Regression 'q' value:    "
        )
    )

    if q_lasso_reg == 1:
        XX_poly_lasso_reg = XX
    else:
        XX_poly_lasso_reg = PolynomialFeatures(q_lasso_reg).fit_transform(XX_scaled)

    RidgePolynomialOrderCrossValidation(XX_scaled, yy_regression)
    q_ridge_reg = int(
        input(
            "Please choose the desired polynomial order for Ridge Regression 'q' value:    "
        )
    )

    if q_ridge_reg == 1:
        XX_poly_ridge_reg = XX
    else:
        XX_poly_ridge_reg = PolynomialFeatures(q_ridge_reg).fit_transform(XX_scaled)

    LogRegCvalueCrossValidation(XX_poly_log_reg, yy_classification)
    C_value_log_reg = int(
        input("Please choose the desired 'C' value for the Logistic Regression:    ")
    )

    LassoRegressionCrossValidation(XX_poly_lasso_reg, yy_regression)
    C_value_lasso = int(
        input("Please choose the desired 'C' value for the Lasso Regression model:    ")
    )

    RidgeRegressionCrossValidation(XX_poly_ridge_reg, yy_regression)
    C_value_ridge = int(
        input("Please choose the desired 'C' value for the Ridge Regression model:    ")
    )

    # k-value 11
    kNN_k_value_finder(XX, yy_classification)
    k_value = int(input("Please choose the desired 'k' value for the kNN model:    "))

    decision_tree_depth_value_finder(XX, yy_classification)
    decision_tree_depth = int(
        input(
            "Please choose the desired 'decision_tree_depth' value for the Decision Tree model:    "
        )
    )

    train, test = train_test_split(np.arange(0, yy_regression.size), test_size=0.2)

    model_ridge = Ridge(fit_intercept=False, alpha=1 / (2 * C_value_ridge)).fit(
        XX_poly_ridge_reg[train], yy_regression[train]
    )
    print(model_ridge.intercept_, model_ridge.coef_)
    y_pred_ridge = model_ridge.predict(XX_poly_ridge_reg)
    y_pred_ridge_test = model_ridge.predict(XX_poly_ridge_reg[test])
    plot_predictions(
        True,
        y_pred_ridge,
        timestamps_in_days,
        y_avg_vol_cars,
        end_time_in_days,
        time_sampling_interval,
    )

    model_lasso = Lasso(fit_intercept=False, alpha=1 / (C_value_lasso)).fit(
        XX_poly_lasso_reg[train], yy_regression[train]
    )
    print(model_lasso.intercept_, model_lasso.coef_)
    y_pred_lasso = model_lasso.predict(XX_poly_lasso_reg)
    y_pred_lasso_test = model_lasso.predict(XX_poly_lasso_reg[test])
    plot_predictions(
        True,
        y_pred_lasso,
        timestamps_in_days,
        y_avg_vol_cars,
        end_time_in_days,
        time_sampling_interval,
    )

    print("Dummy Regressor")
    dummy_regressor = DummyRegressor(strategy="median")
    dummy_regressor.fit(XX_poly_lasso_reg[train], yy_regression[train])
    ydummy_regressor_predictions = dummy_regressor.predict(XX_poly_lasso_reg[test])
    dummy_regressor.score(XX_poly_lasso_reg[test], yy_regression[test])
    plot_predictions(
        True,
        dummy_regressor.predict(XX_poly_lasso_reg),
        timestamps_in_days,
        y_avg_vol_cars,
        end_time_in_days,
        time_sampling_interval,
    )

    X_train_log_reg, X_test_log_reg, y_train_log_reg, y_test_log_reg = train_test_split(
        XX_poly_log_reg, yy_classification, test_size=0.2
    )
    model_classification_log_reg = LogisticRegression(
        penalty="l2", solver="lbfgs", C=C_value_log_reg, max_iter=10000
    )
    model_classification_log_reg.fit(X_train_log_reg, y_train_log_reg)
    y_pred_log_reg = model_classification_log_reg.predict(X_test_log_reg)
    print(model_classification_log_reg.intercept_, model_classification_log_reg.coef_)

    log_reg_confusion_matrix = confusion_matrix(y_test_log_reg, y_pred_log_reg)
    log_reg_classification_report = classification_report(
        y_test_log_reg, y_pred_log_reg
    )

    print("LOGG REGG")
    print(log_reg_confusion_matrix)
    print(log_reg_classification_report)

    X_train, X_test, y_train, y_test = train_test_split(
        XX, yy_classification, test_size=0.2
    )
    model_classification_kNN = KNeighborsClassifier(
        n_neighbors=k_value, weights="uniform"
    )
    model_classification_kNN.fit(X_train, y_train)
    y_pred_kNN = model_classification_kNN.predict(X_test)
    kNN_confusion_matrix = confusion_matrix(y_test, y_pred_kNN)
    kNN_classification_report = classification_report(y_test, y_pred_kNN)
    print("KNN")
    print(kNN_confusion_matrix)
    print(kNN_classification_report)

    model_classification_DecisionTree = DecisionTreeClassifier(
        max_depth=decision_tree_depth
    )
    model_classification_DecisionTree.fit(X_train, y_train)
    y_pred_DecisionTreeClassifier = model_classification_DecisionTree.predict(X_test)
    DecisionTree_confusion_matrix = confusion_matrix(
        y_test, y_pred_DecisionTreeClassifier
    )
    DecisionTree_classification_report = classification_report(
        y_test, y_pred_DecisionTreeClassifier
    )
    print("DecisionTreeClassifier")
    print(DecisionTree_confusion_matrix)
    print(DecisionTree_classification_report)

    dummy_most_frequent = DummyClassifier(strategy="most_frequent")
    dummy_most_frequent.fit(X_train, y_train)
    ydummy_most_frequent_predictions = dummy_most_frequent.predict(X_test)
    dummy_most_frequent_confusion_matrix = confusion_matrix(
        y_test, ydummy_most_frequent_predictions
    )
    dummy_most_frequent_classification_report = classification_report(
        y_test, ydummy_most_frequent_predictions
    )
    print(dummy_most_frequent_confusion_matrix)
    print(dummy_most_frequent_classification_report)

    dummy_uniform = DummyClassifier(strategy="uniform")
    dummy_uniform.fit(X_train, y_train)
    ydummy_uniform_predictions = dummy_uniform.predict(X_test)
    dummy_uniform_confusion_matrix = confusion_matrix(
        y_test, ydummy_uniform_predictions
    )
    dummy_uniform_classification_report = classification_report(
        y_test, ydummy_uniform_predictions
    )
    print(dummy_uniform_confusion_matrix)
    print(dummy_uniform_classification_report)

    # __________________________________________________________ROC Curve__________________________________________________________________

    fpr_log_reg, tpr_log_reg, _ = roc_curve(
        y_test_log_reg, model_classification_log_reg.decision_function(X_test_log_reg)
    )

    y_scores_kNN = model_classification_kNN.predict_proba(X_test)
    fpr_kNN, tpr_kNN, _ = roc_curve(y_test, y_scores_kNN[:, 1])

    y_scores_DecisionTree = model_classification_DecisionTree.predict_proba(X_test)
    fpr_DecisionTree, tpr_DecisionTree, _ = roc_curve(
        y_test, y_scores_DecisionTree[:, 1]
    )

    y_scores_dummy_uniform = dummy_uniform.predict_proba(X_test)
    fpr_dummy_uniform, tpr_dummy_uniform, _ = roc_curve(
        y_test, y_scores_dummy_uniform[:, 1]
    )

    y_scores_dummy_most_frequent = dummy_most_frequent.predict_proba(X_test)
    fpr_dummy_most_frequent, fpr_dummy_most_frequent, _ = roc_curve(
        y_test, y_scores_dummy_most_frequent[:, 1]
    )

    plt.plot(fpr_log_reg, tpr_log_reg, color="r")
    plt.plot(fpr_kNN, tpr_kNN, color="g")
    plt.plot(fpr_DecisionTree, tpr_DecisionTree, color="magenta")
    plt.plot(fpr_dummy_uniform, tpr_dummy_uniform, color="b", linestyle="-.")
    plt.plot(
        fpr_dummy_most_frequent, fpr_dummy_most_frequent, color="black", linestyle=":"
    )
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend(
        [
            "Logistic Regression",
            "kNN Classifier",
            "Decision Tree Classifier",
            "Dummy Classifier(uniform)",
            "Dummy Classifier most_frequent",
        ],
        loc="lower right",
    )
    plt.title("ROC Curve for various trained models")
    plt.savefig("Plots/ROC_Curve_final.png")
    plt.show()
    ridge_mse = mean_squared_error(
        yy_regression[test],
        y_pred_ridge_test,
        multioutput="uniform_average",
        squared=True,
    )

    lasso_mse = mean_squared_error(
        yy_regression[test],
        y_pred_lasso_test,
        multioutput="uniform_average",
        squared=True,
    )

    dummy_regressor_mse = mean_squared_error(
        yy_regression[test],
        ydummy_regressor_predictions,
        multioutput="uniform_average",
        squared=True,
    )
    print(f"Ridge MSE: {ridge_mse}")
    print(f"Lasso MSE: {lasso_mse}")
    print(f"Dummy Regressor MSE: {dummy_regressor_mse}")


if __name__ == "__main__":
    main()
