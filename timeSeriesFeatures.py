import pandas as pd
import numpy as np
import math, sys
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# High Traffic Volume Sites: 628,305,2; Medium Traffic Sites: 48,36,420,3; Low Traffic Sites : 796, 1,402, 665
SITES_LIST = [628, 305, 2, 48, 36, 420, 3, 796, 1, 402, 665]


def cityCenterSiteMetadata():
    df_site_info = pd.read_csv('Datasets/traffic_volumes_site_metadata_jan_jun_2020/its_scats_sites_aug-2020.csv')
    df_site_info = df_site_info[df_site_info['SiteID'].isin(SITES_LIST)]
    return df_site_info


def selected_sites_df(df):
    sites_df_dict = {}
    sites_df_list = list()
    df_selected_sites = df[df['Site'].isin(SITES_LIST)]

    for count, site in enumerate(SITES_LIST):
        site_id = df.Site == site
        df_site = df[site_id]
        sites_df_dict[count] = df_site
        sites_df_list.append(df_site)

    return df_selected_sites, sites_df_dict, sites_df_list


def visualize_site_data(timestamps_in_days, y_avg_vol_cars):
    # plot extracted data
    plt.rc('font', size=18);
    plt.rcParams['figure.constrained_layout.use'] = True
    plt.figure(figsize=(8, 8), dpi=80)
    plt.scatter(timestamps_in_days, y_avg_vol_cars, color='blue', marker='.');
    plt.show()

def visualizeClassifications(df):
    df_class_one = df[df['classification_output'] == -1]
    df_class_two = df[df['classification_output'] == 1]
    precipitation_class_one = df_class_one.iloc[:, 4]
    avg_vol_cars_class_one = df_class_one.iloc[:, 3]
    precipitation_class_two = df_class_two.iloc[:, 4]
    avg_vol_cars_class_two = df_class_two.iloc[:, 3]
    plt.rc('font', size=18)
    plt.rc('font', size=18)
    plt.rcParams['figure.constrained_layout.use'] = True
    plt.scatter(avg_vol_cars_class_one, precipitation_class_one,  color='green', marker='.')
    plt.scatter( avg_vol_cars_class_two, precipitation_class_two, color='blue', marker='+')
    plt.xlabel('precipitation')
    plt.ylabel('avg_vol_cars')
    plt.legend(['Target Value = -1', 'Target Value = +1'],
               bbox_to_anchor=(1.00, 1.15))
    plt.show()

def plot_predictions(plot, y_pred, timestamps_in_days, y_avg_vol_cars, end_time_in_days, time_sampling_interval):
    plt.scatter(timestamps_in_days, y_avg_vol_cars, color='black');
    plt.scatter(end_time_in_days, y_pred, color='blue')
    plt.xlabel("time (days)");
    plt.ylabel("#bikes")
    plt.legend(["training data", "predictions"], loc='upper right')
    day = math.floor(24 * 60 * 60 / time_sampling_interval)  # number of samples per day
    # plt.xlim((4 * 10, 4 * 10 + 4))
    plt.show()


def q_step_ahead_preds(q, dd, lag, plot, y_avg_vol_cars, y_precipitation, timestamps_in_days, time_sampling_interval):
    # dd is trend or seasonality; lag is number of points; q is step size
    stride=1
    X_avg_vol_cars=y_avg_vol_cars[0:y_avg_vol_cars.size-q-lag*dd:stride]
    X_precipitation=y_precipitation[0:y_avg_vol_cars.size-q-lag*dd:stride]
    
    input_features_XX = np.column_stack((X_avg_vol_cars,X_precipitation))
    for i in range(1,lag):
        X_avg_vol_cars = y_avg_vol_cars[i* dd:y_avg_vol_cars.size-q- (lag-i)*dd:stride]
        X_precipitation = y_precipitation[i* dd:y_avg_vol_cars.size-q- (lag-i)*dd:stride]
        input_features_XX = np.column_stack((input_features_XX, X_avg_vol_cars, X_precipitation))
    
    yy = y_avg_vol_cars[lag* dd+q::stride]
    end_time_in_days = timestamps_in_days[lag* dd+q::stride]
    
    train, test = train_test_split(np.arange(0,yy.size),test_size=0.2)
    print(train)
    print(test)
    model = Ridge(fit_intercept=False).fit(input_features_XX[train], yy[train])
    # model = KNeighborsRegressor(n_neighbors =10).fit(input_features_XX[train], yy[train]) 
    print(model.intercept_, model.coef_)

    if plot:
        y_pred = model.predict(input_features_XX)
        plt.scatter(timestamps_in_days, y_avg_vol_cars, color='black')
        plt.scatter(end_time_in_days, y_pred, color='blue')
        plt.xlabel("time (days)")
        plt.ylabel("#volume of cars")
        plt.legend(["training data","predictions"],loc='upper right')
        day=math.floor(24*60*60/time_sampling_interval) # number of samples per day
        plt.xlim(((lag*dd+q)/day,(lag*dd+q)/day+2))
        plt.show()

def experiment_1(y_avg_vol_cars, y_precipitation, timestamps_in_days, time_sampling_interval):
    # prediction using short-term trend
    plot = True
    q_step_ahead_preds(q=10, dd=1, lag=3, plot=plot, y_avg_vol_cars=y_avg_vol_cars, y_precipitation= y_precipitation,
                       timestamps_in_days=timestamps_in_days, time_sampling_interval=time_sampling_interval)

    # # prediction using daily seasonality
    d = math.floor(24 * 60 * 60 / time_sampling_interval)  # number of samples per day
    q_step_ahead_preds(q=d, dd=d, lag=3, plot=plot, y_avg_vol_cars=y_avg_vol_cars, y_precipitation= y_precipitation,
                       timestamps_in_days= timestamps_in_days, time_sampling_interval=time_sampling_interval)

    # # # prediction using weekly seasonality
    w = math.floor(7 * 24 * 60 * 60 / time_sampling_interval)  # number of samples per week
    q_step_ahead_preds(q=w, dd=w, lag=3, plot=plot, y_avg_vol_cars=y_avg_vol_cars, y_precipitation=y_precipitation,
                       timestamps_in_days=timestamps_in_days, time_sampling_interval=time_sampling_interval)


def featureEngineering(time_sampling_interval, y_avg_vol_cars, y_precipitation, y_classification, timestamps_in_days, q, lag, stride):
    d = math.floor(24 * 60 * 60 / time_sampling_interval)  # number of samples per day
    w = math.floor(7 * 24 * 60 * 60 / time_sampling_interval)  # number of samples per week
    len = y_avg_vol_cars.size-w-lag*w-q
    
    X_avg_vol_cars = y_avg_vol_cars[q:q+len:stride]
    X_precipitation = y_precipitation[q:q+len:stride]
    features_XX = np.column_stack((X_avg_vol_cars, X_precipitation))

    for i in range(1, lag):
        X_avg_vol_cars= y_avg_vol_cars[i*w+q:i*w+q+len:stride]
        X_precipitation= y_precipitation[i*w+q:i*w+q+len:stride]
        features_XX = np.column_stack((features_XX, X_avg_vol_cars, X_precipitation ))
   
    for i in range(0, lag):
        X_avg_vol_cars = y_avg_vol_cars[(i*d)+q : (i*d)+q+len : stride]
        X_precipitation = y_precipitation[(i*d)+q : (i*d)+q+len : stride]
        features_XX = np.column_stack((features_XX, X_avg_vol_cars, X_precipitation ))
    
    for i in range(0, lag):
        X_avg_vol_cars = y_avg_vol_cars[i:i + len:stride]
        X_precipitation = y_precipitation[i:i + len:stride]
        features_XX = np.column_stack((features_XX, X_avg_vol_cars, X_precipitation ))

    yy_regression = y_avg_vol_cars[lag*w+w+q:lag*w+w+q+len:stride]
    yy_classification = y_classification[lag*w+w+q:lag*w+w+q+len:stride]
    end_time_in_days = timestamps_in_days[lag*w+w+q:lag*w+w+q+len:stride]
    
    return features_XX, yy_regression, yy_classification, end_time_in_days

def cross_val_classification(time_sampling_interval, y_avg_vol_cars, y_precipitation, y_classification, timestamps_in_days):
    mean_error = []
    std_error = []
    q = 3; stride = 1
    lag_range = [1, 2, 3, 4, 5, 6]

    for lag in lag_range:
        cross_val_XX, _, cross_val_yy_classification, _ = featureEngineering( time_sampling_interval, y_avg_vol_cars, y_precipitation,
                                                                                 y_classification, timestamps_in_days, q, lag, stride)
        model_classification = LogisticRegression(penalty='l2', solver='lbfgs', C=1, max_iter=10000)
        scores = cross_val_score(model_classification, cross_val_XX, cross_val_yy_classification, cv=3, scoring='f1')
        print(scores)
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())

    plt.rc('font', size=18)
    plt.rcParams['figure.constrained_layout.use'] = True
    plt.errorbar(lag_range, mean_error, yerr=std_error, linewidth=3)
    plt.xlabel('q')
    plt.ylabel('F1 Score')
    plt.title('Logistic Regression Cross Validation Results: Polynomial Feature q')
    plt.show()

def main():
    df = pd.read_csv('Datasets/preprocessed-dataset/preproc_classification_data.csv')
    df_site_info_ccity = cityCenterSiteMetadata()
    df, selected_sites_df_dict, selected_sites_df_list = selected_sites_df(df)
    
    SITE_1 = df.Site == 1;
    df_site_1 = df[SITE_1]
    df_site_1.set_index('End_Time')

    # convert date/time to unix timestamp in sec
    all_timestamps_in_sec =pd.array((pd.DatetimeIndex(df_site_1.iloc[:,0])).astype(np.int64))/1000000000
    time_sampling_interval = all_timestamps_in_sec[1] - all_timestamps_in_sec[0]
    print("data sampling interval is %d secs" % time_sampling_interval)

    timestamps_in_days=(all_timestamps_in_sec-all_timestamps_in_sec[0])/60/60/24 # convert timestamp to days
    y_avg_vol_cars = np.extract(all_timestamps_in_sec,df_site_1.iloc[:,3]).astype(np.int64)
    y_precipitation = np.extract(all_timestamps_in_sec,df_site_1.iloc[:,4]).astype(np.int64)
    y_classification = np.extract(all_timestamps_in_sec,df_site_1.iloc[:,6]).astype(np.int64)

    visualize_site_data(timestamps_in_days, y_avg_vol_cars)
    experiment_1(y_avg_vol_cars, y_precipitation, timestamps_in_days, time_sampling_interval)
  

    # putting it together
    q = 10; lag = 3; stride = 1
    XX, yy_regression, yy_classification, end_time_in_days = featureEngineering(time_sampling_interval, y_avg_vol_cars, y_precipitation, y_classification, 
                                                                                timestamps_in_days, q, lag, stride)

    train, test = train_test_split(np.arange(0, yy_regression.size), test_size=0.2)
    model = Ridge(fit_intercept=False).fit(XX[train], yy_regression[train])
    print(model.intercept_, model.coef_)
    y_pred = model.predict(XX)
    plot_predictions(True, y_pred, timestamps_in_days, y_avg_vol_cars, end_time_in_days, time_sampling_interval)


    cross_val_classification(time_sampling_interval, y_avg_vol_cars, y_precipitation, y_classification, timestamps_in_days)

    X_train, X_test, y_train, y_test = train_test_split( XX, yy_classification, test_size=0.2)
    model_classification = LogisticRegression(penalty='l2', solver='lbfgs', C=0.1, max_iter=10000)
    model_classification.fit(X_train, y_train)
    y_pred_classification = model_classification.predict(X_test)
    print(model_classification.intercept_, model_classification.coef_)
    log_reg_confusion_matrix = confusion_matrix(
       y_test, y_pred_classification)
    log_reg_classification_report = classification_report(
         y_test, y_pred_classification, zero_division=1)
    print(log_reg_confusion_matrix)
    print(log_reg_classification_report)


if __name__ == '__main__':
    main()

# TODO:
# Cross-Validation -kFold or timeseries split
# Need to cross-validation, (stride?)
# Different Model implementations
# Include Collab Plots
# Evaluation
# Report
