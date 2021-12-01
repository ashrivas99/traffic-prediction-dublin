import glob
import os
import time
from typing import List
import pandas as pd

DATASETS_PATH = "Datasets/traffic_volumes_data_city_center_jan_jun_2020/"
FILES = glob.glob(os.path.join(DATASETS_PATH, "*.csv"))
FINAL_CSV_PATH = os.getcwd()
row_dict = {}
rows_list = []

DROP_COLUMN_NAMES: List = [
    "Avg_Volume",
    "Weighted_Avg",
    "Weighted_Var",
    "Weighted_Std_Dev",
]

FINAL_CSV_COLUMNS: List = [
    "End_Time",
    "Region",
    "Site",
    "Average_volume_of_all_detectors",
]
df_final = pd.DataFrame(columns=FINAL_CSV_COLUMNS)


def create_final_csv(df: pd.DataFrame):
    df.to_csv(os.path.join(FINAL_CSV_PATH, "Datasets/preprocessed-dataset/preproc_data_run_10.csv"))


def processCSV(df_jan):
    CITY_CENTER = df_jan.Region == "CCITY"
    df_jan = df_jan[CITY_CENTER]

    siteDetectorsSum = 0
    siteDetectorsCount = 0
    siteDetectorsAvg = 0

    row_iterator = df_jan.iterrows()
    _, previous_row = next(row_iterator)  # take first item from row_iterator

    for index, current_row in row_iterator:
        if (previous_row['End_Time'] == current_row['End_Time'] and
                previous_row['Site'] == current_row['Site']):
            siteDetectorsSum += previous_row['Sum_Volume']
            siteDetectorsCount += 1
        elif previous_row['Site'] != current_row['Site']:
            siteDetectorsSum += previous_row['Sum_Volume']
            siteDetectorsCount += 1
            siteDetectorsAvg = int(siteDetectorsSum / siteDetectorsCount)
            siteDetectorsSum = 0
            siteDetectorsCount = 0
            print(pd.to_datetime(previous_row["End_Time"]))
            new_row = {'End_Time': pd.to_datetime(previous_row["End_Time"]), 'Site': previous_row["Site"],
                       'Average_volume_of_all_detectors': siteDetectorsAvg, 'Region': previous_row["Region"]}
            global df_final
            df_final = df_final.append(new_row, ignore_index=True)

        previous_row = current_row


if __name__ == "__main__":
    start_time = time.time()
    for id, file in enumerate(FILES, 1):
        df_file = pd.read_csv(file)
        df_date_okay = df_file.copy()
        df_date_okay['End_Time'] = pd.to_datetime(df_file['End_Time'])
        processCSV(df_date_okay)

    df_final.loc[df_final['Average_volume_of_all_detectors'] >= 19, 'classification_output'] = 1
    df_final.loc[df_final['Average_volume_of_all_detectors'] < 19, 'classification_output'] = -1
    create_final_csv(df_final)
    print(f"Total Time Taken to preprocess the data is {time.time() - start_time} seconds")
