import os
from zipfile import ZipFile

import numpy as np
import pandas as pd
from tensorflow import keras

URI = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
ZIP_FILENAME = "jena_climate_2009_2016.csv.zip"
CSV_FILENAME = "jena_climate_2009_2016.csv"
DATE_TIME_KEY = "Date Time"
TARGET_FEATURE = "T (degC)"

feature_keys = [
    "p (mbar)",
    "T (degC)",
    "Tpot (K)",
    "Tdew (degC)",
    "rh (%)",
    "VPmax (mbar)",
    "VPact (mbar)",
    "VPdef (mbar)",
    "sh (g/kg)",
    "H2OC (mmol/mol)",
    "rho (g/m**3)",
    "wv (m/s)",
    "max. wv (m/s)",
    "wd (deg)",
]

titles = [
    "气压",
    "温度",
    "开尔文温度",
    "露点温度",
    "相对湿度",
    "饱和水汽压",
    "水汽压",
    "水汽压亏缺",
    "比湿度",
    "水汽浓度",
    "密度",
    "风速",
    "最大风速",
    "风向",
]

selected_feature_indices = [0, 1, 5, 7, 8, 10, 11]
selected_features = [feature_keys[i] for i in selected_feature_indices]
selected_titles = [titles[i] for i in selected_feature_indices]
time_feature_names = ["hour_sin", "hour_cos", "day_sin", "day_cos"]
colors = [
    "blue",
    "orange",
    "green",
    "red",
    "purple",
    "brown",
    "pink",
    "gray",
    "olive",
    "cyan",
]


def download_and_load_data(data_dir="."):
    os.makedirs(data_dir, exist_ok=True)
    zip_path = keras.utils.get_file(
        fname=ZIP_FILENAME,
        origin=URI,
        cache_dir=data_dir,
        cache_subdir="",
    )
    with ZipFile(zip_path, "r") as zip_file:
        zip_file.extractall(data_dir)

    csv_path = os.path.join(data_dir, CSV_FILENAME)
    df = pd.read_csv(csv_path)
    return df


def get_selected_features(df):
    features = df[selected_features].copy()
    date_time = pd.to_datetime(
        df[DATE_TIME_KEY], format="%d.%m.%Y %H:%M:%S", errors="raise"
    )
    features.index = date_time

    hour = date_time.dt.hour + date_time.dt.minute / 60.0
    day_of_year = date_time.dt.dayofyear - 1

    features["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    features["day_sin"] = np.sin(2 * np.pi * day_of_year / 365.0)
    features["day_cos"] = np.cos(2 * np.pi * day_of_year / 365.0)

    return features


def normalize_features(features, train_split):
    training_slice = features.iloc[:train_split]
    mean = training_slice.mean(axis=0)
    std = training_slice.std(axis=0)
    normalized = (features - mean) / std
    return normalized, mean, std


def split_features(features, train_split):
    train_data = features.iloc[:train_split]
    val_data = features.iloc[train_split:]
    return train_data, val_data


def build_timeseries_datasets(
    train_data,
    val_data,
    features,
    train_split,
    past=720,
    future=72,
    step=6,
    batch_size=256,
):
    start = past + future
    end = start + train_split
    sequence_length = int(past / step)
    target_index = features.columns.get_loc(TARGET_FEATURE)

    x_train = train_data.iloc[:, :].values
    y_train = features.iloc[start:end, [target_index]].values

    dataset_train = keras.preprocessing.timeseries_dataset_from_array(
        x_train,
        y_train,
        sequence_length=sequence_length,
        sampling_rate=step,
        batch_size=batch_size,
    )

    x_end = len(val_data) - past - future
    label_start = train_split + past + future

    x_val = val_data.iloc[:x_end, :].values
    y_val = features.iloc[label_start:, [target_index]].values

    dataset_val = keras.preprocessing.timeseries_dataset_from_array(
        x_val,
        y_val,
        sequence_length=sequence_length,
        sampling_rate=step,
        batch_size=batch_size,
    )

    return dataset_train, dataset_val, sequence_length
