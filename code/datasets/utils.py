# Copyright © 2026, SAS Institute Inc., Cary, NC, USA.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pandas as pd
from scipy.io import arff

# datasets -- download/preprocess
from imblearn.datasets import fetch_datasets
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer


IMB_DATASETS = ["ecoli", "yeast_me2", "solar_flare_m0", "abalone", "car_eval_34", "car_eval_4", "mammography", "abalone_19"]
# CSM_DATASETS = ["diabetes", "phoneme"]
CSM_DATASETS = []
# BIG_DATASETS = ["higgs", "MiniBooNE"]
BIG_DATASETS = []
# MIX_DATASETS = ["cardio", "churn"]
MIX_DATASETS = []
# ALL_DATASETS = IMB_DATASETS + CSM_DATASETS
ALL_DATASETS = IMB_DATASETS

dtypes = {
    "cardio": {
        "age": int,
        "gender": "category",
        'height': int,
        'weight': float,
        'ap_hi': int,
        'ap_lo': int,
        'cholesterol': "category",
        'gluc': "category",
        'smoke': "category",
        'alco': "category",
        'active': "category",
        'cardio': "category",
        },
    "churn": {
        "CreditScore": int,
        "Geography": "category",
        'Gender': "category",
        'Age': int,
        'Tenure': int,
        'Balance': float,
        'NumOfProducts': "category",
        'HasCrCard': "category",
        'IsActiveMember': "category",
        'EstimatedSalary': float,
        'Exited': "category",
        },
}


def load_data(data_name):
    if data_name in IMB_DATASETS:
        data = fetch_datasets()[data_name]
        X, y = data.data, data.target

    if data_name in CSM_DATASETS:
        df = pd.read_csv(f"../data/{data_name}.csv")
        X = df.iloc[:, :-1].to_numpy()
        y = df.iloc[:, -1].to_numpy()

    if data_name in BIG_DATASETS:
        arff_file = arff.loadarff(f"../data/{data_name}.arff")
        df = pd.DataFrame(arff_file[0])
        df = imbalance_data(data_name, df)

        X = df.iloc[:, 1:].to_numpy()
        y = df.iloc[:, 0].to_numpy().astype(int)

    if data_name in MIX_DATASETS:
        if data_name == "cardio":
            df = pd.read_csv(f"../data/{data_name}.csv", sep=";", index_col="id").astype(dtypes[data_name])
            df = df.reset_index(drop=True)

        if data_name == "churn":
            df = pd.read_csv(f"../data/{data_name}.csv").astype(dtypes[data_name])
            df.drop(columns=["RowNumber", "CustomerId", "Surname"], inplace=True)

        df = imbalance_data(data_name, df)

        X = df.iloc[:, :-1].to_numpy()
        y = df.iloc[:, -1].to_numpy()
    
    return X, y


def imbalance_data(data_name, df, desired_ratio=26):
    # manual cleaning
    if data_name == "higgs":
        target_str = "class"
    if data_name == "MiniBooNE":
        target_str = "signal"
        df[target_str] = df[target_str].map({b'False': 0, b'True': 1})
    if data_name == "cardio":
        target_str = "cardio"
    if data_name == "churn":
        target_str = "Exited"

    df = df.dropna()
    df = df.drop_duplicates()
    df = df.loc[df.drop_duplicates(subset=df.select_dtypes(include='number').columns).index]

    if data_name == "cardio":
        # undersample negative class by a fraction of 5
        majority_df = df[df[target_str] == 0].sample(n=len(df[df[target_str] == 0]) // 5, random_state=42)
        minority_df = df[df[target_str] == 1]
        # shuffle
        df = pd.concat([majority_df, minority_df]).sample(frac=1, random_state=42)
    
    n_majority = len(df[df[target_str] == 0]) 
    n_minority = n_majority // desired_ratio

    # undersample positive class
    majority_df = df[df[target_str] == 0]
    minority_df = df[df[target_str] == 1].sample(n=n_minority, random_state=42)
    
    # shuffle
    df = pd.concat([majority_df, minority_df]).sample(frac=1, random_state=42)

    return df


def prepare_data(X, y, dtypes=None):
    if dtypes:
        # process numerical columns between -1 and 1 and encode categorical columns 
        num_idx = [i for i, (col, dtype) in enumerate(dtypes.items()) if dtype in [int, float]]
        cat_idx = [i for i, (col, dtype) in enumerate(dtypes.items()) if dtype == "category"][:-1]
        preprocessor = ColumnTransformer([
            ('num', MinMaxScaler(feature_range=(-1, 1)), num_idx),
            ('cat', OrdinalEncoder(), cat_idx)
        ])
        processed_X = preprocessor.fit_transform(X)

    else:
        # scale numerical columns between -1 and 1
        scaler = MinMaxScaler(feature_range=(-1, 1))
        processed_X = scaler.fit_transform(X)
    
    if np.min(y) == -1:
        enc_y = pd.Series(y).map({-1: 0, 1: 1}).to_numpy()
    else:
        enc_y = y.copy()
    
    return processed_X, enc_y
