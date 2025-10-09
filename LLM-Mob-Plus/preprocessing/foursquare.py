import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
import os
import argparse  

from joblib import Parallel, delayed

from tqdm import tqdm
import pickle as pickle


def applyParallelPD(dfGrouped, func, n_jobs, print_progress=True, **kwargs):
    df_ls = Parallel(n_jobs=n_jobs)(
        delayed(func)(group, **kwargs) for _, group in tqdm(dfGrouped, disable=not print_progress)
    )
    return pd.concat(df_ls)


def _get_time(df):
    df["started_at"] = pd.to_datetime(df["started_at"])
    min_day = pd.to_datetime(df["started_at"].min().date())
    df["started_at"] = df["started_at"].dt.tz_localize(tz=None)
    df["start_day"] = (df["started_at"] - min_day).dt.days
    df["start_min"] = df["started_at"].dt.hour * 60 + df["started_at"].dt.minute
    df["weekday"] = df["started_at"].dt.weekday
    return df


def enrich_time_info(sp):
    tqdm.pandas(desc="Time enriching")
    sp = applyParallelPD(sp.groupby("user_id", group_keys=False), _get_time, n_jobs=-1, print_progress=True)
    sp.drop(columns={"started_at", "name", "category"}, inplace=True)
    sp.sort_values(by=["user_id", "start_day", "start_min"], inplace=True)
    sp = sp.reset_index(drop=True)

    sp["user_id"] = sp["user_id"].astype(int)

    sp.index.name = "id"
    sp.reset_index(inplace=True)
    return sp


def get_dataset(dataset_name):
    raw_foursquare_dir = "./dataset"
    output_dir = "./models/MHSA/foursquare/data"
    os.makedirs(output_dir, exist_ok=True) 

    train_data = pd.read_csv(os.path.join(raw_foursquare_dir, dataset_name, "train.csv"))
    vali_data = pd.read_csv(os.path.join(raw_foursquare_dir, dataset_name, "valid.csv"))
    test_data = pd.read_csv(os.path.join(raw_foursquare_dir, dataset_name, "test.csv"))

    train_data["split"] = "train"
    vali_data["split"] = "valid"
    test_data["split"] = "test"

    foursquare = pd.concat([train_data, vali_data, test_data], axis=0).reset_index(drop=True)

    foursquare_enriched = enrich_time_info(foursquare)

    enc = OrdinalEncoder(
        dtype=np.int64,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    ).fit(train_data["location_id"].values.reshape(-1, 1))
    train_data["location_id"] = enc.transform(train_data["location_id"].values.reshape(-1, 1)) + 2
    print(
        f"Min location id: {train_data.location_id.min()}, "
        f"Max location id: {train_data.location_id.max()}, "
        f"Unique location id: {train_data.location_id.nunique()}, "
        f"Total count: {len(train_data.location_id)}"
    )

    foursquare_afterUser = foursquare_enriched.copy()

    foursquare_afterUser["location_id"] = (
        enc.transform(foursquare_afterUser["location_id"].values.reshape(-1, 1)) + 2
    )

    enc = OrdinalEncoder(dtype=np.int64)
    foursquare_afterUser["user_id"] = enc.fit_transform(foursquare_afterUser["user_id"].values.reshape(-1, 1)) + 1

    print(
        f"Min user id: {foursquare_afterUser.user_id.min()}, "
        f"Max user id: {foursquare_afterUser.user_id.max()}, "
        f"Unique user id: {foursquare_afterUser.user_id.nunique()}, "
        f"Total count: {len(foursquare_afterUser.user_id)}"
    )

    foursquare_afterUser["longitude"] = (
        2
        * (foursquare_afterUser["longitude"] - foursquare_afterUser["longitude"].min())
        / (foursquare_afterUser["longitude"].max() - foursquare_afterUser["longitude"].min())
        - 1
    )
    foursquare_afterUser["latitude"] = (
        2
        * (foursquare_afterUser["latitude"] - foursquare_afterUser["latitude"].min())
        / (foursquare_afterUser["latitude"].max() - foursquare_afterUser["latitude"].min())
        - 1
    )

    foursquare_afterUser.to_csv(os.path.join(output_dir, f"dataSet_foursquare_{dataset_name}.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess dataset.")
    parser.add_argument('--dataset_name', type=str, required=True, choices=['tky', 'nyc', 'ca'],
                        help='The dataset to process.')
    args = parser.parse_args()

    get_dataset(dataset_name=args.dataset_name)