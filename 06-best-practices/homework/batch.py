#!/usr/bin/env python
# coding: utf-8

import sys
import os
from datetime import datetime
import pickle
import pandas as pd


# def get_input_path(year, month):
#     default_input_pattern = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
#     input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
#     return input_pattern.format(year=year, month=month)


# def get_output_path(year, month):
#     default_output_pattern = 's3://nyc-duration-prediction-alexey/taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet'
#     output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
#     return output_pattern.format(year=year, month=month)


def main(year, month):
    # input_file = get_input_path(year, month)
    
    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'

    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    return input_file, dv, lr


def read_data(filename):

    df = pd.read_parquet(filename)

    return df

def prepare_data(df, categorical):

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

def create_test_data():

    options = {
        'client_kwargs': {
            'endpoint_url': 'http://localhost:4566' #S3_ENDPOINT_URL
        }
    }

    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
    ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df_input = pd.DataFrame(data, columns=columns)

    df_input.to_parquet(
        "s3://nyc-duration/{year:04d}-{month:02d}.parquet",
        engine='pyarrow',
        compression=None,
        index=False,
        storage_options=options
    )


if __name__ == '__main__':

    year = int(sys.argv[1])
    month = int(sys.argv[2])

    categorical = ['PULocationID', 'DOLocationID']

    input_file, dv, lr =  main(year, month)

    df_read = read_data(input_file)
    df = prepare_data(df_read, categorical)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print('predicted mean duration:', y_pred.mean())

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    output_file = f'taxi_type=yellow_year={year:04d}_month={month:02d}.parquet'

    df_result.to_parquet(output_file, engine='pyarrow', index=False)

    create_test_data()
