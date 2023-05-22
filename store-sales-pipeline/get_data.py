import pandas as pd
import os
import pickle
from pathes import path_to_data, path_to_temp
from tqdm import tqdm
import torch


def get_family_list():
    df_train = pd.read_csv(os.path.join(path_to_data, 'train.csv'))
    family_list = sorted(set(df_train.family))
    return family_list


def get_sales():
    print('loading sales')
    family_list = get_family_list()
    df_train = pd.read_csv(os.path.join(path_to_data, 'train.csv'))
    if os.path.exists(os.path.join(path_to_temp, 'sales.pkl')):
        with open(os.path.join(path_to_temp, 'sales.pkl'), 'rb') as f:
            sales = pickle.load(f)
    else:
        sales_list = []
        for family in tqdm(family_list):
            for store in range(1, 54 + 1):
                df = df_train[(df_train.family == family) & (df_train.store_nbr == store)].sort_values(["date"])
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                df = df.asfreq('D')
                df['sales'] = df['sales'].interpolate()
                s = df.sales
                s = s.reset_index(drop=True)
                sales_list.append(s)
        sales = torch.tensor(pd.concat(sales_list, axis=1).values, dtype=torch.float32)

        with open(os.path.join(path_to_temp, 'sales.pkl'), 'wb') as f:
            pickle.dump(sales, f)

    return sales


def get_covariates():
    print('loading covariates')
    f = open(os.path.join(path_to_data, 'future_covariates.pt'), 'rb')
    future_covariates = torch.load(f, map_location=torch.device('cpu'))
    f = open(os.path.join(path_to_data, 'only_past_covariates.pt'), 'rb')
    past_covariates = torch.load(f, map_location=torch.device('cpu'))
    return past_covariates, future_covariates