import pandas as pd
import os
import pickle
import numpy as np
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
                s = df_train[(df_train.family == family) & (df_train.store_nbr == store)].sales
                s = s.reset_index(drop=True)
                sales_list.append(s)
        sales = torch.tensor(pd.concat(sales_list, axis=1).values, dtype=torch.float32)

        with open(os.path.join(path_to_temp, 'sales.pkl'), 'wb') as f:
            pickle.dump(sales, f)

    return sales
