from tqdm import tqdm
import pandas as pd
import os
from pathes import path_to_data, path_to_temp
from get_data import get_family_list


def horizons_to_submission(horizons):
    print('making submission')
    pd_submission = pd.read_csv(os.path.join(path_to_data, 'test.csv'))
    pd_submission['sales'] = 0
    family_list = get_family_list()
    i = 0
    for family in family_list:
        for store_nbr in range(1, 54 + 1):
            pd_submission.loc[
                (pd_submission.family == family) & (pd_submission.store_nbr == store_nbr), ['sales']] = horizons[:, i].flatten().numpy()
            i += 1
    pd_submission.drop(['date', 'store_nbr', 'family', 'onpromotion'], axis=1, inplace=True)
    pd_submission.to_csv(os.path.join(path_to_temp, 'submission.csv'), index=False)
