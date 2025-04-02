from utils.misc import csv2yaml
import os

file_name = 'Adnex_v2_radiomic_features_Params_filters_8_demo-Train-Val'
file_dir = '/research/m324371/Project/adnexal/radiomic_files/SelectKBest-k-10/'
save_dir = None

# os.makedirs(save_dir, exist_ok=True)

csv2yaml(file_name, file_dir, save_dir)