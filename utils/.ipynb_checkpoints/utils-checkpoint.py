
import pandas as pd
import numpy as np
import os
import yaml
from box import Box

# ======================================================================
def csv2yaml(file_name, file_dir, save_dir:str=None):
    """
    Author @ Mrinal Kanti Dhar
    November 15, 2024
    
    This code is to create a yaml file that contains radiomic feature names.
    It reads a csv file and stores the header names (basically radiomic feature names) in a yaml file.

    Args:
        - file_name: File name without extension
        - file_dir: Directory of the file
        - save_dir: Directory to save. If None, then it considers file_dir as save_dir
    """
    # Read csv file
    csv_file = pd.read_csv(os.path.join(file_dir, file_name + '.csv'))

    # Read headers
    headers = list(csv_file.keys())

    # Create a dictionary
    headers_dict = {}
    headers_dict["long"] = headers
    headers_dict["short"] = None

    # Save as yaml
    save_dir = file_dir if save_dir is None else save_dir

    with open(os.path.join(save_dir, file_name + '.yaml'), "w") as f:
        yaml.dump(headers_dict, f)
    