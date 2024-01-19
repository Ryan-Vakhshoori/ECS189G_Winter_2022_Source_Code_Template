from code.stage_2_code.Dataset_Loader import Dataset_Loader
import torch
import numpy as np

# ---- Multi-Layer Perceptron script ----
if 1:
    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    # ------------------------------------------------------

    # ---- objection initialization setction ---------------
    train_data_obj = Dataset_Loader('numbers', '')
    train_data_obj.dataset_source_folder_path = '../../data/stage_2_data/'
    train_data_obj.dataset_source_file_name = 'test.csv'
    x = train_data_obj.load()