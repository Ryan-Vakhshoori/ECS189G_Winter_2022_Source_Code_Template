from code.stage_4_code.Dataset_Loader import Dataset_Loader
from code.stage_4_code.ModelExecution import ModelExecution
import numpy as np

if 1:
    #---- parameter section -------------------------------
    None
    np.random.seed(1)
    #------------------------------------------------------

    #---- objection initialization section ---------------
    train_data_obj = Dataset_Loader('reviews', '')
    train_data_obj.dataset_source_folder_path = '../../data/stage_4_data/text_classification/train'
    test_data_obj = Dataset_Loader('numbers', '')
    test_data_obj.dataset_source_folder_path = '../../data/stage_4_data/text_classification/test'

    setting_obj = ModelExecution('model execution', '')

    setting_obj.prepare(loaded_obj, method_obj, result_obj, evaluate_obj)
