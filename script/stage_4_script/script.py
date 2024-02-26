from code.stage_4_code.Dataset_Loader_TC import Dataset_Loader
from code.stage_4_code.Method_RNN_TC import Method_RNN
from code.stage_4_code.Result_Saver import Result_Saver
from code.stage_4_code.ModelExecution import ModelExecution
from code.stage_4_code.Evaluation_Metrics import Evaluation_Metrics
import numpy as np

if 1:
    #---- parameter section -------------------------------
    None
    np.random.seed(1)
    #------------------------------------------------------

    #---- objection initialization section ---------------
    loaded_obj = Dataset_Loader('reviews', '')
    loaded_obj.dataset_source_folder_path = '../../data/stage_4_data/text_classification'
    new_data = loaded_obj.load()
    print(new_data)
    method_obj = Method_RNN()

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_4_result/RNN_'
    result_obj.result_destination_file_name = 'prediction_result'
    setting_obj = ModelExecution('model execution', '')
    evaluate_obj = Evaluation_Metrics('accuracy, precision, f1 score, recall', '')

    result_obj.result_destination_file_name = 'prediction_result_1'
    setting_obj.prepare(loaded_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    epoch, train_loss = setting_obj.load_run()
