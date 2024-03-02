from code.stage_4_code.Dataset_Loader_TG import Dataset_Loader
from code.stage_4_code.Method_RNN_TG import Method_RNN_TG
from code.stage_4_code.Result_Saver import Result_Saver
from code.stage_4_code.Model_Execution_TG import ModelExecution
from code.stage_4_code.Evaluation_Metrics import Evaluation_Metrics
from code.stage_4_code.Graphing import Graph
import numpy as np

if 1:
    # ---- parameter section -------------------------------
    None
    np.random.seed(1)
    # ------------------------------------------------------

    # ---- object initialization section -------------------
    loaded_obj = Dataset_Loader('reviews', '')
    loaded_obj.dataset_source_folder_path = '../../data/stage_4_data/text_generation'
    loaded_obj.dataset_source_file_name = '/data'
    new_data = loaded_obj.load()

    graph_obj = Graph()
    method_obj = Method_RNN_TG('RNNModel', '', [], "", "", "", new_data["word_to_one_hot"], new_data["one_hot_to_word"])

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_4_result/RNN_'
    result_obj.result_destination_file_name = 'prediction_result'
    setting_obj = ModelExecution('model execution', '')
    evaluate_obj = Evaluation_Metrics('accuracy, precision, f1 score, recall', '')

    result_obj.result_destination_file_name = 'prediction_result_1'

    print('************ Start (Model 1) ************')
    setting_obj.prepare(loaded_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    train_loss, epoch = setting_obj.load_test_data()
    print('************ Finish ************')
    graph_obj.traininglossgraph(epoch, train_loss)
