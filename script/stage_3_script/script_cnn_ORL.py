from code.stage_3_code.Dataset_Loader_ORL import Dataset_Loader
from code.stage_3_code.Method_CNN_ORL import Method_CNN
from code.stage_3_code.Result_Saver import Result_Saver
from code.stage_3_code.ModelExecution import ModelExecution
from code.stage_3_code.Evaluation_Metrics import Evaluation_Metrics
from code.stage_3_code.Graphing import Graph
import torch
import numpy as np

# ---- Multi-Layer Perceptron script ----
if 1:
    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    # ------------------------------------------------------

    # ---- objection initialization section ---------------
    loaded_obj = Dataset_Loader('numbers', '')
    loaded_obj.dataset_source_folder_path = '../../data/stage_3_data/'
    loaded_obj.dataset_source_file_name = 'ORL'

    graph_obj = Graph()

    method_obj = Method_CNN('ORLModel', '', [32, 64, 128, 19712], "", "")

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_3_result/CNN_'
    result_obj.result_destination_file_name = 'ORL_prediction_result'

    setting_obj = ModelExecution('model execution', '')

    evaluate_obj = Evaluation_Metrics('accuracy', '')
    # ------------------------------------------------------

    print('************ Start (Model 1) ************')
    setting_obj.prepare(loaded_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    [accuracy, precision, f1_score, recall], train_loss, epoch = setting_obj.load_test_data()
    print('************ Overall Performance ************')
    print('ORL Accuracy: ' + str(accuracy))
    print('ORL Precision: ' + str(precision))
    print('ORL F1 Score: ' + str(f1_score))
    print('ORL Recall: ' + str(recall))
    print('************ Finish ************')
    graph_obj.traininglossgraph(epoch, train_loss)

    method_obj = Method_CNN('ORLModel', '', [32, 128, 256, 39424], "", "")
    result_obj.result_destination_file_name = 'ORL_prediction_result_2'
    print('************ Start (Model 2) ************')
    setting_obj.prepare(loaded_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    [accuracy, precision, f1_score, recall], train_loss, epoch = setting_obj.load_test_data()
    print('************ Overall Performance ************')
    print('ORL Accuracy: ' + str(accuracy))
    print('ORL Precision: ' + str(precision))
    print('ORL F1 Score: ' + str(f1_score))
    print('ORL Recall: ' + str(recall))
    print('************ Finish ************')
    graph_obj.traininglossgraph(epoch, train_loss)

    method_obj = Method_CNN('ORLModel', '', [16, 32, 64, 9856], "", "")
    result_obj.result_destination_file_name = 'ORL_prediction_result_2'
    print('************ Start (Model 3) ************')
    setting_obj.prepare(loaded_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    [accuracy, precision, f1_score, recall], train_loss, epoch = setting_obj.load_test_data()
    print('************ Overall Performance ************')
    print('ORL Accuracy: ' + str(accuracy))
    print('ORL Precision: ' + str(precision))
    print('ORL F1 Score: ' + str(f1_score))
    print('ORL Recall: ' + str(recall))
    print('************ Finish ************')
    graph_obj.traininglossgraph(epoch, train_loss)