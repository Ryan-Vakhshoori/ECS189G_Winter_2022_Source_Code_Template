from code.stage_2_code.Dataset_Loader import Dataset_Loader
from code.stage_2_code.Method_MLP import Method_MLP
from code.stage_2_code.Result_Saver import Result_Saver
from code.stage_2_code.ModelExecution import ModelExecution
from code.stage_2_code.Evaluation_Metrics import Evaluation_Metrics
from code.stage_2_code.Graphing import Graph
import torch
import numpy as np

# ---- Multi-Layer Perceptron script ----
if 1:
    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    # ------------------------------------------------------

    # ---- objection initialization section ---------------
    train_data_obj = Dataset_Loader('numbers', '')
    train_data_obj.dataset_source_folder_path = '../../data/stage_2_data/'
    train_data_obj.dataset_source_file_name = 'train.csv'
    test_data_obj = Dataset_Loader('numbers', '')
    test_data_obj.dataset_source_folder_path = '../../data/stage_2_data/'
    test_data_obj.dataset_source_file_name = 'test.csv'
    # x = train_data_obj.load()

    method_obj = Method_MLP(mName='multi-layer perceptron', mDescription='', hidden_layers=[], optimizer="adam",activation_function="relu")
    method_obj_1 = Method_MLP(mName='multi-layer perceptron', mDescription='', hidden_layers=[(28,50),(50,28)], optimizer="adam",activation_function="sigmoid")
    method_obj_2 = Method_MLP(mName='multi-layer perceptron', mDescription='', hidden_layers=[(28,50), (50,28)], optimizer="adam",activation_function="tanh")
    method_obj_3 = Method_MLP(mName='multi-layer perceptron', mDescription='', hidden_layers=[(28,50),(50,28)], optimizer="adam",activation_function="relu")
    method_obj_4 = Method_MLP(mName='multi-layer perceptron', mDescription='', hidden_layers=[(28,28)], optimizer="adam",activation_function="relu")

    graph_obj = Graph()

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_2_result/MLP_'
    result_obj.result_destination_file_name = 'prediction_result'
    setting_obj = ModelExecution('model execution', '')
    evaluate_obj = Evaluation_Metrics('accuracy, precision, f1 score, recall', '')
    # ------------------------------------------------------

    print('************ Start (Model 1) ************')
    setting_obj.prepare(train_data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    epoch, train_loss = setting_obj.load_run()
    graph_obj.traininglossgraph(epoch, train_loss)
    setting_obj.prepare(test_data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    accuracy, precision, f1_score, recall = setting_obj.load_test_data()
    print('************ Overall Performance ************')
    print('MLP Accuracy: ' + str(accuracy))
    print('MLP Precision: ' + str(precision))
    print('MLP F1 Score: ' + str(f1_score))
    print('MLP Recall: ' + str(recall))
    print('************ Finish ************')

    print('************ Start (Model 2) ************')
    setting_obj.prepare(train_data_obj, method_obj_1, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    epoch, train_loss = setting_obj.load_run()
    graph_obj.traininglossgraph(epoch, train_loss)
    setting_obj.prepare(test_data_obj, method_obj_1, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    accuracy, precision, f1_score, recall = setting_obj.load_test_data()
    print('************ Overall Performance ************')
    print('MLP Accuracy: ' + str(accuracy))
    print('MLP Precision: ' + str(precision))
    print('MLP F1 Score: ' + str(f1_score))
    print('MLP Recall: ' + str(recall))
    print('************ Finish ************')

    print('************ Start (Model 3) ************')
    setting_obj.prepare(train_data_obj, method_obj_2, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    epoch, train_loss = setting_obj.load_run()
    graph_obj.traininglossgraph(epoch, train_loss)
    setting_obj.prepare(test_data_obj, method_obj_2, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    accuracy, precision, f1_score, recall = setting_obj.load_test_data()
    print('************ Overall Performance ************')
    print('MLP Accuracy: ' + str(accuracy))
    print('MLP Precision: ' + str(precision))
    print('MLP F1 Score: ' + str(f1_score))
    print('MLP Recall: ' + str(recall))
    print('************ Finish ************')

    print('************ Start (Model 4) ************')
    setting_obj.prepare(train_data_obj, method_obj_3, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    epoch, train_loss = setting_obj.load_run()
    graph_obj.traininglossgraph(epoch, train_loss)
    setting_obj.prepare(test_data_obj, method_obj_3, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    accuracy, precision, f1_score, recall = setting_obj.load_test_data()
    print('************ Overall Performance ************')
    print('MLP Accuracy: ' + str(accuracy))
    print('MLP Precision: ' + str(precision))
    print('MLP F1 Score: ' + str(f1_score))
    print('MLP Recall: ' + str(recall))
    print('************ Finish ************')

    print('************ Start (Model 5) ************')
    setting_obj.prepare(train_data_obj, method_obj_4, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    epoch, train_loss = setting_obj.load_run()
    graph_obj.traininglossgraph(epoch, train_loss)
    setting_obj.prepare(test_data_obj, method_obj_4, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    accuracy, precision, f1_score, recall = setting_obj.load_test_data()
    print('************ Overall Performance ************')
    print('MLP Accuracy: ' + str(accuracy))
    print('MLP Precision: ' + str(precision))
    print('MLP F1 Score: ' + str(f1_score))
    print('MLP Recall: ' + str(recall))
    print('************ Finish ************')


    # -------------------------------------------------------