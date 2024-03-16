from code.stage_5_code.Dataset_Loader_Node_Classification import Dataset_Loader
from code.stage_5_code.Result_Saver import Result_Saver
from code.stage_5_code.ModelExecution import ModelExecution
from code.stage_5_code.Evaluation_Metrics import Evaluation_Metrics
from code.stage_5_code.Method_GNN_Pubmed import Method_GNN_Pubmed
from code.stage_5_code.Graphing import Graph
import numpy as np

if 1:
    #---- parameter section -------------------------------
    None
    np.random.seed(1)
    #------------------------------------------------------

    #---- objection initialization section ---------------
    loaded_obj = Dataset_Loader(95, 'pubmed')
    loaded_obj.dataset_source_folder_path = '../../data/stage_5_data/pubmed'
    loaded_obj.dataset_name = 'pubmed'
    #output = loaded_obj.load()
    # print(output)
    # print(output['graph']['X'].shape[1])
    # graph_obj = Graph()
    graph_obj = Graph()
    method_obj = Method_GNN_Pubmed('GNN', '', 20, 2, "adam", "")

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_5_result/GNN_'
    setting_obj = ModelExecution('model execution', '')
    evaluate_obj = Evaluation_Metrics('accuracy', '')

    result_obj.result_destination_file_name = 'prediction_result_1'
    # seed: 2147483648
    print('************ Start (GNN Model 1) ************')
    setting_obj.prepare(loaded_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    [accuracy, precision, f1_score, recall], train_loss, epoch = setting_obj.load_test_data()
    print('************ Overall Performance ************')
    print('GNN Pubmed Accuracy: ' + str(accuracy.item()))
    print('GNN Pubmed Precision: ' + str(precision))
    print('GNN Pubmed F1 Score: ' + str(f1_score))
    print('GNN Pubmed Recall: ' + str(recall))
    print('************ Finish ************')
    graph_obj.traininglossgraph(epoch, train_loss)

    method_obj = Method_GNN_Pubmed('GNN', '', 32, 2, "adam", "")

    result_obj.result_destination_file_name = 'prediction_result_2'

    print('************ Start (GNN Model 2) ************')
    setting_obj.prepare(loaded_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    [accuracy, precision, f1_score, recall], train_loss, epoch = setting_obj.load_test_data()
    print('************ Overall Performance ************')
    print('GNN Pubmed Accuracy: ' + str(accuracy.item()))
    print('GNN Pubmed Precision: ' + str(precision))
    print('GNN Pubmed F1 Score: ' + str(f1_score))
    print('GNN Pubmed Recall: ' + str(recall))
    print('************ Finish ************')
    graph_obj.traininglossgraph(epoch, train_loss)

    method_obj = Method_GNN_Pubmed('GNN', '', 16, 2, "", "")

    result_obj.result_destination_file_name = 'prediction_result_3'

    print('************ Start (GNN Model 3) ************')
    setting_obj.prepare(loaded_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    [accuracy, precision, f1_score, recall], train_loss, epoch = setting_obj.load_test_data()
    print('************ Overall Performance ************')
    print('GNN Pubmed Accuracy: ' + str(accuracy.item()))
    print('GNN Pubmed Precision: ' + str(precision))
    print('GNN Pubmed F1 Score: ' + str(f1_score))
    print('GNN Pubmed Recall: ' + str(recall))
    print('************ Finish ************')
    graph_obj.traininglossgraph(epoch, train_loss)