from code.stage_5_code.Dataset_Loader_Node_Classification import Dataset_Loader
from code.stage_5_code.Graphing import Graph
from code.stage_5_code.Method_GCN import Method_GCN
from code.stage_5_code.Result_Saver import Result_Saver
from code.stage_5_code.ModelExecution import ModelExecution
from code.stage_5_code.Evaluation_Metrics import Evaluation_Metrics

if 1:
    loaded_obj = Dataset_Loader()
    loaded_obj.dataset_source_folder_path = '../../data/stage_5_data/cora'
    graph_obj = Graph()
    method_obj = Method_GCN()

    result_obj = Result_Saver()
    result_obj.result_destination_folder_path = '../../results/stage_5_data/GCN_'
    setting_obj = ModelExecution()
    evaluate_obj = Evaluation_Metrics()

    result_obj.result_destination_file_name = 'prediction_result_cora'

    print('************ Start (GCN Model 1) ************')
    setting_obj.prepare(loaded_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    [accuracy, precision, f1_score, recall], train_loss, epoch = setting_obj.load_test_data()
    print('************ Overall Performance ************')
    print('RNN Text Classification Accuracy: ' + str(accuracy))
    print('RNN Text Classification Precision: ' + str(precision))
    print('RNN Text Classification F1 Score: ' + str(f1_score))
    print('RNN Text Classification Recall: ' + str(recall))
    print('************ Finish ************')
    graph_obj.traininglossgraph(epoch, train_loss)