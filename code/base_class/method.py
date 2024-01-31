'''
Base MethodModule class for all models and frameworks
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD


import abc


class method:
    '''
    MethodModule: Abstract Class
    Entries: method_name: the name of the MethodModule 
             method_description: the textual description of the MethodModule
             
             method_start_time: start running time of MethodModule
             method_stop_time: stop running time of MethodModule
             method_running_time: total running time of the MethodModule
             method_training_time: time cost of the training phrase
             method_testing_time: time cost of the testing phrase
    '''
    
    method_name = None
    method_description = None
    method_hidden_layers = []
    method_optimizer = None
    method_activation_function = None
    
    data = None
    
    method_start_time = None
    method_stop_time = None
    method_running_time = None
    method_training_time = None
    method_testing_time = None

    # initialization function
    def __init__(self,mName=None,mDescription=None,hidden_layers=[], optimizer="adam", activation_function="relu",):
        self.method_name = mName
        self.method_description = mDescription
        self.method_hidden_layers = hidden_layers
        self.method_optimizer = optimizer
        self.method_activation_function = activation_function

    # running function
    @abc.abstractmethod
    def run(self, trainData, trainLabel, testData):
        return
