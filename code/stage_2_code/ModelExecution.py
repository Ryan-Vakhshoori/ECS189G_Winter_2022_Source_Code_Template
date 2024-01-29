'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.setting import setting
from sklearn.model_selection import KFold
import numpy as np


class ModelExecution(setting):
    def load_run_save_evaluate(self):
        # load dataset
        train_loaded_data = self.dataset['train'].load()
        test_loaded_data = self.dataset['test'].load()
        X_train, X_test = np.array(train_loaded_data['X']), np.array(test_loaded_data['X'])
        y_train, y_test = np.array(train_loaded_data['y']), np.array(test_loaded_data['y'])

        self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
        learned_result = self.method.run()

        self.result.data = learned_result
        self.result.save()


        score_list = self.evaluate.evaluate(learned_result)

        return np.mean(score_list), np.std(score_list)

