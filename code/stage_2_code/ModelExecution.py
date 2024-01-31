'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.setting import setting
from sklearn.model_selection import KFold
import numpy as np


class ModelExecution(setting):
    def load_run(self):
        # load dataset
        train_loaded_data = self.dataset.load()

        X_train= np.array(train_loaded_data['X'])
        y_train= np.array(train_loaded_data['y'])

        self.method.data = {'X': X_train, 'y': y_train}
        self.method.run()

    def load_test_data(self):
        test_loaded_data = self.dataset.load()
        X_test, y_test = np.array(test_loaded_data['X']), np.array(test_loaded_data['y'])

        y_pred = self.method.test(X_test)
        learned_result = {'pred_y': y_pred, 'true_y': y_test}
        self.result.data = learned_result
        self.result.save()
        self.evaluate.data = learned_result
        return self.evaluate.evaluate()
