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
        loaded_data = self.dataset.load()
        learned_result = self.method.run(loaded_data['X'], loaded_data['y'])

        self.result.data = learned_result
        self.result.save()

        score_list = self.evaluate.evaluate(learned_result)

        return np.mean(score_list), np.std(score_list)

