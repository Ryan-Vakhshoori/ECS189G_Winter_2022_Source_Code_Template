'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.setting import setting
import numpy as np


class ModelExecution(setting):
    def load_test_data(self):
        loaded_data = self.dataset.load()
        self.method.data = loaded_data["train_data"]
        resulting_loss, epoch = self.method.run()
        result_data = {'resulting_loss': resulting_loss, 'epochs': epoch}
        self.evaluate.data = result_data
        self.result.data = result_data
        self.result.save()
        return resulting_loss, epoch