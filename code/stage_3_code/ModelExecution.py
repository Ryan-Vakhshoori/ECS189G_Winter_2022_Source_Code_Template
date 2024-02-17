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
        self.method.data = loaded_data
        train_results = self.method.run()
        accuracy = train_results['accuracy']
        # accuracy, resulting_loss, epoch = self.method.run()
        self.evaluate.data = accuracy
        self.result.data = accuracy
        self.result.save()
        # return {'evaluate': accuracy, 'train_loss': resulting_loss, 'epoch': epoch}
        return train_results
