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
        predict, labels, resulting_loss, epoch = self.method.run()
        learned_result = {'pred_y': predict, 'true_y': labels}
        self.result.data = learned_result
        self.result.save()
        self.evaluate.data = learned_result
        return {'evaluate': self.evaluate.evaluate(), 'train_loss': resulting_loss, 'epoch': epoch}
