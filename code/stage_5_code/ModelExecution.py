'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.setting import setting

class ModelExecution(setting):
    def load_test_data(self):
        loaded_data = self.dataset.load()
        self.method.data = loaded_data
        resulting_loss, epoch, predicted_labels, actual_labels = self.method.run()
        result_data = {'pred_y': predicted_labels, 'true_y': actual_labels}
        self.evaluate.data = result_data
        self.result.data = result_data
        self.result.save()
        return self.evaluate.evaluate(), resulting_loss, epoch
