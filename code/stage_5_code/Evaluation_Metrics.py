'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score,precision_recall_fscore_support


class Evaluation_Metrics(evaluate):
    data = None

    def evaluate(self):
        preds = self.data['pred_y'].max(1)[1].type_as(self.data['true_y'])
        correct = preds.eq(self.data['true_y']).double()
        correct = correct.sum()
        return correct / len(self.data['true_y'])
        # precision, recall, f1score, _ = precision_recall_fscore_support(self.data['true_y'],
        #                                                                 self.data['pred_y'],
        #                                                                 average='macro',zero_division=1)
        # print('evaluating performance...')
        # return [accuracy_score(self.data['true_y'], self.data['pred_y']),precision,f1score,recall]
