"""Contains evaluation class for Bernoulli Naive Bayes Classifier."""
import copy
from operator import itemgetter
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
import config

class BernoulliNBTester():
    """Evaluation class for Bernoulli Naive Bayes Classifier."""
    def __init__(self, obj, trainer, class_labels):
        self.accuracy_score = None
        self.predictions = {}
        self.data = obj
        self.trainer = trainer
        self.make_predictions(trainer, class_labels)

    def make_predictions(self, trainer, class_labels):
        """Computes probabilities and makes predictions on dataset."""
        for index in range(len(self.data.data_set)):
            new_z = copy.deepcopy(trainer.Z_const)
            for key in class_labels:
                for feature in self.data.data_set[index]:
                    if feature in trainer.cond_probs:
                        feature_prod = trainer.cond_probs[feature][key]
                        new_z[key] += np.math.log(feature_prod/(1 - feature_prod), 10)
            preds = {key: 10**(new_z[key] - max(new_z.values())) for key in new_z}
            self.data.data_pred.append(self.predictions[0][0])
            self.predictions[index] = sorted(preds.items(), key=itemgetter(1), reverse=True)

    def print_confusion_matrix(self, class_labels):
        """Prints confusion matrix."""
        cmtx = confusion_matrix(self.data.data_labels, self.data.data_pred, labels=class_labels)
        print('row is the truth, column is the system output')
        print('\t \t \t', config.LABEL_STRING)
        for row_idx in range(cmtx.shape[0]):
            cmtx_str = class_labels[row_idx] + ' '
            for col_idx in range(cmtx.shape[1]):
                cmtx_str += str(cmtx[row_idx][col_idx]) + ' '
            print(cmtx_str)
        print('\n')

    def get_accuracy_score(self):
        """Returns accuracy score."""
        return accuracy_score(self.data.data_labels, self.data.data_pred)
        