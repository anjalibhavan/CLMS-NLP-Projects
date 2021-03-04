"""Main function for training and evaluating a Bernoulli Naive Bayes classifier."""
from collections import Counter
import sys
import config
from data_reader import TrainData, TestData
from bernoulli_nb_trainer import BernoulliNBTrainer
from bernoulli_nb_tester import BernoulliNBTester
from data_writer import write_data

# Initializing variables
train_data = TrainData(sys.argv[1])
test_data = TestData(sys.argv[2])
config.CLASS_PRIOR_DELTA = float(sys.argv[3])
config.COND_PROB_DELTA = float(sys.argv[4])
config.MODEL_FILE = sys.argv[5]
config.SYS_OUTPUT = sys.argv[6]
class_labels = Counter(train_data.data_labels)

# Model training and evaluation
trainer = BernoulliNBTrainer(class_labels)
train_eval = BernoulliNBTester(train_data, trainer, class_labels)
test_eval = BernoulliNBTester(test_data, trainer, class_labels)
write_data(train_eval, test_eval, trainer)

# Printing training results
for label in class_labels:
    config.LABEL_STRING += label + ' '

class_labels = sorted(class_labels)
print('Confusion matrix for the training data:')
train_eval.print_confusion_matrix(class_labels)
print('Training accuracy=', train_eval.get_accuracy_score())
print('\n\n')
print('Confusion matrix for the test data:')
test_eval.print_confusion_matrix(class_labels)
print('Test accuracy=', test_eval.get_accuracy_score())
