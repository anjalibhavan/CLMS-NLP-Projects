"""Contains functions for writing model and predictions to files."""
import numpy as np
import config

def write_to_model_file(trainer):
    """Writes Naive Bayes model to model file."""
    with open(config.MODEL_FILE, 'w') as g:
        g.write('%%%%% prior prob P(c) %%%%%')
        g.write('\n')
        for key in trainer.class_labels:
            g.write(key + ' ' + str(10**trainer.priors[key]) + ' ' + str(trainer.priors[key]))
            g.write('\n')
        g.write('%%%%% conditional prob P(f|c) %%%%%')
        g.write('\n')
        for key in trainer.class_labels:
            g.write('%%%%% conditional prob P(f|c) c=' + key + ' %%%%%')
            g.write('\n')
            for feature in config.FEATURE_DICT:
                prod = trainer.cond_probs[feature][key]
                g.write(feature + ' ' + key + ' ' + str(prod) + ' ' + str(np.math.log(prod, 10)))
                g.write('\n')


def write_to_sys_file_helper(obj, g):
    """Helper function for writing predictions to sys file."""
    for index in range(len(obj.data_set)):
        finalstr = "array" + str(index) + ' ' + obj.data_labels[index] + ' '
        for pair in obj.predictions[index]:
            finalstr += pair[0] + ' ' + str(pair[1]/sum(obj.predictions.values())) + ' '
        g.write(finalstr)
        g.write('\n')


def write_to_sys_file(train_eval, test_eval):
    """Writes predictions to sys file in required format."""
    with open(config.SYS_OUTPUT, 'w') as g:
        g.write('%%%%% training data:')
        g.write('\n')
        write_to_sys_file_helper(train_eval, g)
        g.write('\n\n')
        g.write('%%%%% test data:')
        g.write('\n')
        write_to_sys_file_helper(test_eval, g)


def write_data(train_eval, test_eval, trainer):
    """Main function for writing to model and sys files."""
    write_to_model_file(trainer)
    write_to_sys_file(train_eval, test_eval)
