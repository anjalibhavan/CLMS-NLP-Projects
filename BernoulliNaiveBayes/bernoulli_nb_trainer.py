"""Contains training class for Bernoulli Naive Bayes Classifier."""

import numpy as np
import config

class BernoulliNBTrainer():
    """Training class for Bernoulli Naive Bayes Classifier."""
    def __init__(self, class_labels):
        self.Z_const = {}
        self.priors = {}
        self.cond_probs = {}
        self.class_labels = class_labels
        self.calc_priors()
        self.calc_cond_prob()
        self.calc_constant_factor()

    def calc_priors(self):
        """Calculates prior probabilities from the data"""
        label_counts = sum(self.class_labels.values())
        denominator = label_counts + len(self.class_labels)*config.CLASS_PRIOR_DELTA
        for key in self.class_labels:
            numerator = config.CLASS_PRIOR_DELTA + self.class_labels[key]
            self.priors[key] = np.math.log(numerator/denominator, 10)

    def calc_cond_prob(self):
        """Calculates conditional probabilities from the data"""
        for key in self.class_labels:
            denominator = self.class_labels[key] + 2*config.COND_PROB_DELTA
            for feature in config.FEATURE_DICT:
                numerator = config.COND_PROB_DELTA + config.FEATURE_DICT[feature][key]
                if feature not in self.cond_probs:
                    self.cond_probs[feature] = {}
                self.cond_probs[feature][key] = numerator/denominator

    def calc_constant_factor(self):
        """Calculates constant factor (Z) beforehand for faster computation"""
        for key in self.priors:
            tempkey = 1
            for feature in config.FEATURE_DICT:
                tempkey *= 1 - self.cond_probs[feature][key]
            self.Z_const[key] = self.priors[key] + np.math.log(tempkey, 10)
