""" Contains classes for initializing data and reading from files."""
from abc import ABCMeta, abstractmethod
import config

class DataReader(metaclass=ABCMeta):
    """ Abstract class for reading data from file."""
    def __init__(self, data_file):
        self.data_file = data_file
        self.data_set = {}
        self.data_labels = []
        self.data_pred = []

    @abstractmethod
    def read_data(self):
        """Read data from file"""
        pass

class TrainData(DataReader):
    """Initializes training data."""
    def __init__(self, train_file):
        DataReader.__init__(self, train_file)
        self.read_data()

    def read_data(self):
        """Reads training data and constructs feature/label counts dictionary from file."""
        with open(self.data_file, 'r') as g:
            for index, line in enumerate(g):
                line_list = line.split()
                label = line_list[0]
                self.data_labels.append(label)
                for pair in line_list[1:]:
                    f = pair.split(':')[0]
                    self.data_set[index].add(f)
                    if f not in config.FEATURE_DICT:
                        config.FEATURE_DICT[f] = {}
                    config.FEATURE_DICT[f][label] += 1


class TestData(DataReader):
    """Initializes test data."""
    def __init__(self, test_file):
        DataReader.__init__(self, test_file)
        self.read_data()

    def read_data(self):
        """Reads test data from file."""
        with open(self.data_file, 'r') as g:
            for index, line in enumerate(g):
                line_list = line.split()
                label = line_list[0]
                self.data_set[index] = set(pair.split(':')[0] for pair in line_list[1:])
                self.data_labels.append(label)
