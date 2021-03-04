"""Contains class for parsing files and writing output"""

def is_number(value):
    """Checks if string is numeric"""
    try:
        float(value)
        return True
    except ValueError:
        return False


class FileParser():
    """Class for parsing files and writing output"""
    def __init__(self, test_file, boundary_file, model_file):
        self.weights = {}
        self.test_set = {}
        self.sentence_lengths = {}
        self.weights = {}
        self.final_probs = {}
        self.tagset = set()
        self.test_labels = {}
        self.parse_test_file(test_file)
        self.parse_boundary_file(boundary_file)
        self.parse_model_file(model_file)

    def parse_test_file(self, test_file):
        """Parses test file"""
        with open(test_file, 'r') as f:
            for line in f:
                line_list = line.split()
                instance, wordpos, word = line_list[0].split('-', maxsplit=2)
                if int(instance) - 1 not in self.test_labels:
                    self.test_labels[int(instance) - 1] = {}
                if int(instance) - 1 not in self.test_set:
                    self.test_set[int(instance) - 1] = {}
                self.test_labels[int(instance) - 1][int(wordpos)] = line_list[1]
                self.tagset.add(line_list[1])
                tags = set(term for term in line_list[2:] if not is_number(term))
                self.test_set[int(instance) - 1][int(wordpos)] = [word, tags]

    def parse_boundary_file(self, boundary_file):
        """Parses boundary file"""
        with open(boundary_file, 'r') as f:
            for i, line in enumerate(f):
                self.sentence_lengths[i] = int(line.rstrip('\n'))

    def parse_model_file(self, model_file):
        """Parses model file"""
        with open(model_file, 'r') as f:
            cur_term = None
            for line in f:
                line_list = line.split()
                if line_list[0] == 'FEATURES':
                    label = line_list[3]
                    cur_term = label
                    continue
                else:
                    feat = line_list[0]
                    if feat not in self.weights:
                        self.weights[feat] = {}
                    self.weights[feat][cur_term] = float(line_list[1])

    def write_to_sys_file(self, sent_idx, word_idx, f):
        """Write predictions to sys output file"""
        final_output = []
        final_output.append(str(sent_idx + 1) + '-' + str(word_idx) + '-')
        final_output.append(self.final_probs[sent_idx][word_idx][0])
        final_output.append(str(self.final_probs[sent_idx][word_idx][1]))
        final_output.append(self.test_set[sent_idx][word_idx][0])
        final_output.append(self.test_labels[sent_idx][word_idx])
        f.write(" ".join(final_output))
        f.write('\n')
