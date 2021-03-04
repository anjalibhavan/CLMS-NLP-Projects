"""Main function for POS Tagging using MaxEnt + BeamSearch"""
import sys
import copy
import config
from utils import calc_base_weights, get_topn_nodes, prune_nodes, get_class_label, calc_tag_probs
from file_parser import FileParser
from tree_node import Node

# Declare all variables
TEST_FILE = sys.argv[1]
BOUNDARY_FILE = sys.argv[2]
MODEL_FILE = sys.argv[3]
SYS_OUTPUT = sys.argv[4]
config.BEAM_SIZE = int(sys.argv[5])
config.TOPN = int(sys.argv[6])
config.TOPK = int(sys.argv[7])
data = FileParser(TEST_FILE, BOUNDARY_FILE, MODEL_FILE)

# Beam search
for sent_idx in range(len(data.test_set)):
    total_paths = {}
    data.final_probs[sent_idx] = {}
    root = Node('BOS', None, None, 0)
    history = copy.deepcopy(data.test_set[sent_idx][0][1])
    tag1 = 'prevTwoTags=BOS+BOS'
    tag2 = 'prevT=BOS'
    
    base_weights = calc_base_weights(data.weights, data.tagset, history)
    tag_dict = calc_tag_probs(data.weights, base_weights, tag1, tag2)
    top_n = get_topn_nodes(tag_dict, root)
    pruned_nodes = prune_nodes(top_n, total_paths)
    data.final_probs[sent_idx][0] = get_class_label(pruned_nodes)

    for word_idx in range(1, data.sentence_lengths[sent_idx]):
        nodelist = []
        history = copy.deepcopy(data.test_set[sent_idx][word_idx][1])
        base_weights = calc_base_weights(data.weights, data.tagset, history)
        for node in total_paths[word_idx - 1]:
            tag1 = 'prevTwoTags=' + node.parent.tag + '+' + node.tag
            tag2 = 'prevT=' + node.tag
            tag_dict = calc_tag_probs(data.weights, base_weights, tag1, tag2)
            top_n = get_topn_nodes(tag_dict, node)
            nodelist += top_n
        pruned_nodes = prune_nodes(nodelist, total_paths)
        data.final_probs[sent_idx][word_idx] = get_class_label(pruned_nodes)


# Writing to sys output
with open(SYS_OUTPUT, 'w') as f:
    f.write('%%%%% test data:')
    f.write('\n')
    for sent_idx, sentence in enumerate(data.final_probs):
        for word_idx, word in enumerate(data.final_probs[sent_idx]):
            data.write_to_sys_file(sent_idx, word_idx, f)
