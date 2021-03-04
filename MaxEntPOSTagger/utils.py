"""Utility functions"""
from collections import Counter
from operator import itemgetter
import numpy as np
from tree_node import Node
import config


def myfunc(node):
    return 10**(node.cum_log_prob)


def fun(node):
    return np.math.log(node.cond_prob, 10) + node.parent.cum_log_prob


def calc_base_weights(weights, tagset, instance):
    """Get base weights for each instance"""
    tag_dict = Counter()
    for tag in tagset:
        final_sum = weights['<default>'][tag]
        for feature in instance:
            final_sum += weights[feature][tag]
        tag_dict[tag] = final_sum
    return tag_dict


def calc_tag_probs(weights, base_weights, feat1, feat2):
    """Calculate P(tag/history)"""
    tag_dict = Counter()
    for tag in base_weights:
        if feat1 in weights:
            tag_dict[tag] = base_weights[tag] + weights[feat1][tag]
        if feat2 in weights:
            tag_dict[tag] += weights[feat2][tag]
        tag_dict[tag] = np.exp(tag_dict[tag])
    return tag_dict


def get_topn_nodes(tag_dict, parent_node):
    """Get top N nodes based on tag probabilities"""
    top_n = []
    dictsum = sum(tag_dict.values())
    for tag_pair in sorted(tag_dict.items(), key=itemgetter(1), reverse=True)[:config.TOPN]:
        new_node = Node(tag_pair[0], parent_node, tag_pair[1]/dictsum,
                        np.math.log(tag_pair[1]/dictsum, 10) + parent_node.cum_log_prob)
        top_n.append(new_node)
    return top_n


def prune_nodes(nodelist, total_paths):
    """Get pruned nodes based on cumulative log probabilities"""
    top_k = sorted(nodelist, key=myfunc, reverse=True)[:config.TOPK]
    max_prob = top_k[0].cum_log_prob
    pruned_nodes = []
    for node in top_k:
        if node.cum_log_prob + config.BEAM_SIZE >= max_prob:
            pruned_nodes.append(node)
    total_paths[len(total_paths)] = pruned_nodes
    return pruned_nodes


def get_class_label(pruned_nodes):
    """Get class label from pruned nodes"""
    max_node = max(pruned_nodes, key=fun)
    return (max_node.tag, max_node.cond_prob)
