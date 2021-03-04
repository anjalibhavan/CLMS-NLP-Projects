"""Contains tree node class"""
from dataclasses import dataclass

@dataclass
class Node:
    """Tree node class"""
    def __init__(self, tag, parent, cond_prob, cum_log_prob):
        self.tag = tag
        self.parent = parent
        self.cond_prob = cond_prob
        self.cum_log_prob = cum_log_prob
