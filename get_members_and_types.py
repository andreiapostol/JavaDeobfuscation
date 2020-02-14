import sys

from graph_pb2 import Graph
from graph_pb2 import FeatureNode, FeatureEdge
from method_tokens import tokenize_methods_for_file, get_source_dict_graph, get_id_to_node_graph, tokenize_methods_for_graph
from ast_traversal_helpers import *

import numpy as np
import string
import re

def is_number(n):
    is_number = True
    try:
        num = float(n)
        is_number = num == num
    except ValueError:
        is_number = False
    return is_number

def is_number_type(Type):
    return Type in ('int', 'Integer', 'java.lang.Integer', \
        'double', 'Double', 'java.lang.Double', \
        'long', 'Long', 'java.lang.Long', \
        'short', 'Short', 'java.lang.Short', \
        'float', 'Float', 'java.lang.Short', \
        'char') or \
        is_number(Type)

def is_similar_type(first, second, type_mapping):
    if is_number_type(first) and is_number_type(second):
        return True
    if first not in type_mapping or second not in type_mapping:
        return False
    if first == second:
        return True
    return (type_mapping[first] == type_mapping[second]) or \
        (is_number_type(type_mapping[first]) and is_number_type(type_mapping[second]))

def curate(name):
    if name == "LBBRACKET":
        return "["
    elif name == "RBBRACKET":
        return "]"
    return name    

def compute_names_and_types(nodes, id_mapping, source_mapping):
    mapping = dict()
    for node in nodes:
        name = get_variable_name(node, id_mapping, source_mapping)
        Type = get_variable_type(node, id_mapping, source_mapping)
        if '' in (name, Type):
            continue
        mapping[name] = Type
    return mapping

def get_type_mapping(g, id_mapping = None, source_mapping = None):
    if id_mapping == None:
        id_mapping = get_id_to_node_graph(g)
    if source_mapping == None:
        source_mapping = get_source_dict_graph(g)
    root = g.ast_root
    all_members = get_variables(root, id_mapping, source_mapping)
    all_members.extend(get_classes(root, id_mapping, source_mapping))
    all_members.extend(get_methods(root, id_mapping, source_mapping))
    return compute_names_and_types(all_members, id_mapping, source_mapping)

if __name__ == "__main__":
    filePath = sys.argv[1]
    with open(filePath, "rb") as f:
        graph = Graph()
        graph.ParseFromString(f.read())
        type_mapping = get_type_mapping(graph)
        