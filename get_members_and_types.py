import sys

from graph_pb2 import Graph
from graph_pb2 import FeatureNode, FeatureEdge
from method_tokens import tokenize_methods_for_file, get_source_dict_graph, get_id_to_node_graph, tokenize_methods_for_graph
from ast_traversal_helpers import *

import numpy as np
import string
import re

def get_variables(node, id_mapping, source_mapping):
    return get_subtrees_based_on_function(node, id_mapping, source_mapping, \
        is_variable_node, set())

def get_classes(node, id_mapping, source_mapping):
    return get_subtrees_based_on_function(node, id_mapping, source_mapping, \
        is_class_node, set())

def get_methods(node, id_mapping, source_mapping):
    return get_subtrees_based_on_function(node, id_mapping, source_mapping, \
        is_method_node, set())

# def get_variables_types(variables, id_mapping, source_mapping):
#     mapping = dict()

def get_name(node, id_mapping, source_mapping):
    name_node = get_subtrees_based_on_function(node, id_mapping, source_mapping, \
        is_name_node, set(), 2)
    if name_node == None or len(name_node) == 0:
        return ''
    name_node = name_node[0]
    return get_all_terminals(name_node, id_mapping, source_mapping)[0].contents

def curate(name):
    if name == "LBBRACKET":
        return "["
    elif name == "RBBRACKET":
        return "]"
    return name

def get_type(node, id_mapping, source_mapping):
    type_node = get_subtrees_based_on_function(node, id_mapping, source_mapping, \
        is_type_node, set(), 2)
    if type_node == None or len(type_node) == 0:
        return ''
    type_node = type_node[0]
    type_name_nodes = get_all_terminals(type_node, id_mapping, source_mapping)
    # result = ''
    # result = type_name_nodes[-1].contents
    # print(type_node)
    # for type_name_node in type_name_nodes:
    #     print(type_name_node.contents)
    #     result += curate(type_name_node.contents)
    return type_name_nodes[-1].contents
    

def compute_names_and_types(nodes, id_mapping, source_mapping):
    mapping = dict()
    for node in nodes:
        name = get_name(node, id_mapping, source_mapping)
        Type = get_type(node, id_mapping, source_mapping)
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
        