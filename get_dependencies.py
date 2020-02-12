import os
import sys

from graph_pb2 import Graph
from graph_pb2 import FeatureNode, FeatureEdge
from method_tokens import tokenize_methods_for_file, get_source_dict_graph, get_id_to_node_graph, tokenize_methods_for_graph

import numpy as np
import string
import re

import pygraphviz as pgv

def is_terminal(node):
    return node == None or (node.type in (FeatureNode.TOKEN, FeatureNode.IDENTIFIER_TOKEN))

def is_expression_node(node):
    return node.type == FeatureNode.FAKE_AST and node.contents == "EXPRESSION"

def is_boolean_verifier_token(node):
    return node.contents in ("LESS_THAN", "NOT_EQUAL_TO", "GREATER_THAN", "EQUAL_TO")

def get_subtrees_based_on_function(node, id_mapping, source_mapping, tree_function, visited, max_depth = 100000):
    sol = []
    if node.id in visited or max_depth <= 0:
        return sol
    visited.add(node.id)
    if tree_function(node):
        return [node]
    if is_terminal(node):
        return sol
    edges = source_mapping.get(node.id)
    if (edges != None and len(edges) > 0):
        for edge in edges:
            if (edge.type not in (FeatureEdge.NEXT_TOKEN, FeatureEdge.GUARDED_BY)):
                sol.extend(get_subtrees_based_on_function(id_mapping.get(edge.destinationId), \
                    id_mapping, source_mapping, tree_function, visited, max_depth - 1))
    return sol

# CONDITIONAL_AND
# MULTIPLY_ASSIGNMENT
# POSTFIX_DECREMENT
def get_if_arrays(root, id_mapping, source_mapping):
    return get_subtrees_based_on_function(root, id_mapping, source_mapping, \
        lambda node : node.type == FeatureNode.AST_ELEMENT and node.contents == "IF", set())

def get_expression_arrays(root, id_mapping, source_mapping):
    return get_subtrees_based_on_function(root, id_mapping, source_mapping, \
        is_expression_node, set())

def get_terminal_variables(node, id_mapping, source_mapping):
    return get_subtrees_based_on_function(node, id_mapping, source_mapping, \
        lambda node : is_terminal(node) and node.type == FeatureNode.IDENTIFIER_TOKEN, set())

def get_statement_branches(node, id_mapping, source_mapping):
    return get_subtrees_based_on_function(node, id_mapping, source_mapping, \
        lambda that_node : not is_terminal(that_node) and that_node != node, set())

def get_assignment_dependencies(statement_type_node, id_mapping, source_mapping):
    branches = get_statement_branches(statement_type_node, id_mapping, source_mapping)
    sol = []
    if len(branches) == 2:
        left_variables = get_terminal_variables(branches[0], id_mapping, source_mapping)
        right_variables = get_terminal_variables(branches[1], id_mapping, source_mapping)
        for lv in left_variables:
            for rv in right_variables:
                sol.append((lv.contents, rv.contents, statement_type_node.contents))
    return sol

def get_boolean_verifiers_array(node, id_mapping, source_mapping):
    return get_subtrees_based_on_function(node, id_mapping, source_mapping, \
        is_boolean_verifier_token, set())
# def get_conditional_dependencies(statement_type_node, id_mapping, source_mapping):
#     branches = get_statement_branches(statement_type_node, id_mapping, source_mapping)

def get_dependencies_from_expression(node, id_mapping, source_mapping):
    dft = []
    if not is_expression_node(node):
        return dft
    edges = source_mapping.get(node.id)
    if edges == None or len(edges) != 1:
        return dft
    statement_type_node = id_mapping[edges[0].destinationId]
    statement_type = statement_type_node.contents
    if statement_type.find("ASSIGNMENT") != -1:
       return get_assignment_dependencies(statement_type_node, id_mapping, source_mapping)
    return dft

def get_dependencies_from_boolean(node, id_mapping, source_mapping):
    dft = []
    if not is_boolean_verifier_token(node):
        return dft
    statement_type= node.contents
    left_branch = get_subtrees_based_on_function(node, id_mapping, source_mapping, \
        lambda node : node.contents == "LEFT_OPERAND", set(), 2)[0]
    right_branch = get_subtrees_based_on_function(node, id_mapping, source_mapping, \
        lambda node : node.contents == "RIGHT_OPERAND", set(), 2)[0]
    left_terminals = get_terminal_variables(left_branch, id_mapping, source_mapping)
    right_terminals = get_terminal_variables(right_branch, id_mapping, source_mapping)
    if (len(left_terminals) == 0 or len(right_terminals) == 0):
        return dft
    if (len(left_terminals) != 1 or len(right_terminals) != 1):
        return dft
    for lt in left_terminals:
        for rt in right_terminals:
            dft.append((lt.contents, rt.contents, statement_type))
    return dft
    

def get_dependencies(root, id_mapping, source_mapping):
    top_expressions = get_expression_arrays(root, id_mapping, source_mapping)
    top_booleans = get_boolean_verifiers_array(root, id_mapping, source_mapping)
    sol = []
    for top_expression in top_expressions:
        sol.extend(get_dependencies_from_expression(top_expression, id_mapping, source_mapping))
    for top_boolean in top_booleans:
        sol.extend(get_dependencies_from_boolean(top_boolean, id_mapping, source_mapping))
    return sol

def get_statement_subtrees(g):
    id_mapping = get_id_to_node_graph(g)
    source_mapping = get_source_dict_graph(g)
    root = g.ast_root
    
    dependencies = get_dependencies(root, id_mapping, source_mapping)
    visual_graph = get_visual_graph(dependencies)
    return dependencies, visual_graph
    # print(source_mapping)
    # print(terminal_variables)
    # print(ifs)


def get_color(dependency_type):
    if (dependency_type.find("ASSIGNMENT") != -1):
        return 'green'
    if (dependency_type in ("LESS_THAN", "NOT_EQUAL_TO", "GREATER_THAN", "EQUAL_TO")):
        return 'blue'
    return 'red'

def get_visual_graph(dependencies):
    G=pgv.AGraph(directed=True)
    for (start, end, edge_type) in dependencies:
        G.add_edge(start, end, color=get_color(edge_type))
        new_edge = G.get_edge(start, end)
        new_edge.attr['label'] = edge_type
    return G

if __name__ == "__main__":
    filePath = sys.argv[1]
    with open(filePath, "rb") as f:
        graph = Graph()
        graph.ParseFromString(f.read())
        dependencies, visual_graph = get_statement_subtrees(graph)
        print(dependencies)
        visual_graph.layout() # default to neato
        visual_graph.draw('dependencies.png')

