import sys

from graph_pb2 import Graph
from graph_pb2 import FeatureNode, FeatureEdge
from method_tokens import tokenize_methods_for_file, get_source_dict_graph, get_id_to_node_graph, tokenize_methods_for_graph
from ast_traversal_helpers import *
from get_members_and_types import get_type_mapping, is_similar_type

import numpy as np
import string
import re

import pygraphviz as pgv

def handle_single_member_select(ms, id_mapping, source_mapping):
    branches = get_subtrees_based_on_function(ms, id_mapping, source_mapping, \
            lambda that_node : not is_terminal(that_node) and that_node != ms, set(), 2)
    if (len(branches) < 2):
        return []
    left_branch = branches[0]
    right_branch = branches[1]
    lb_variables = get_terminal_variables(left_branch, id_mapping, source_mapping)
    rb_variables = get_terminal_variables(right_branch, id_mapping, source_mapping)
    if (len(rb_variables) == 1 and len(lb_variables) == 1):
        return [(lb_variables[0].contents, rb_variables[0].contents, ms.contents, 0)]
    inner_left = handle_member_selects(left_branch, id_mapping, source_mapping)
    if (len(rb_variables) == 0 or len(inner_left) == 0):
        return inner_left
    last = inner_left[-1]
    inner_left.append((last[1], rb_variables[-1].contents, ms.contents, 0))
    return inner_left

def handle_member_selects(node, id_mapping, source_mapping):
    member_selects = get_member_select_dependencies(node, id_mapping, source_mapping)
    sol = []
    for ms in member_selects:
        sol.extend(handle_single_member_select(ms, id_mapping, source_mapping))
    return sol

def get_names(variables, getter = lambda x : x.contents, condition = lambda x : True):
    sol = []
    for variable in variables:
        if condition(variable):
            sol.append(getter(variable))
    return sol

def is_addition_or_substraction(op):
    return op == "PLUS" or op == "MINUS" or op == "PLUS_ASSIGNMENT" or op == "MINUS_ASSIGNMENT"

def is_multiplication_or_division(op):
    return op == "MULTIPLY" or op == "DIVIDE" or op == "MULTIPLY_ASSIGNMENT" or op == "DIVIDE_ASSIGNMENT"

def commutable_operations(inner, outer):
    if (len(inner) == 0):
        return True
    if (outer == "ASSIGNMENT"):
        return True
    verify_function = None
    if is_addition_or_substraction(outer):
        verify_function = is_addition_or_substraction
    elif is_multiplication_or_division(outer):
        verify_function = is_multiplication_or_division
    else:
        return False
    for stmt in inner:
        if not verify_function(stmt):
            return False
    return True
    
def get_assignment_dependencies(statement_type_node, id_mapping, source_mapping, variable_types = None, level = 0):
    branches = get_statement_branches(statement_type_node, id_mapping, source_mapping)
    sol = []
    if (statement_type_node.contents == "IDENTIFIER"):
        return sol
    if len(branches) == 2:
        left_variables = get_terminal_variables(branches[0], id_mapping, source_mapping)
        right_variables = get_terminal_variables(branches[1], id_mapping, source_mapping)
        lv = get_names(left_variables)
        rv = get_names(right_variables)
        
        if len(right_variables) == 1:
            for lv in left_variables:
                    sol.append((lv.contents, right_variables[0].contents, statement_type_node.contents, level))
        else:
            inner_dependencies = get_dependencies(branches[1], id_mapping, source_mapping, level + 1)
            if (len(left_variables) == 1):
                if (commutable_operations(list(set(map(lambda x : x[2], inner_dependencies))), statement_type_node.contents) \
                    and len(inner_dependencies) > 0 and variable_types != None):
                    for inner in inner_dependencies:
                        if is_similar_type(left_variables[0].contents, inner[0], variable_types):
                            sol.append((left_variables[0].contents, inner[0], statement_type_node.contents, level))
                    last = inner_dependencies[-1]
                    if is_similar_type(left_variables[0].contents, last[1], variable_types):
                        sol.append((left_variables[0].contents, last[1], statement_type_node.contents, level))
                else:
                    sol.append((left_variables[0].contents, '_', statement_type_node.contents, level))
            sol.extend(inner_dependencies)
    elif len(branches) == 1:
        branch = branches[0]
        variables = get_terminal_variables(branch, id_mapping, source_mapping)
        if (len(variables) == 1):
            sol.append((variables[0].contents, '1', statement_type_node.contents, level))
    return sol

def get_existing_dependencies(root, id_mapping, source_mapping):
    all_edges_arrays = source_mapping.values()
    dependencies = []
    for edge_array in all_edges_arrays:
        for edge in edge_array:
            if edge.type in (FeatureEdge.COMPUTED_FROM, FeatureEdge.LAST_WRITE):
                from_node = id_mapping[edge.sourceId]
                to_node = id_mapping[edge.destinationId]
                dependencies.append((from_node.contents, to_node.contents, \
                "COMPUTED_FROM" if edge.type == FeatureEdge.COMPUTED_FROM \
                    else "LAST_WRITE", 0))
    return dependencies
    
def get_boolean_verifiers_array(node, id_mapping, source_mapping):
    return get_subtrees_based_on_function(node, id_mapping, source_mapping, \
        is_boolean_verifier_token, set())

def get_dependencies_from_expression(node, id_mapping, source_mapping, variable_types = None, level = 0):
    dft = []
    if not is_expression_node(node):
        return dft
    edges = source_mapping.get(node.id)
    if edges == None or len(edges) != 1:
        return dft
    statement_type_node = id_mapping[edges[0].destinationId]
    statement_type = statement_type_node.contents
    return get_assignment_dependencies(statement_type_node, id_mapping, source_mapping, variable_types, level)

def get_dependencies_from_boolean(node, id_mapping, source_mapping, level = 0):
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
            dft.append((lt.contents, rt.contents, statement_type, level))
    return dft

def get_method_argument_dependencies(node, id_mapping, source_mapping):
    parameters_node = get_parameters_from_method(node, id_mapping, source_mapping)
    method_name = get_variable_name(node, id_mapping, source_mapping)
    dependencies = []
    if method_name == None or method_name == '':
        return dependencies
    if parameters_node == None or len(parameters_node) < 1:
        return dependencies
    parameters_node = parameters_node[0]
    variables_nodes = get_variables(parameters_node, id_mapping, source_mapping)
    if variables_nodes == None or len(variables_nodes) < 1:
        return dependencies
    for variable_node in variables_nodes:
        variable_name = get_variable_name(variable_node, id_mapping, source_mapping)
        if variable_name == None or variable_name == '':
            continue
        dependencies.append((method_name, variable_name, 'HAS_ARG', 0))
    return dependencies

def get_all_method_argument_dependencies(root, id_mapping, source_mapping):
    method_nodes = get_methods(root, id_mapping, source_mapping)
    dependencies = []
    for method_node in method_nodes:
        current_dependencies = get_method_argument_dependencies(method_node, id_mapping, source_mapping)
        dependencies.extend(current_dependencies)
    return dependencies

def get_dependencies(root, id_mapping, source_mapping, variable_types = None, level = 0):
    top_expressions = get_expression_operation_arrays(root, id_mapping, source_mapping)
    top_booleans = get_boolean_verifiers_array(root, id_mapping, source_mapping)
    sol = []
    for top_expression in top_expressions:
        sol.extend(get_dependencies_from_expression(top_expression, id_mapping, source_mapping, variable_types, level))
    for top_boolean in top_booleans:
        sol.extend(get_dependencies_from_boolean(top_boolean, id_mapping, source_mapping, level))
    sol.extend(handle_member_selects(root, id_mapping, source_mapping))
    return sol

def remove_level_information(dependencies):
    result = []
    for dependency in dependencies:
        result.append(dependency[:-1])
    return result

def get_all_dependencies(g, id_mapping = None, source_mapping = None, variable_types = None):
    if id_mapping == None:
        id_mapping = get_id_to_node_graph(g)
    if source_mapping == None:
        source_mapping = get_source_dict_graph(g)
    root = g.ast_root
    
    dependencies = get_dependencies(root, id_mapping, source_mapping, variable_types)
    existent_dependencies = get_existing_dependencies(root, id_mapping, source_mapping)
    dependencies.extend(existent_dependencies)
    method_dependencies = get_all_method_argument_dependencies(root, id_mapping, source_mapping)
    dependencies.extend(method_dependencies)
    dependencies = remove_level_information(dependencies)

    return list(set(dependencies))

def shorten(s):
    if s == "ASSIGNMENT":
        return "ASGN"
    if s == "EQUAL":
        return "EQ"
    if s == "GREATER":
        return "GR"
    if s == "LESS":
        return "LS"
    if s == "MULTIPLY":
        return "MLY"
    if s == "PLUS":
        return "PS"
    if s == "MINUS":
        return "MNS"
    if s == "INCREMENT":
        return "INC"
    if s == "DECREMENT":
        return "DECR"
    if s == "POSTFIX":
        return "PFX"
    if s == "MEMBER":
        return "MBR"
    if s == "METHOD":
        return "MTD"
    if s == "SELECT":
        return "SCT"
    if s == "DIVIDE":
        return "DVD"
    if s == "INVOCATION":
        return "INVC"
    if s == "COMPUTED":
        return "CMP"
    if s == "WRITE":
        return "WRT"
    if s.find("_") == -1:
        return s
    parts = s.split("_")
    if (len(parts) == 1):
        return s
    result = ""
    for part in parts:
        result += shorten(part)
        result += "_"
    return result[:-1]

def get_color(dependency_type):
    if is_some_assignment(dependency_type):
        return 'green'
    elif is_addition_or_substraction(dependency_type) or is_multiplication_or_division(dependency_type):
        return 'orange'
    elif is_boolean_verifier(dependency_type):
        return 'purple'
    elif is_postfix(dependency_type):
        return 'yellow'
    elif is_existent(dependency_type):
        return 'black'
    elif dependency_type == "HAS_ARG":
        return 'pink'
    elif dependency_type == 'MEMBER_SELECT':
        return "red"
    return 'grey'

def get_visual_graph(dependencies, variable_types):
    G=pgv.AGraph(directed=True, strict=True, overlap=False, splines='true', nodesep="2", forcelabels="true")
    for variable_name in variable_types.keys():
        G.add_node(variable_name, color='blue')
    for (start, end, edge_type) in dependencies:
        G.add_edge(start, end, color=get_color(edge_type), decorate=False)
        new_edge = G.get_edge(start, end)
        new_edge.attr['label'] = shorten(edge_type)
    return G

if __name__ == "__main__":
    filePath = sys.argv[1]
    with open(filePath, "rb") as f:
        graph = Graph()
        graph.ParseFromString(f.read())
        id_mapping = get_id_to_node_graph(graph)
        source_mapping = get_source_dict_graph(graph)
        variable_types = get_type_mapping(graph, id_mapping, source_mapping)
        dependencies = get_all_dependencies(graph, id_mapping, source_mapping, variable_types)
        visual_graph = get_visual_graph(dependencies, variable_types)

        visual_graph.layout()
        visual_graph.draw('dependencies.png', prog='circo')
