from graph_pb2 import Graph
from graph_pb2 import FeatureNode, FeatureEdge
from method_tokens import tokenize_methods_for_file, get_source_dict_graph, get_id_to_node_graph, tokenize_methods_for_graph

def is_boolean_verifier(s):
    return s in ("LESS_THAN", "NOT_EQUAL_TO", "GREATER_THAN", "EQUAL_TO", "GREATER_THAN_EQUAL", "LESS_THAN_EQUAL")

def is_some_assignment(s):
    return s.find("ASSIGNMENT") != -1

def is_postfix(s):
    return s.find("POSTFIX") != -1

def is_existent(s):
    return s in ("COMPUTED_FROM", "LAST_WRITE")

def is_terminal(node):
    return node == None or (node.type in (FeatureNode.TOKEN, FeatureNode.IDENTIFIER_TOKEN, FeatureNode.TYPE))

def is_expression_node(node):
    return node.type == FeatureNode.FAKE_AST and node.contents == "EXPRESSION"

def is_boolean_verifier_token(node):
    return is_boolean_verifier(node.contents)

def is_variable_node(node):
    return not is_terminal(node) and node.contents == "VARIABLE"

def is_operation_token(node):
    return node.contents in ("PLUS", "MINUS", "MULTIPLY", "DIVIDE")

def is_member_select_token(node):
    return node.contents == "MEMBER_SELECT"

def is_class_node(node):
    return not is_terminal(node) and node.contents == "CLASS"

def is_method_node(node):
    return not is_terminal(node) and node.contents == "METHOD"

def is_type_node(node):
    return not is_terminal(node) and node.contents in ("TYPE", "RETURN_TYPE")

def is_name_node(node):
    return not is_terminal(node) and node.contents in ("NAME", "SIMPLE_NAME")

def get_terminal_variables(node, id_mapping, source_mapping):
    return get_subtrees_based_on_function(node, id_mapping, source_mapping, \
        lambda node : is_terminal(node) and node.type == FeatureNode.IDENTIFIER_TOKEN, set())

def get_all_terminals(node, id_mapping, source_mapping):
    return get_subtrees_based_on_function(node, id_mapping, source_mapping, \
        lambda node : is_terminal(node), set())

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

