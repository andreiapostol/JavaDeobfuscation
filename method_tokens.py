#!/usr/bin/env python

import os
import sys

from graph_pb2 import Graph
from graph_pb2 import FeatureNode

# Method for checking if a node is a method
def isMethod(node):
    return node.type == FeatureNode.AST_ELEMENT and node.contents == "METHOD"

# Method that decides whether a node is a token
def isToken(node):
    return node.type in (FeatureNode.TOKEN, FeatureNode.IDENTIFIER_TOKEN)

# Retrieve token leaf nodes, by DFS
def get_leaf_nodes(nodeId, sourceDict, nodeDict, visited):
    if (nodeId in visited):
        return []
    visited.add(nodeId)
    if (nodeId == None or nodeDict.get(nodeId) == None):
        return []
    if (nodeDict.get(nodeId).type in [FeatureNode.TOKEN, FeatureNode.IDENTIFIER_TOKEN]):
        return [nodeDict.get(nodeId)]
    edgeTo = sourceDict.get(nodeId)
    if (edgeTo == None):
        return []
    to_return = []
    for edge in edgeTo:
        to_return += get_leaf_nodes(edge.destinationId, sourceDict, nodeDict, visited)
    return to_return

# Reorder leaf nodes from top to bottom
def reorder_leaves(leaves_arr, sourceDict, nodeDict):
    leaves_map = dict()
    for (index, node) in enumerate(leaves_arr):
        leaves_map[node.id] = index
    length = len(leaves_arr)
    index_sum = int(((length - 1) * length) / 2)
    for node in leaves_arr:
        if (node.id in sourceDict) and ((sourceDict[node.id][0]).destinationId in leaves_map):
            index_sum -= leaves_map[(sourceDict[node.id][0]).destinationId]
    current = leaves_arr[index_sum]
    to_return = []
    for _ in range(length):
        to_return.append(current)
        if current.id in sourceDict:
            current = nodeDict[(sourceDict[current.id][0]).destinationId]
        else:
            break
    return to_return

def tokenize_methods_for_graph(g, full = False):
    token_count = len(list(filter(lambda n:n.type in 
                                (FeatureNode.TOKEN,
                                FeatureNode.IDENTIFIER_TOKEN), g.node)))
    to_print_len = min(len(g.node), 100)
    sourceIdsInEdge = get_source_dict_graph(g)
    idsInNode = get_id_to_node_graph(g)
    all_results = []
    for node in g.node:
        if isMethod(node):
            initial_leaves = reorder_leaves(get_leaf_nodes(node.id, sourceIdsInEdge, idsInNode, set()), \
                                            sourceIdsInEdge, idsInNode)
            correct = [n if full else n.contents for n in filter(isToken, initial_leaves)]
            all_results.append(correct)
    return all_results

# Get tokens for given file
def tokenize_methods_for_file(path, full = False):
    with open(path, "rb") as f:
        g = Graph()
        g.ParseFromString(f.read())
        return tokenize_methods_for_graph(g, full)

def get_source_dict(path):
    with open(path, "rb") as f:
        g = Graph()
        g.ParseFromString(f.read())
        return get_source_dict_graph(g)

def get_source_dict_graph(g):
    sourceIdsInEdge = dict()
    for edge in g.edge:
        cur = sourceIdsInEdge.get(edge.sourceId, [])
        cur.append(edge)
        sourceIdsInEdge[edge.sourceId] = cur
    return sourceIdsInEdge

def get_id_to_node(path):
    with open(path, "rb") as f:
        g = Graph()
        g.ParseFromString(f.read())
        return get_id_to_node_graph(g)

def get_id_to_node_graph(g):
    idsInNode = dict()
    for node in g.node:
        idsInNode[node.id] = node
    return idsInNode


def write_arrays_to_file(file_path, results):
    with open(file_path, 'w') as f:
        for result in results:
            for cur in result:
                f.write(cur + " ")
            f.write("\n")

def main(path):
    all_results = tokenize_methods_for_file(path)
    write_name = os.path.basename(path)
    result_file_name = "method-tokenized (" + write_name + ").txt"
    write_arrays_to_file(result_file_name, all_results)

if __name__ == "__main__":
  main(sys.argv[1])