#!/usr/bin/env python

import os
import sys
import glob
from graph_pb2 import Graph
from graph_pb2 import FeatureNode

# Escape \n
def escape(token):
    return token.encode("unicode_escape").decode("utf-8")

# Retrieve token leaf nodes, by DFS
def isToken(node):
    return node.type in (FeatureNode.TOKEN, FeatureNode.IDENTIFIER_TOKEN)

# Get first n tokens from a file
def get_n_tokens(path, n):
    with open(path, "rb") as f:
        g = Graph()
        g.ParseFromString(f.read())
        to_print_len = min(len(g.node), n)
        token_list = []
        for node in filter(isToken, g.node):
            token_list.append(node.contents)
            to_print_len -= 1
            if to_print_len <= 0:
                break
        return token_list

def append_arrays_to_file(file_path, results):
    with open(file_path, 'a') as f:
        for result in results:
            for cur in result:
                f.write(cur + " ")
            f.write("\n")

def main(path):
    results = []
    n = 100
    output_path = "first_100_tokens.txt"
    open(output_path, 'w').close()
    if os.path.isfile(path):
        token_list = get_n_tokens(path, n)
        results.append(token_list)
    else:
        real = 0
        # Recurse through all the files with the .java.proto extension in the folder
        for filename in glob.iglob(path + '**/*.java.proto', recursive=True):
            results.append(get_n_tokens(filename, n))
            real += 1
            # Minimize file interaction time while not overloading memory
            if (len(results) == 100):
                append_arrays_to_file(output_path, results)
                results = []
                if (real % 500 == 0):
                    print(real)

if __name__ == "__main__":
  main(sys.argv[1])