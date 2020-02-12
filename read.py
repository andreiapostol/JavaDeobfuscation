#!/usr/bin/env python

import os
import sys

from graph_pb2 import Graph
from graph_pb2 import FeatureNode

def main(path):
  with open(path, "rb") as f:
    g = Graph()
    g.ParseFromString(f.read())
    token_count = len(list(filter(lambda n:n.type in 
                             (FeatureNode.TOKEN,
                              FeatureNode.IDENTIFIER_TOKEN), g.node)))
    token_count = len(set(g.node.startLineNumber))
                              # startLineNumber
    print("%s contains %d tokens" % (g.sourceFile, token_count))

if __name__ == "__main__":
  main(sys.argv[1])
