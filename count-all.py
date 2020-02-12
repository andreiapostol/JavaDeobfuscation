#!/usr/bin/env python

import os
import sys
import glob
from graph_pb2 import Graph
from graph_pb2 import FeatureNode

# Get statistics for the given path
def count_one(path):
  with open(path, "rb") as f:
    g = Graph()
    g.ParseFromString(f.read())
    token_count = len(list(filter(lambda n:n.type in 
                             (FeatureNode.TOKEN,
                              FeatureNode.IDENTIFIER_TOKEN), g.node)))
    # max_line = max(list(map(lambda n: n.endLineNumber, g.node)))
    max_line = g.ast_root.endLineNumber
    javadoc_comments = len(list(filter(lambda n:n.type == FeatureNode.COMMENT_JAVADOC, g.node)))
    return token_count, max_line, javadoc_comments

def main(path):
  token_count, max_line, javadoc_comments = 0, 0, 0
  if os.path.isfile(path):
    token_count, max_line, javadoc_comments = count_one(path)
  else:
    # Recurse through all the files with the .java.proto extension in the folder
    for filename in glob.iglob(path + '**/*.java.proto', recursive=True):
      n_token_count, n_max_line, n_javadoc_comments = count_one(filename)
      token_count += n_token_count
      max_line += n_max_line
      javadoc_comments += n_javadoc_comments
  print("%s contains %d tokens, %d lines and %d JavaDoc comments" % (path, token_count, max_line, javadoc_comments))
    
      

if __name__ == "__main__":
  main(sys.argv[1])
