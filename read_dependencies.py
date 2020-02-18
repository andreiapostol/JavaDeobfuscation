from glob import glob
from os.path import join, exists
from tqdm import tqdm
import time
import pickle
import sys
from get_dependencies import get_visual_graph

def read_all_dependencies(main_path):
    joined = main_path + '/**/*.deps'
    all_files = glob(joined, recursive=True)
    all_deps = []
    for f in all_files:
        current_deps = pickle.load(open(f, "rb"))
        all_deps.append(current_deps)
    return all_deps

def draw_dependency(depdict, draw_path="read_dep_example.png"):
    types = depdict["types"]
    dependencies = depdict["dependencies"]
    visual_graph = get_visual_graph(dependencies, types)
    visual_graph.layout()
    visual_graph.draw(draw_path, prog='circo')

def main(directory = "./extracted-features"):
    start = time.time()
    all_deps = read_all_dependencies(directory)
    save_name = "all_dependencies.dat"
    if not exists(save_name):
        pickle.dump(all_deps, open(save_name, "wb"))
    draw_dependency(all_deps[123])

if __name__ == "__main__":
    if sys.argv != None and len(sys.argv) >= 2:
        main(sys.argv[1])
    else:
        main()