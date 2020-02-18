from get_dependencies import get_types_and_dependencies
from get_members_and_types import get_type_mapping
from glob import glob
from os.path import join
from tqdm import tqdm
import time
import sys
import pickle

def read_all_proto_names(main_path):
    joined = main_path +'/**/*.proto'
    all_files = glob(joined, recursive=True)
    return all_files

def create_graphs_and_types(main_path):
    all_protos = read_all_proto_names(main_path)

    for i in tqdm(range(len(all_protos))):
        proto = all_protos[i]
        types, dependencies = get_types_and_dependencies(proto)
        
        save_name = proto.replace(".proto", ".deps")
        to_save = dict()
        to_save["types"] = types
        to_save["dependencies"] = dependencies
        pickle.dump(to_save, open(save_name, "wb"))

def main(directory = "./extracted-features"):
    create_graphs_and_types(directory)

if __name__ == "__main__":
    if sys.argv != None and len(sys.argv) >= 2:
        main(sys.argv[1])
    else:
        main()
