import pickle
import sys
import numpy as np

def load_dependencies(file_name):
    return pickle.load(open(file_name, "rb"))

def get_user_names(types_deps):
    all_names = dict()
    for type_dep in types_deps:
        current_types = type_dep["types"]
        if current_types != None:
            for name in current_types.keys():
                if name not in all_names:
                    all_names[name] = 0
                all_names[name] = all_names[name] + 1
    total_user_nodes = sum(all_names.values())
    return all_names, total_user_nodes

def get_edges_name_mapping(types_deps):
    edge_name_to_index_mapping = dict()
    edge_index_to_name_mapping = dict()
    current_id = 0
    for type_dep in types_deps:
        current_deps = type_dep["dependencies"]
        for (start, end, dep_type) in current_deps:
            if dep_type not in edge_name_to_index_mapping:
                edge_name_to_index_mapping[dep_type] = current_id
                edge_index_to_name_mapping[current_id] = dep_type
                current_id += 1
    return edge_name_to_index_mapping, edge_index_to_name_mapping

def get_nodelabels_and_newedges(types_deps, edge_name_to_index_mapping):
    new_edges = []
    node_labels = []
    for type_dep in types_deps:
        current_deps = type_dep["dependencies"]
        seen_name_dict = dict()
        for (start, end, edge_type) in current_deps:
            if start not in seen_name_dict:
                seen_name_dict[start] = len(node_labels)
                node_labels.append(start)
            if end not in seen_name_dict:
                seen_name_dict[end] = len(node_labels)
                node_labels.append(end)
            new_edges.append((seen_name_dict[start], seen_name_dict[end], \
                edge_name_to_index_mapping[edge_type]))
    return node_labels, new_edges

def get_adj_lists(new_edges, edge_name_to_index_mapping):
    adj_lists = [[] for _ in range(len(edge_name_to_index_mapping))]
    for (start_id, end_id, edge_type) in new_edges:
        adj_lists[edge_type].append([start_id, end_id])
    return adj_lists

def create_gnn_data(types_deps):
    all_user_names, total_user_nodes = get_user_names(types_deps)
    edge_name_to_index_mapping, edge_index_to_name_mapping = \
        get_edges_name_mapping(types_deps)
    node_labels, new_edges = get_nodelabels_and_newedges(types_deps, edge_name_to_index_mapping)
    adj_lists = get_adj_lists(new_edges, edge_name_to_index_mapping)
    gnn_data = dict()
    gnn_data["node_labels"] = node_labels
    gnn_data["adj_lists"] = adj_lists
    gnn_data["edge_name_to_index_mapping"] = edge_name_to_index_mapping
    gnn_data["edge_index_to_name_mapping"] = edge_index_to_name_mapping
    return gnn_data
    # min_length = max(map(lambda x : len(x), adj_lists))
    # for (index, l) in enumerate(adj_lists):
    #     if len(l) == min_length:
    #         print(index, edge_index_to_name_mapping[index], len(l))
    #         break

def main(file_name = "all_dependencies(<1MB).dat"):
    mapping = load_dependencies(file_name)
    gnn_data = create_gnn_data(mapping)
    save_name = file_name.replace(".dat", ".gnn")
    pickle.dump(gnn_data, open(save_name, "wb"))

if __name__ == "__main__":
    if sys.argv != None and len(sys.argv) >= 2:
        main(sys.argv[1])
    else:
        main()