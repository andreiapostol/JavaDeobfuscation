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
    occurences = dict()
    edge_index_to_name_mapping = dict()
    current_id = 0
    for type_dep in types_deps:
        current_deps = type_dep["dependencies"]
        for (start, end, dep_type) in current_deps:
            if dep_type not in edge_name_to_index_mapping:
                edge_name_to_index_mapping[dep_type] = current_id
                edge_index_to_name_mapping[current_id] = dep_type
                current_id += 1
            if dep_type not in occurences:
                occurences[dep_type] = 0
            occurences[dep_type] += 1
    return edge_name_to_index_mapping, edge_index_to_name_mapping, occurences

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

# def create_gnn_data(types_deps):
#     all_user_names, total_user_nodes = get_user_names(types_deps)
#     edge_name_to_index_mapping, edge_index_to_name_mapping = \
#         get_edges_name_mapping(types_deps)
    
#     node_labels, new_edges = get_nodelabels_and_newedges(types_deps, edge_name_to_index_mapping)
#     adj_lists = get_adj_lists(new_edges, edge_name_to_index_mapping)
#     gnn_data = dict()
#     gnn_data["node_labels"] = node_labels
#     gnn_data["adj_lists"] = adj_lists
#     gnn_data["edge_name_to_index_mapping"] = edge_name_to_index_mapping
#     gnn_data["edge_index_to_name_mapping"] = edge_index_to_name_mapping
#     return gnn_data

def filter_unknown_nodes(all_graphs, new_length):
    id_to_usecases = dict()
    for i in tqdm(range(len(graphs))):
        cur_graph = graphs[i]
        if (cur_graph["user_defined_nodes_number"] > 0):
            nodes = cur_graph["nodes"]
            for node in nodes:
                if node not in id_to_usecases:
                    id_to_usecases[node] = 0
                id_to_usecases[node] += 1
    sorted_ids = sorted(id_to_usecases.items(), key=lambda kv: -1 * kv[1])
    most_k_used = dict()
    for i in range(new_length - 1):
        most_k_used[sorted_ids[i][0]] = i
    return most_k_used

def get_all_tokens_mapping(graphs, new_length = 5000):
    from_name_to_id = dict()
    from_id_to_name = []
    from_id_to_usecases = dict()
    current_id = 0

    for graph in graphs:
        types = graph["types"]
        for t in types.keys():
            if t not in from_name_to_id:
                from_name_to_id[t] = len(from_id_to_name)
                from_id_to_name.append(t)
            if from_name_to_id[t] not in from_id_to_usecases:
                from_id_to_usecases[from_name_to_id[t]] = 0
            from_id_to_usecases[from_name_to_id[t]] += 1

    total_user_defined = len(from_id_to_name)

    for graph in graphs:
        dependencies = graph["dependencies"]
        for (start, end, edge_type) in dependencies:
            if start not in from_name_to_id:
                from_name_to_id[start] = len(from_id_to_name)
                from_id_to_name.append(start)
            if end not in from_name_to_id:
                from_name_to_id[end] = len(from_id_to_name)
                from_id_to_name.append(end)

    sorted_ids = sorted(from_id_to_usecases.items(), key=lambda kv: -1 * kv[1])[:new_length]
    sorted_ids = sorted(sorted_ids, key=lambda kv: -1 if kv[0] <= total_user_defined else 1)
    print(sorted_ids)
    renewed_name_to_id = dict()
    renewed_id_to_name = []

    for i in range(len(sorted_ids)):
        renewed_name_to_id[from_id_to_name[sorted_ids[i][0]]] = i
        renewed_id_to_name.append(from_id_to_name[sorted_ids[i][0]])
    
    renewed_total_user_defined = 0
    for i in range(len(sorted_ids)):
        if sorted_ids[i][0] <= total_user_defined:
            renewed_total_user_defined += 1
    renewed_id_to_name.append("UNKNOWN")
    return renewed_name_to_id, renewed_id_to_name, renewed_total_user_defined
    # return from_name_to_id, from_id_to_name, total_user_defined    

def create_graph(graphs, name_to_id_mapping, edge_name_to_index_mapping):
    new_graphs = []

    for graph in graphs:
        total_nodes = []
        user_defined_nodes_number = 0
        added_nodes_set = set()
        types = graph["types"]
        deps = graph["dependencies"]
        new_deps = []
        adj_lists = [[] for _ in edge_name_to_index_mapping.keys()]

        for t in types.keys():
            if t not in added_nodes_set and t in name_to_id_mapping:
                added_nodes_set.add(t)
                total_nodes.append(name_to_id_mapping.get(t, len(name_to_id_mapping)))
        user_defined_nodes_number = len(added_nodes_set)

        for (start, end, dep_type) in deps:
            if start not in name_to_id_mapping and end not in name_to_id_mapping:
                continue
            if start not in added_nodes_set:
                added_nodes_set.add(start)
                total_nodes.append(name_to_id_mapping.get(start, len(name_to_id_mapping)))
            if end not in added_nodes_set:
                added_nodes_set.add(end)
                total_nodes.append(name_to_id_mapping.get(end, len(name_to_id_mapping)))
            if dep_type in edge_name_to_index_mapping:
                current_edge = (name_to_id_mapping.get(start, len(name_to_id_mapping)), name_to_id_mapping.get(end, len(name_to_id_mapping)), \
                    edge_name_to_index_mapping[dep_type])
                new_deps.append(current_edge)
                adj_lists[edge_name_to_index_mapping[dep_type]].append((current_edge[0], current_edge[1]))

        cur_graph = dict()
        cur_graph["nodes"] = total_nodes
        cur_graph["user_defined_nodes_number"] = user_defined_nodes_number
        cur_graph["edges"] = new_deps
        cur_graph["adj_lists"] = adj_lists
        new_graphs.append(cur_graph)
    
    return new_graphs

def get_n_edges_mapping(occurences, to_keep = 20):
    first_to_keep = sorted(occurences.items(), key=lambda x : -1 * x[1])[1:to_keep+1]
    new_name_to_id = dict()
    new_id_to_name = []
    current_index = 0
    for (key, _) in first_to_keep:
        new_name_to_id[key] = len(new_id_to_name)
        new_id_to_name.append(key)
    return new_name_to_id, new_id_to_name

def create_multiple_graphs_data(graphs):
    all_user_variables_mapping, total_user_variables_used = get_user_names(graphs)
    _, _, occurences = get_edges_name_mapping(graphs)
    edge_name_to_index_mapping, edge_index_to_name_mapping = get_n_edges_mapping(occurences, 25)
    name_to_id_mapping, id_to_name_list, total_user_defined = get_all_tokens_mapping(graphs, 2500)
    to_return = dict()
    to_return["name_to_id_mapping"] = name_to_id_mapping
    to_return["ids_to_names"] = id_to_name_list
    to_return["total_user_defined_nodes"] = total_user_defined
    to_return["number_of_edges"] = len(edge_index_to_name_mapping)
    to_return["edge_name_to_id"] = edge_name_to_index_mapping
    to_return["edge_id_to_name"] = edge_index_to_name_mapping

    new_graphs = create_graph(graphs, name_to_id_mapping, edge_name_to_index_mapping)
    index = len(new_graphs)-1
    old = graphs[index]
    newg = new_graphs[index]
    print(old["types"])
    print(newg["nodes"], newg["user_defined_nodes_number"])
    print(newg["adj_lists"])
    print(to_return["total_user_defined_nodes"])
    to_return["graphs"] = new_graphs
    return to_return

def main(file_name = "all_dependencies(<1MB).dat", concatenate=False):
    mapping = load_dependencies(file_name)
    gnn_data = None
    if concatenate:
        gnn_data = create_gnn_data(mapping)
    else:
        gnn_data = create_multiple_graphs_data(mapping)
    # save_name = "SerializedData/" + file_name.replace(".dat", "(conc).gnn")
    save_name = "GNN-Implementation-TF2/data/deobfuscation/train.pkl.gz"
    pickle.dump(gnn_data, open(save_name, "wb"))

if __name__ == "__main__":
    if sys.argv != None and len(sys.argv) >= 2:
        main(sys.argv[1])
    else:
        main()