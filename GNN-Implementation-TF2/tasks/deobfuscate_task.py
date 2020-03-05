from collections import namedtuple
from typing import Any, Dict, List, Iterable, Iterator, NamedTuple

import tensorflow as tf
import numpy as np
import pickle
from dpu_utils.utils import RichPath, LocalPath
import os.path
from .sparse_graph_task import Sparse_Graph_Task, DataFold, MinibatchData
from utils import micro_f1
from tqdm import tqdm
import random

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
ALPHABET_DICT = {char: idx + 2 for (idx, char) in enumerate(ALPHABET)}  # "0" is PAD, "1" is UNK
ALPHABET_DICT["PAD"] = 0
ALPHABET_DICT["UNK"] = 1

class GraphSample(NamedTuple):
    adjacency_lists: List[np.ndarray]
    type_to_num_incoming_edges: np.ndarray
    node_features: np.ndarray
    types: np.ndarray
    nodes_mask: np.ndarray
    labels: np.ndarray

class Deobfuscate_Task(Sparse_Graph_Task):
    @classmethod
    def default_params(cls):
        params = super().default_params()
        params.update({
            'add_self_loop_edges': False,
            'tie_fwd_bkwd_edges': False,
            'use_graph': True,
            'activation_function': "relu",
            'out_layer_dropout_keep_prob': 1.0,
            'initial_node_feature_size': 100

        })
        return params

    @staticmethod
    def default_data_path() -> str:
        return "data/deobfuscation"

    @staticmethod
    def name() -> str:
        return "DEOBFUSCATE"

    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)

        # Things that will be filled once we load data:
        self.__num_edge_types = 0
        self.__initial_node_feature_size = self.params["initial_node_feature_size"]
        self.batch_graph_size = 250

        self.__num_labels = None
        self.all_user_nodes = None
        self.user_defined_nodes_number = None
        self.edge_mapping = None
        self.graphs = None
        self.embedding_layer = None

    def get_metadata(self) -> Dict[str, Any]:
        metadata = super().get_metadata()
        metadata['num_edge_types'] = self.__num_edge_types
        metadata['initial_node_feature_size'] = self.__initial_node_feature_size
        metadata['num_labels'] = self.__num_labels
        return metadata

    def restore_from_metadata(self, metadata: Dict[str, Any]) -> None:
        super().restore_from_metadata(metadata)
        self.__num_edge_types = metadata['num_edge_types']
        self.__initial_node_feature_size = metadata['initial_node_feature_size']
        self.__num_labels = metadata['num_labels']

    @property
    def num_edge_types(self) -> int:
        return self.__num_edge_types

    @property
    def initial_node_feature_size(self) -> int:
        return self.__initial_node_feature_size

    @property
    def has_test_data(self) -> bool:
        return DataFold.TEST in self._loaded_data

    def load_data(self, path: RichPath) -> None:

        all_loaded_graphs, properties = self.__load_data(path, data_fold=DataFold.TRAIN)
        print("INITIAL FEATURE SIZE " + str(self.__initial_node_feature_size))

        random.shuffle(all_loaded_graphs)
        self.all_user_nodes = properties["all_user_nodes"]
        self.user_defined_nodes_number = properties["user_defined_nodes_number"]
        # self.node_embedder = properties["node_embedder"]
        self.edge_mapping = properties["edge_mapping"]
        self.__num_labels = properties["__num_labels"] + 1
        self.__num_types = properties["__num_types"]
        self.__num_edge_types = properties["__num_edge_types"]
        # self.num_edge_types = self.__num_edge_types

        size = len(all_loaded_graphs)
        self._loaded_data[DataFold.TRAIN] = all_loaded_graphs[:int(size/2)]
        self._loaded_data[DataFold.VALIDATION] = all_loaded_graphs[int(size/2):2*int(size/2)]

        self.name_embedding_layer = tf.keras.layers.Embedding(input_dim=self.__num_labels + 1, output_dim = int(self.__initial_node_feature_size / 2))
        self.type_embedding_layer = tf.keras.layers.Embedding(input_dim=self.__num_types + 1, output_dim = int(self.__initial_node_feature_size / 2))

        # print(self._loaded_data[DataFold.TRAIN][0].node_features)
        print("Loaded all data! The number of diff nodes is " + str(self.__num_labels) + ", the number of diff edges is " + str(self.__num_edge_types) + \
            ", the total number of all graphs is " + str(len(all_loaded_graphs)) + ", number of types " + str(self.__num_types))
    
    def create_graph_sample(self, old_graph, num_edge_types, num_labels):
        old_adjacency_lists = old_graph["adj_lists"]
        old_nodes = old_graph["nodes"]
        old_types = old_graph["types"]

        node_name_ids = np.array(old_nodes, dtype=int)
        node_type_ids = np.array(old_types, dtype=int)
        for i in range(len(old_nodes)):
            if i < old_graph["user_defined_nodes_number"]:
                node_name_ids[i] = num_labels + 1

        type_to_num_incoming_edges = np.zeros((num_edge_types, len(old_nodes)), dtype=int)
        
        old_node_to_new_node = dict()
        current_id = 0
        for old_node in old_nodes:
            if old_node not in old_node_to_new_node:
                old_node_to_new_node[old_node] = current_id
                current_id += 1

        new_adjacency_lists = []
        for (edge_id, edges) in enumerate(old_adjacency_lists):
            new_adjacency_lists.append(np.zeros((len(edges), 2), dtype=int))
            # new_adjacency_lists.append([])
            for (idx, (start_node, end_node)) in enumerate(edges):
                new_adjacency_lists[-1][idx][0] = start_node
                new_adjacency_lists[-1][idx][1] = end_node
                # new_adjacency_lists[-1].append((start_node, end_node))
                type_to_num_incoming_edges[edge_id][end_node] += 1

        nodes_mask = [i < old_graph["user_defined_nodes_number"] and node_id != num_labels for (i, node_id) in enumerate(old_nodes)]
        nodes_mask = np.array(nodes_mask, dtype=bool)

        labels = np.array(old_nodes, dtype=int)

        return GraphSample(new_adjacency_lists, \
            type_to_num_incoming_edges, \
            node_name_ids, \
            node_type_ids, \
            nodes_mask, \
            labels
            )

    def __load_data(self, data_dir: RichPath, data_fold: DataFold) -> List[GraphSample]:
        if data_fold == None:
            data_fold = "train"
        if data_fold == DataFold.TRAIN:
            data_name = "train"
        elif data_fold == DataFold.VALIDATION:
            data_name = "valid"
        elif data_fold == DataFold.TEST:
            data_name = "test"
        else:
            raise ValueError("Unknown data fold '%s'" % str(data_fold))

        print(" Loading DEOBFUSCATION %s data from %s." % (data_name, data_dir))

        if data_dir.join("%s-saved.pkl.gz" % data_name).is_file():
            read_data = data_dir.join("%s-saved.pkl.gz" % data_name).read_by_file_suffix()
            return read_data["all_graphs"], read_data["properties"]

        all_untensorised = data_dir.join("%s.pkl.gz" % data_name).read_by_file_suffix()

        graphs = all_untensorised["graphs"]
        
        properties = dict()
        properties["all_user_nodes"] = all_untensorised["name_to_id_mapping"]
        properties["user_defined_nodes_number"] = all_untensorised["total_user_defined_nodes"]
        properties["edge_mapping"] = all_untensorised["edge_name_to_id"]
        properties["__num_labels"] = len(properties["all_user_nodes"])
        properties["__num_edge_types"] = len(properties["edge_mapping"])
        properties["__num_types"] = len(all_untensorised["type_to_id"])

        all_graphs = []
        for i in tqdm(range(len(graphs))):
            old_graph = graphs[i]
            if (old_graph["user_defined_nodes_number"] > 0):
                all_graphs.append(self.create_graph_sample(old_graph, properties["__num_edge_types"], len(properties["all_user_nodes"])))

        print_graph_number = 2500
        print([all_untensorised["ids_to_names"][x] for x in all_graphs[print_graph_number].labels], all_graphs[print_graph_number].nodes_mask)
        print(all_graphs[print_graph_number])
        to_save = dict()
        to_save["all_graphs"] = all_graphs
        to_save["properties"] = properties
        data_dir.join("%s-saved.pkl.gz" % data_name).save_as_compressed_file(to_save)

        print("Saved modified data to %s-saved.pkl.gz" % data_name)

        return all_graphs, properties

    def load_eval_data_from_path(self, path: RichPath) -> Iterable[Any]:
        raise NotImplementedError()

    def make_task_input_model(self,
                              placeholders: Dict[str, tf.Tensor],
                              model_ops: Dict[str, tf.Tensor],
                              ) -> None:
        placeholders['initial_node_features'] = \
            tf.compat.v1.placeholder(dtype=tf.float32, shape=[None], name='initial_node_features')
        placeholders['adjacency_lists'] = \
            [tf.compat.v1.placeholder(dtype=tf.int32, shape=[None, 2], name='adjacency_e%s' % e)
                for e in range(self.__num_edge_types)]
        placeholders['type_to_num_incoming_edges'] = \
            tf.compat.v1.placeholder(dtype=tf.float32, shape=[self.__num_edge_types, None], name='type_to_num_incoming_edges')
        placeholders['node_names'] = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None], name='node_names')
        placeholders['node_types'] = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None], name='node_types')
        # placeholders['labels'] = tf.compat.v1.placeholder(tf.int32, [None], name='labels')
        # placeholders['nodes_mask'] = tf.compat.v1.placeholder(tf.float32, [None], name='nodes_mask')

        node_features = self.name_embedding_layer(placeholders['node_names'])
        type_features = self.type_embedding_layer(placeholders['node_types'])
        model_ops['initial_node_features'] = tf.concat(values=[node_features, type_features], axis=1)
        # self.embedding_layer(placeholders['initial_node_features'])
        model_ops['adjacency_lists'] = placeholders['adjacency_lists']
        model_ops['type_to_num_incoming_edges'] = placeholders['type_to_num_incoming_edges']

    def make_task_output_model(self,
                               placeholders: Dict[str, tf.Tensor],
                               model_ops: Dict[str, tf.Tensor],
                               ) -> None:
        placeholders['labels'] = tf.compat.v1.placeholder(tf.int32, [None], name='labels')
        placeholders['nodes_mask'] = tf.compat.v1.placeholder(tf.float32, [None], name='nodes_mask')

        final_node_representations = model_ops["final_node_representations"]
        output_label_logits = \
            tf.keras.layers.Dense(units=self.__num_labels, # can probably use only user defined labels
                                  use_bias=False,
                                  activation=None,
                                  name="OutputDenseLayer",
                                  )(final_node_representations)  # Shape [V, Classes]

        print("Output label logits shape is " + str(np.shape(output_label_logits)))

        num_masked_preds = tf.reduce_sum(input_tensor=tf.cast(placeholders['nodes_mask'], tf.float32))
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output_label_logits,
                                                                labels=placeholders['labels'])
        total_loss = tf.reduce_sum(input_tensor=losses * tf.cast(placeholders['nodes_mask'], tf.float32))
        
        most_likely_pred = tf.argmax(input=output_label_logits, axis=1, output_type=tf.int32)
        correct_preds = tf.equal(most_likely_pred, placeholders['labels'])
        num_masked_correct = tf.reduce_sum(input_tensor=tf.cast(correct_preds, tf.float32) * tf.cast(placeholders['nodes_mask'], tf.float32))
        accuracy = num_masked_correct / num_masked_preds
        tf.compat.v1.summary.scalar('accuracy', accuracy)

        model_ops['task_metrics'] = {
            'loss': total_loss / num_masked_preds,
            'total_loss': total_loss,
            'accuracy': accuracy,
        }
    
    def combine_graphs(self, first_graph, second_graph):
        old_nodes_length = np.shape(first_graph.node_features)[0]

        node_features = np.concatenate((first_graph.node_features, second_graph.node_features))
        types = np.concatenate((first_graph.types, second_graph.types))

        type_to_num_incoming_edges = np.zeros(((self.__num_edge_types, old_nodes_length + len(second_graph.node_features))))
        type_to_num_incoming_edges[:,:old_nodes_length] = first_graph.type_to_num_incoming_edges
        type_to_num_incoming_edges[:,old_nodes_length:] = second_graph.type_to_num_incoming_edges

        labels = np.concatenate((first_graph.labels, second_graph.labels))
        nodes_mask = np.concatenate((first_graph.nodes_mask, second_graph.nodes_mask))

        adjacency_lists = []
        for (edges_id, edges) in enumerate(first_graph.adjacency_lists):
            adjacency_lists.append(np.zeros((len(edges) + len(second_graph.adjacency_lists[edges_id]), 2)))
            for (idx, values) in enumerate(edges):
                adjacency_lists[-1][idx][0] = values[0]
                adjacency_lists[-1][idx][1] = values[1]
            for (idx, values) in enumerate(second_graph.adjacency_lists[edges_id]):
                adjacency_lists[-1][idx + len(edges)][0] = values[0] + old_nodes_length
                adjacency_lists[-1][idx + len(edges)][1] = values[1] + old_nodes_length

        return GraphSample(adjacency_lists, \
            type_to_num_incoming_edges, \
            node_features, \
            types, \
            nodes_mask, \
            labels
            )

    def make_minibatch_iterator(self,
                                data: Iterable[Any],
                                data_fold: DataFold,
                                model_placeholders: Dict[str, tf.Tensor],
                                max_nodes_per_batch: int,
                                ) -> Iterator[MinibatchData]:
        if data_fold in (DataFold.TRAIN, DataFold.VALIDATION):
            np.random.shuffle(data)
        
        combined_graphs = None
        end_val = int(len(data))
        for i in range(end_val):
            if i % self.batch_graph_size == 0:
                combined_graphs = data[i]
            else:
                cur_graph = data[i]
                combined_graphs = self.combine_graphs(combined_graphs, cur_graph)
            
            if (i + 1) % self.batch_graph_size == 0 or i == end_val - 1:
                feed_dict = {
                    model_placeholders['node_names']: combined_graphs.node_features,
                    model_placeholders['node_types']: combined_graphs.types,
                    model_placeholders['type_to_num_incoming_edges']: combined_graphs.type_to_num_incoming_edges,
                    model_placeholders['num_graphs']: self.batch_graph_size if (i != end_val - 1) or (end_val % self.batch_graph_size == 0) \
                        else end_val % self.batch_graph_size,
                    model_placeholders['labels']: combined_graphs.labels,
                    model_placeholders['nodes_mask']: combined_graphs.nodes_mask
                }

                for i in range(self.__num_edge_types):
                    feed_dict[model_placeholders["adjacency_lists"][i]] = combined_graphs.adjacency_lists[i]

                yield MinibatchData(feed_dict=feed_dict,
                    num_graphs = self.batch_graph_size if (i != end_val - 1) or (end_val % self.batch_graph_size == 0) \
                        else end_val % self.batch_graph_size,
                    num_nodes = np.shape(combined_graphs.node_features)[0],
                    num_edges = sum(len(adj_list) for adj_list in combined_graphs.adjacency_lists)
                )


    def early_stopping_metric(self,
                              task_metric_results: List[Dict[str, np.ndarray]],
                              num_graphs: int,
                              ) -> float:
        return np.sum([m['loss'] for m in task_metric_results])

    def pretty_print_epoch_task_metrics(self, task_metric_results: List[Dict[str, np.ndarray]], num_graphs: int) -> str:
        return "Acc: %.2f%%" % (np.sum([m['accuracy'] for m in task_metric_results]) / len(task_metric_results)*100,)
        # return "Acc: %.2f%%" % (task_metric_results[0]['accuracy']*100,)

