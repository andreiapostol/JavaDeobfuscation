from collections import namedtuple
from typing import Any, Dict, Iterator, List, Iterable

import tensorflow as tf
import numpy as np
import pickle
from dpu_utils.utils import RichPath

from .sparse_graph_task import Sparse_Graph_Task, DataFold, MinibatchData
from utils import micro_f1


GraphSample = namedtuple('GraphSample', ['adjacency_lists',
                                         'type_to_node_to_num_incoming_edges',
                                         'node_labels',
                                         ])

ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
ALPHABET_DICT = {char: idx + 2 for (idx, char) in enumerate(ALPHABET)}  # "0" is PAD, "1" is UNK
ALPHABET_DICT["PAD"] = 0
ALPHABET_DICT["UNK"] = 1
class Deobfuscate_Task(Sparse_Graph_Task):
    @classmethod
    def default_params(cls):
        params = super().default_params()
        params.update({
            'add_self_loop_edges': False,
            'tie_fwd_bkwd_edges': False,
            'out_layer_dropout_keep_prob': 1.0,
        })
        return params

    @staticmethod
    def default_data_path() -> str:
        return "data/deobfuscation"

    @staticmethod
    def name() -> str:
        return "DEOBFUSCATION"

    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)

        # Things that will be filled once we load data:
        self.__num_edge_types = 0
        self.__initial_node_feature_size = 0
        self.__num_labels = 0

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
        self._loaded_data[DataFold.TRAIN] = self.__load_data(path, DataFold.TRAIN)
    
    def __load_data(self, data_dir: RichPath, data_fold: DataFold) -> List[GraphSample]:
        if data_fold == DataFold.TRAIN:
            data_name = "train"
        elif data_fold == DataFold.VALIDATION:
            data_name = "valid"
        elif data_fold == DataFold.TEST:
            data_name = "test"
        else:
            raise ValueError("Unknown data fold '%s'" % str(data_fold))
        print(" Loading DEOBFUSCATION %s data from %s." % (data_name, data_dir))
        all_untensorised = data_dir.join("%s.pkl.gz" % data_name).read_by_file_suffix()

        # untensorised_data = 


    def load_eval_data_from_path(self, path: RichPath) -> Iterable[Any]:
        raise NotImplementedError()

    def make_task_input_model(self,
                              placeholders: Dict[str, tf.Tensor],
                              model_ops: Dict[str, tf.Tensor],
                              ) -> None:
        placeholders['initial_node_features'] = \
            tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, self.initial_node_feature_size], name='initial_node_features')
        placeholders['adjacency_lists'] = \
            [tf.compat.v1.placeholder(dtype=tf.int32, shape=[None, 2], name='adjacency_e%s' % e)
                for e in range(self.num_edge_types)]
        placeholders['type_to_num_incoming_edges'] = \
            tf.compat.v1.placeholder(dtype=tf.float32, shape=[self.num_edge_types, None], name='type_to_num_incoming_edges')

        model_ops['initial_node_features'] = placeholders['initial_node_features']
        model_ops['adjacency_lists'] = placeholders['adjacency_lists']
        model_ops['type_to_num_incoming_edges'] = placeholders['type_to_num_incoming_edges']

    def make_task_output_model(self,
                               placeholders: Dict[str, tf.Tensor],
                               model_ops: Dict[str, tf.Tensor],
                               ) -> None:
        raise NotImplementedError()

    def make_minibatch_iterator(self,
                                data: Iterable[Any],
                                data_fold: DataFold,
                                model_placeholders: Dict[str, tf.Tensor],
                                max_nodes_per_batch: int,
                                ) -> Iterator[MinibatchData]:
        raise NotImplementedError()

    def early_stopping_metric(self,
                              task_metric_results: List[Dict[str, np.ndarray]],
                              num_graphs: int,
                              ) -> float:
        raise NotImplementedError()

    def pretty_print_epoch_task_metrics(self,
                                        task_metric_results: List[Dict[str, np.ndarray]],
                                        num_graphs: int,
                                        ) -> str:
        return "SMTH"

