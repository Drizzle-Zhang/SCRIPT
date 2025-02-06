import torch
import pickle
import dgl
import numpy as np
import os
import random


class PretrainLoader:
    def __init__(self, root_path, edge_file_name, peak_file_name, node_folder_name, node_shard=True):
        # self.config_path = config_path
        # if self.config_path:
        #     os.system(f"cp {self.config_path} {root_path}")

        self.edge_file_path = os.path.join(root_path, edge_file_name)
        if self.edge_file_path.endswith('.pt'):
            self.edge_index_cre = torch.load(str(self.edge_file_path))
        elif self.edge_file_path.endswith('.pkl'):
            with open(self.edge_file_path, 'rb') as f:
                self.edge_index_cre = pickle.load(f)
        else:
            raise NotImplementedError(f"file path {self.edge_file_path} is not supported")

        self.peak_file_path = os.path.join(root_path, peak_file_name)
        if self.peak_file_path.endswith('.pt'):
            self.array_peak = torch.load(str(self.peak_file_path))
        elif self.peak_file_path.endswith('.pkl'):
            with open(self.peak_file_path, 'rb') as f:
                self.array_peak = pickle.load(f)
        else:
            raise NotImplementedError(f"file path {self.peak_file_path} is not supported")

        if node_shard:
            self.node_folder_name = os.path.join(root_path, node_folder_name)
            self.node_folder_file = os.listdir(str(self.node_folder_name))
        else:
            self.node_folder_name = root_path
            self.node_folder_file = [node_folder_name]

        self.number_gene, self.number_node = self._get_graph_args()
        self.graph = self.build_graph()

    def _get_graph_args(self):
        mask_numpy = np.array([0 if peak[:3] == 'chr' else 1 for peak in self.array_peak])
        number_gene = int(np.sum(mask_numpy))
        number_node = self.array_peak.shape[0]
        return number_gene, number_node

    def build_graph(self):
        graph = dgl.graph((self.edge_index_cre[0], self.edge_index_cre[1]))
        graph = dgl.add_self_loop(graph)
        return graph
