# _*_ coding: utf-8 _*_
# @author: Drizzle_Zhang
# @file: model.py
# @time: 2023/2/1 14:30


from time import time
import os
import subprocess
from typing import Optional, Tuple, Union
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, device
from torch.nn import Module
from torch.optim import Optimizer
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn.norm import LayerNorm
from torch_geometric.nn.conv import GraphConv, MessagePassing
# from torch_geometric.nn import GraphConv, BatchNorm, LayerNorm, TransformerConv, GATv2Conv, ChebConv
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
import scanpy as sc
import numpy as np
import pandas as pd
import anndata as ad
import captum.attr as attr
from audtorch.metrics.functional import pearsonr
from torch_sparse import SparseTensor, set_diag, matmul
from scipy import stats
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans
from tqdm import tqdm


class GATConvW(MessagePassing):
    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]],
            out_channels: int,
            heads: int = 1,
            concat: bool = True,
            negative_slope: float = 0.2,
            dropout: float = 0.0,
            add_self_loops: bool = True,
            edge_dim: Optional[int] = None,
            fill_value: Union[float, Tensor, str] = 'mean',
            bias: bool = True,
            share_weights: bool = False,
            **kwargs,
    ):
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.share_weights = share_weights

        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * out_channels, bias=bias,
                                weight_initializer='glorot')
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(in_channels, heads * out_channels,
                                    bias=bias, weight_initializer='glorot')
        else:
            self.lin_l = Linear(in_channels[0], heads * out_channels,
                                bias=bias, weight_initializer='glorot')
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(in_channels[1], heads * out_channels,
                                    bias=bias, weight_initializer='glorot')

        self.att = nn.Parameter(torch.Tensor(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
                                   weight_initializer='glorot')
        else:
            self.lin_edge = None

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        _alpha: OptTensor
        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, edge_weight: OptTensor = None,
                return_attention_weights: bool = None):
        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2
            x_l = self.lin_l(x).view(-1, H, C)
            if self.share_weights:
                x_r = x_l
            else:
                x_r = self.lin_r(x).view(-1, H, C)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2
            x_l = self.lin_l(x_l).view(-1, H, C)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)

        assert x_l is not None
        assert x_r is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=(x_l, x_r),
                             edge_attr=edge_attr, edge_weight=edge_weight, size=None)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: Tensor, x_i: Tensor, edge_attr: OptTensor, edge_weight: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        x = x_i + x_j

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            x += edge_attr

        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        x_j = x_j * alpha.unsqueeze(-1)

        if edge_weight is None:
            return x_j
        else:
            edge_weight = edge_weight.view(-1, 1, 1)
            return edge_weight * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class SCReGAT(torch.nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, num_head: int, num_gene: int,
                 num_celltype: int, num_nodes: int):
        super(SCReGAT, self).__init__()
        torch.manual_seed(12345)
        self.num_nodes = num_nodes
        self.num_gene = num_gene

        self.lin1_x = nn.Linear(input_channels, hidden_channels)
        self.lin1_edge = nn.Linear(input_channels, hidden_channels)
        self.conv1 = GATConvW(hidden_channels, hidden_channels, heads=num_head, dropout=0.5,
                              edge_dim=hidden_channels, add_self_loops=False)
        self.ln_1 = LayerNorm(self.num_nodes)
        self.lin2 = nn.Linear(1, hidden_channels)

        self.conv2 = GATConvW(hidden_channels, hidden_channels,
                              heads=1, dropout=0.5, add_self_loops=False)
        self.ln_2 = LayerNorm(num_gene)

        self.lin3 = nn.Linear(num_gene, num_gene)
        self.lin4 = nn.Linear(num_gene, num_celltype)

    def forward(self, x: Tensor, edge_index: Tensor, edge_tf: Tensor, batch: Tensor,
                edge_weight: Optional[Tensor] = None, edge_weight_tf: Optional[Tensor] = None):
        batchsize = len(torch.unique(batch))
        x_edge = x[edge_index[0, :], :] * x[edge_index[1, :], :]
        x = self.lin1_x(x).sigmoid()
        x_edge = self.lin1_edge(x_edge).sigmoid()
        x, atten_w = self.conv1(x, edge_index, x_edge, edge_weight, return_attention_weights=True)
        x = x.view(batchsize, self.num_nodes, -1)
        x = torch.mean(x, dim=-1, keepdim=False)
        # x_1 = self.ln_1(x)
        x_1 = x
        x = x_1.unsqueeze(-1).view(batchsize*self.num_nodes, -1)
        x = self.lin2(x).sigmoid()

        x, atten_w2 = self.conv2(x, edge_tf, edge_attr=None, edge_weight=edge_weight_tf,
                                 return_attention_weights=True)
        x = x.view(batchsize, self.num_nodes, -1)
        x = torch.mean(x, dim=-1, keepdim=False)
        x_2 = x_1 + x

        x_gene = x_2[:, :self.num_gene]
        x_gene = self.ln_2(x_gene)
        x_label = self.lin3(x_gene).relu()
        x_label = F.dropout(x_label, p=0.5, training=self.training)
        x_label = self.lin4(x_label)

        return F.log_softmax(x_label, dim=1), F.log_softmax(x_gene, dim=1), atten_w, atten_w2


class GCN(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, num_gene, num_nodes):
        super(GCN, self).__init__()
        self.num_nodes = num_nodes
        self.num_gene = num_gene

        torch.manual_seed(12345)
        self.conv1 = GraphConv(input_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)

        self.ln = LayerNorm(num_gene)

    def forward(self, x, edge_index, batch, edge_weight=None):
        batchsize = len(torch.unique(batch))
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = self.conv2(x, edge_index, edge_weight)

        x = torch.mean(x, dim=1, keepdim=True)
        x = x.view(batchsize, x.shape[0]//batchsize)

        x_1 = x[:, :self.num_gene]
        x_1 = self.ln(x_1)
        return F.log_softmax(x_1, dim=1)


class GAT(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, num_head, num_gene, num_nodes):
        super(GAT, self).__init__()
        torch.manual_seed(12345)
        self.num_nodes = num_nodes
        self.num_gene = num_gene

        self.lin1_x = nn.Linear(input_channels, hidden_channels)
        self.lin1_edge = nn.Linear(input_channels, hidden_channels)
        self.conv1 = GATConvW(hidden_channels, hidden_channels, heads=num_head, dropout=0.5,
                              edge_dim=hidden_channels, add_self_loops=False)

        self.ln = LayerNorm(num_gene)

    def forward(self, x, edge_index, batch, edge_weight=None):
        if len(x.shape) == 1:
            x = x.view(x.shape[0], 1)
        batchsize = len(torch.unique(batch))
        x_edge = x[edge_index[0, :], :] * x[edge_index[1, :], :]
        x = self.lin1_x(x).relu()
        x_edge = self.lin1_edge(x_edge).relu()
        x, atten_w = self.conv1(x, edge_index, x_edge, edge_weight, return_attention_weights=True)

        x = x.view(batchsize, self.num_nodes, -1)
        # x = torch.squeeze(self.proj(x))
        x = torch.mean(x, dim=-1, keepdim=False)

        x_1 = x[:, -self.num_gene:]
        x_1 = self.ln(x_1)
        return F.log_softmax(x_1, dim=1)


class PretrainGAT(torch.nn.Module):
    def __init__(self, input_channels, emb_channels, num_head, num_gene, num_nodes):
        super(PretrainGAT, self).__init__()
        torch.manual_seed(12345)
        self.num_nodes = num_nodes
        self.num_gene = num_gene

        self.hidden_channels = emb_channels
        self.lin1_x = nn.Linear(input_channels, self.hidden_channels)
        self.lin1_emb = nn.Linear(emb_channels, self.hidden_channels)
        self.lin1_edge = nn.Linear(input_channels, self.hidden_channels)
        self.lin1_concat = nn.Linear(self.hidden_channels*2, self.hidden_channels*2)
        self.conv1 = GATConvW(self.hidden_channels*2, self.hidden_channels*2, heads=num_head,
                              dropout=0.5, edge_dim=self.hidden_channels, add_self_loops=False)

        self.ln = LayerNorm(num_gene)

        self.lin2 = nn.Linear(self.hidden_channels*2*num_head, 1)
        self.lin3 = nn.Linear(num_gene, num_gene)

    def forward(self, x, edge_index, batch, emb=None, edge_weight=None):
        if len(x.shape) == 1:
            x = x.view(x.shape[0], 1)
        batchsize = len(torch.unique(batch))
        x_edge = x[edge_index[0, :], :] * x[edge_index[1, :], :]
        x = self.lin1_x(x)
        # x_emb = self.lin1_emb(emb)
        x_emb = emb
        x_concat = torch.concat([x, x_emb], dim=1)
        x_concat = self.lin1_concat(x_concat).relu()
        x_edge = self.lin1_edge(x_edge).relu()
        x, atten_w = self.conv1(
            x_concat, edge_index, x_edge, edge_weight, return_attention_weights=True)

        x = x.view(batchsize, self.num_nodes, -1)
        x = self.lin2(x).squeeze(-1)
        # x = torch.mean(x, dim=-1, keepdim=False)

        x_1 = x[:, -self.num_gene:]
        x_1 = self.ln(x_1)
        return F.log_softmax(x_1, dim=1)


class GATLabelRelu(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, num_head, num_gene, num_celltype, num_nodes):
        super(GATLabelRelu, self).__init__()
        torch.manual_seed(12345)
        self.num_nodes = num_nodes
        self.num_gene = num_gene

        self.lin1_x = nn.Linear(input_channels, hidden_channels)
        self.lin1_edge = nn.Linear(input_channels, hidden_channels)
        self.conv1 = GATConvW(hidden_channels, hidden_channels, heads=num_head, dropout=0.5,
                              edge_dim=hidden_channels, add_self_loops=False)

        self.ln = LayerNorm(num_gene)

        # self.lin2 = nn.Linear(hidden_channels*num_head, 1)
        self.lin3 = nn.Linear(num_gene, num_gene)
        self.lin4 = nn.Linear(num_gene, num_celltype)

    def forward(self, x, edge_index, batch, edge_weight=None):
        if len(x.shape) == 1:
            x = x.view(x.shape[0], 1)
        batchsize = len(torch.unique(batch))
        x_edge = x[edge_index[0, :], :] * x[edge_index[1, :], :]
        x = self.lin1_x(x).relu()
        x_edge = self.lin1_edge(x_edge).relu()
        x, atten_w = self.conv1(x, edge_index, x_edge, edge_weight, return_attention_weights=True)

        x = x.view(batchsize, self.num_nodes, -1)
        # x = self.lin2(x).squeeze(-1)
        x = torch.mean(x, dim=-1, keepdim=False)

        x_1 = x[:, -self.num_gene:]
        x_1 = self.ln(x_1)
        x_2 = self.lin3(x_1).relu()
        x_2 = F.dropout(x_2, p=0.5, training=self.training)
        x_2 = self.lin4(x_2)
        return F.log_softmax(x_2, dim=1), F.log_softmax(x_1, dim=1)


class GATLabel(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, num_head, num_gene, num_celltype, num_nodes):
        super(GATLabel, self).__init__()
        torch.manual_seed(12345)
        self.num_nodes = num_nodes
        self.num_gene = num_gene

        self.lin1_x = nn.Linear(input_channels, hidden_channels)
        self.lin1_edge = nn.Linear(input_channels, hidden_channels)
        self.conv1 = GATConvW(hidden_channels, hidden_channels, heads=num_head, dropout=0.5,
                              edge_dim=hidden_channels, add_self_loops=False)

        self.ln = LayerNorm(num_gene)

        # self.lin2 = nn.Linear(hidden_channels*num_head, 1)
        self.lin3 = nn.Linear(num_gene, num_gene)
        self.lin4 = nn.Linear(num_gene, num_celltype)

    def forward(self, x, edge_index, batch, edge_weight=None):
        if len(x.shape) == 1:
            x = x.view(x.shape[0], 1)
        batchsize = len(torch.unique(batch))
        x_edge = x[edge_index[0, :], :] * x[edge_index[1, :], :]
        x = self.lin1_x(x).sigmoid()
        x_edge = self.lin1_edge(x_edge).sigmoid()
        x, atten_w = self.conv1(x, edge_index, x_edge, edge_weight, return_attention_weights=True)

        x = x.view(batchsize, self.num_nodes, -1)
        # x = self.lin2(x).squeeze(-1)
        x = torch.mean(x, dim=-1, keepdim=False)

        x_1 = x[:, -self.num_gene:]
        x_1 = self.ln(x_1)
        x_2 = self.lin3(x_1).sigmoid()
        x_2 = F.dropout(x_2, p=0.5, training=self.training)
        x_2 = self.lin4(x_2)
        return F.log_softmax(x_2, dim=1), F.log_softmax(x_1, dim=1)


class GATFinetune(torch.nn.Module):
    def __init__(self, input_channels, emb_channels, num_head,
                 num_gene, num_celltype, num_nodes, pretrain):
        super(GATFinetune, self).__init__()
        torch.manual_seed(12345)
        self.num_nodes = num_nodes
        self.num_gene = num_gene
        self.pretrain = pretrain

        self.hidden_channels = emb_channels
        self.lin1_x = nn.Linear(input_channels, self.hidden_channels)
        self.lin1_edge = nn.Linear(input_channels, self.hidden_channels)
        if self.pretrain:
            self.lin1_concat = nn.Linear(self.hidden_channels*2, self.hidden_channels*2)
            self.conv1 = GATConvW(self.hidden_channels*2, self.hidden_channels*2, heads=num_head,
                                  dropout=0.5, edge_dim=self.hidden_channels, add_self_loops=False)
            self.lin2 = nn.Linear(self.hidden_channels*2*num_head, 1)
        else:
            self.conv1 = GATConvW(self.hidden_channels, self.hidden_channels, heads=num_head,
                                  dropout=0.5, edge_dim=self.hidden_channels, add_self_loops=False)
            self.lin2 = nn.Linear(self.hidden_channels*num_head, 1)

        self.ln = LayerNorm(num_gene)

        self.lin3 = nn.Linear(num_gene, num_gene)
        self.lin4 = nn.Linear(num_gene, num_celltype)

    def forward(self, x, edge_index, batch, emb=None, edge_weight=None):
        if len(x.shape) == 1:
            x = x.view(x.shape[0], 1)
        batchsize = len(torch.unique(batch))
        x_edge = x[edge_index[0, :], :] * x[edge_index[1, :], :]
        x = self.lin1_x(x)
        # x_emb = self.lin1_emb(emb)
        if self.pretrain:
            x_emb = emb
            x_concat = torch.concat([x, x_emb], dim=1)
            x_concat = self.lin1_concat(x_concat).relu()
        else:
            x_concat = x.relu()
        x_edge = self.lin1_edge(x_edge).relu()
        x, atten_w = self.conv1(
            x_concat, edge_index, x_edge, edge_weight, return_attention_weights=True)

        x = x.view(batchsize, self.num_nodes, -1)
        x = self.lin2(x).squeeze(-1)
        # x = torch.mean(x, dim=-1, keepdim=False)

        x_1 = x[:, -self.num_gene:]
        x_1 = self.ln(x_1)
        x_2 = self.lin3(x_1).relu()
        x_2 = F.dropout(x_2, p=0.5, training=self.training)
        x_2 = self.lin4(x_2)
        return F.log_softmax(x_2, dim=1), F.log_softmax(x_1, dim=1)


class GATConcat(torch.nn.Module):
    def __init__(self, input_channels, emb_channels, num_head,
                 num_gene, num_celltype, num_nodes, pretrain):
        super(GATConcat, self).__init__()
        torch.manual_seed(12345)
        self.num_nodes = num_nodes
        self.num_gene = num_gene
        self.pretrain = pretrain

        self.hidden_channels = emb_channels
        self.lin1_x = nn.Linear(input_channels, self.hidden_channels)
        self.lin1_edge = nn.Linear(input_channels, self.hidden_channels)
        if self.pretrain:
            self.lin1_concat = nn.Linear(self.hidden_channels*2, self.hidden_channels*2)
            self.conv1 = GATConvW(self.hidden_channels*2, self.hidden_channels*2, heads=num_head,
                                  dropout=0.5, edge_dim=self.hidden_channels, add_self_loops=False)
            self.lin2 = nn.Linear(self.hidden_channels*2*num_head, 1)
        else:
            self.conv1 = GATConvW(self.hidden_channels, self.hidden_channels, heads=num_head,
                                  dropout=0.5, edge_dim=self.hidden_channels, add_self_loops=False)
            self.lin2 = nn.Linear(self.hidden_channels*num_head, 1)

        self.ln = LayerNorm(num_gene)

        self.lin3 = nn.Linear(num_gene, num_gene)
        self.lin4 = nn.Linear(num_gene, num_celltype)

    def forward(self, x, edge_index, batch, emb=None, edge_weight=None):
        if len(x.shape) == 1:
            x = x.view(x.shape[0], 1)
        batchsize = len(torch.unique(batch))
        x_edge = x[edge_index[0, :], :] * x[edge_index[1, :], :]
        x = self.lin1_x(x)
        # x_emb = self.lin1_emb(emb)
        if self.pretrain:
            x_emb = emb
            x_concat = torch.concat([x, x_emb], dim=1)
            x_concat = self.lin1_concat(x_concat).relu()
        else:
            x_concat = x.relu()
        x_edge = self.lin1_edge(x_edge).relu()
        x, atten_w = self.conv1(
            x_concat, edge_index, x_edge, edge_weight, return_attention_weights=True)

        x = x.view(batchsize, self.num_nodes, -1)
        x = self.lin2(x).squeeze(-1)
        # x = torch.mean(x, dim=-1, keepdim=False)

        x_1 = x[:, -self.num_gene:]
        x_1 = self.ln(x_1)

        return -1, F.log_softmax(x_1, dim=1)


class GATLabelConcat(torch.nn.Module):
    def __init__(self, input_channels, emb_channels, num_head, num_gene, num_celltype, num_nodes):
        super(GATLabelConcat, self).__init__()
        torch.manual_seed(12345)
        self.num_nodes = num_nodes
        self.num_gene = num_gene

        self.hidden_channels = emb_channels
        self.lin1_x = nn.Linear(input_channels, self.hidden_channels)
        self.lin1_emb = nn.Linear(emb_channels, self.hidden_channels)
        self.lin1_edge = nn.Linear(input_channels, self.hidden_channels)
        self.lin1_concat = nn.Linear(self.hidden_channels*2, self.hidden_channels*2)
        self.conv1 = GATConvW(self.hidden_channels*2, self.hidden_channels*2, heads=num_head,
                              dropout=0.5, edge_dim=self.hidden_channels, add_self_loops=False)

        self.ln = LayerNorm(num_gene)

        self.lin2 = nn.Linear(self.hidden_channels*2*num_head, 1)
        self.lin3 = nn.Linear(num_gene, num_gene)
        self.lin4 = nn.Linear(num_gene, num_celltype)

    def forward(self, x, edge_index, batch, emb=None, edge_weight=None):
        if len(x.shape) == 1:
            x = x.view(x.shape[0], 1)
        batchsize = len(torch.unique(batch))
        x_edge = x[edge_index[0, :], :] * x[edge_index[1, :], :]
        x = self.lin1_x(x)
        # x_emb = self.lin1_emb(emb)
        x_emb = emb
        x_concat = torch.concat([x, x_emb], dim=1)
        x_concat = self.lin1_concat(x_concat).relu()
        x_edge = self.lin1_edge(x_edge).relu()
        x, atten_w = self.conv1(
            x_concat, edge_index, x_edge, edge_weight, return_attention_weights=True)

        x = x.view(batchsize, self.num_nodes, -1)
        x = self.lin2(x).squeeze(-1)
        # x = torch.mean(x, dim=-1, keepdim=False)

        x_1 = x[:, -self.num_gene:]
        x_1 = self.ln(x_1)
        x_2 = self.lin3(x_1).relu()
        x_2 = F.dropout(x_2, p=0.5, training=self.training)
        x_2 = self.lin4(x_2)
        return F.log_softmax(x_2, dim=1), F.log_softmax(x_1, dim=1)


class GATLabelConcatMLP(torch.nn.Module):
    def __init__(self, input_channels, emb_channels, num_head, num_gene, num_celltype, num_nodes):
        super(GATLabelConcatMLP, self).__init__()
        torch.manual_seed(12345)
        self.num_nodes = num_nodes
        self.num_gene = num_gene

        self.hidden_channels = emb_channels
        self.lin1_x = nn.Linear(input_channels, self.hidden_channels)
        self.lin1_emb = nn.Linear(emb_channels, self.hidden_channels)
        self.lin1_edge = nn.Linear(input_channels, self.hidden_channels)
        self.lin1_concat = nn.Linear(self.hidden_channels*2, self.hidden_channels*2)
        self.conv1 = GATConvW(self.hidden_channels*2, self.hidden_channels*2, heads=num_head,
                              dropout=0.5, edge_dim=self.hidden_channels, add_self_loops=False)

        self.ln = LayerNorm(num_gene)

        self.lin2 = nn.Linear(self.hidden_channels*2*num_head, 1)
        self.lin3 = nn.Linear(num_gene, num_gene)
        self.lin4 = nn.Linear(num_gene, num_celltype)

    def forward(self, x, edge_index, batch, emb=None, edge_weight=None):
        if len(x.shape) == 1:
            x = x.view(x.shape[0], 1)
        batchsize = len(torch.unique(batch))
        x_edge = x[edge_index[0, :], :] * x[edge_index[1, :], :]
        x = self.lin1_x(x)
        # x_emb = self.lin1_emb(emb)
        x_emb = emb
        x_concat = torch.concat([x, x_emb], dim=1)
        x_concat = self.lin1_concat(x_concat).relu()
        x_edge = self.lin1_edge(x_edge).relu()
        x, atten_w = self.conv1(
            x_concat, edge_index, x_edge, edge_weight, return_attention_weights=True)

        x = x.view(batchsize, self.num_nodes, -1)
        x = self.lin2(x).squeeze(-1)
        # x = torch.mean(x, dim=-1, keepdim=False)

        x_1 = x[:, -self.num_gene:]
        x_1 = self.ln(x_1)
        x_3 = self.lin3(x_1)
        x_2 = x_3.relu()
        x_2 = F.dropout(x_2, p=0.5, training=self.training)
        x_2 = self.lin4(x_2)
        return F.log_softmax(x_2, dim=1), F.log_softmax(x_3, dim=1)


class GATLabelConcat3(torch.nn.Module):
    def __init__(self, input_channels, emb_channels, num_head, num_gene, num_celltype, num_nodes):
        super(GATLabelConcat3, self).__init__()
        torch.manual_seed(12345)
        self.num_nodes = num_nodes
        self.num_gene = num_gene

        self.hidden_channels = emb_channels
        self.lin1_x = nn.Linear(input_channels, self.hidden_channels)
        self.lin1_emb = nn.Linear(emb_channels, self.hidden_channels)
        self.lin1_edge = nn.Linear(input_channels, self.hidden_channels)
        self.lin1_concat = nn.Linear(self.hidden_channels*3, self.hidden_channels*3)
        self.conv1 = GATConvW(self.hidden_channels*3, self.hidden_channels*3, heads=num_head,
                              dropout=0.5, edge_dim=self.hidden_channels, add_self_loops=False)

        self.ln = LayerNorm(num_gene)

        self.lin2 = nn.Linear(self.hidden_channels*3*num_head, 1)
        self.lin3 = nn.Linear(num_gene, num_gene)
        self.lin4 = nn.Linear(num_gene, num_celltype)

    def forward(self, x, edge_index, batch, emb=None, edge_weight=None):
        if len(x.shape) == 1:
            x = x.view(x.shape[0], 1)
        batchsize = len(torch.unique(batch))
        x_edge = x[edge_index[0, :], :] * x[edge_index[1, :], :]
        x = self.lin1_x(x)
        x_emb = self.lin1_emb(emb)
        # x_emb = emb
        x_concat = torch.concat([x, x_emb, x + x_emb], dim=1)
        x_concat = self.lin1_concat(x_concat).relu()
        x_edge = self.lin1_edge(x_edge).relu()
        x, atten_w = self.conv1(
            x_concat, edge_index, x_edge, edge_weight, return_attention_weights=True)

        x = x.view(batchsize, self.num_nodes, -1)
        x = self.lin2(x).squeeze(-1)
        # x = torch.mean(x, dim=-1, keepdim=False)

        x_1 = x[:, -self.num_gene:]
        x_1 = self.ln(x_1)
        x_2 = self.lin3(x_1).relu()
        x_2 = F.dropout(x_2, p=0.5, training=self.training)
        x_2 = self.lin4(x_2)
        return F.log_softmax(x_2, dim=1), F.log_softmax(x_1, dim=1)


class GATLabelConcat2(torch.nn.Module):
    def __init__(self, input_channels, emb_channels, num_head, num_gene, num_celltype, num_nodes):
        super(GATLabelConcat2, self).__init__()
        torch.manual_seed(12345)
        self.num_nodes = num_nodes
        self.num_gene = num_gene

        self.hidden_channels = emb_channels
        self.lin1_x = nn.Linear(input_channels, self.hidden_channels)
        self.lin1_emb = nn.Linear(emb_channels, self.hidden_channels)
        self.lin1_edge = nn.Linear(input_channels, self.hidden_channels)
        self.conv1 = GATConvW(self.hidden_channels*2, self.hidden_channels*2, heads=num_head,
                              dropout=0.5, edge_dim=self.hidden_channels, add_self_loops=False)

        self.ln = LayerNorm(num_gene)

        self.lin2 = nn.Linear(self.hidden_channels*2*num_head, 1)
        self.lin3 = nn.Linear(num_gene, num_gene)
        self.lin4 = nn.Linear(num_gene, num_celltype)

    def forward(self, x, edge_index, batch, emb=None, edge_weight=None):
        if len(x.shape) == 1:
            x = x.view(x.shape[0], 1)
        batchsize = len(torch.unique(batch))
        x_edge = x[edge_index[0, :], :] * x[edge_index[1, :], :]
        x = self.lin1_x(x)
        x_emb = self.lin1_emb(emb)
        # x_emb = emb
        x_concat = torch.concat([x, x_emb], dim=1).relu()
        x_edge = self.lin1_edge(x_edge).relu()
        x, atten_w = self.conv1(
            x_concat, edge_index, x_edge, edge_weight, return_attention_weights=True)

        x = x.view(batchsize, self.num_nodes, -1)
        x = self.lin2(x).squeeze(-1)
        # x = torch.mean(x, dim=-1, keepdim=False)

        x_1 = x[:, -self.num_gene:]
        x_1 = self.ln(x_1)
        x_2 = self.lin3(x_1).relu()
        x_2 = F.dropout(x_2, p=0.5, training=self.training)
        x_2 = self.lin4(x_2)
        return F.log_softmax(x_2, dim=1), F.log_softmax(x_1, dim=1)


class GATLabelConcatPool(torch.nn.Module):
    def __init__(self, input_channels, emb_channels, num_head, num_gene, num_celltype, num_nodes):
        super(GATLabelConcatPool, self).__init__()
        torch.manual_seed(12345)
        self.num_nodes = num_nodes
        self.num_gene = num_gene

        self.hidden_channels = emb_channels
        self.lin1_x = nn.Linear(input_channels, self.hidden_channels)
        self.lin1_emb = nn.Linear(emb_channels, self.hidden_channels)
        self.lin1_edge = nn.Linear(input_channels, self.hidden_channels)
        self.conv1 = GATConvW(self.hidden_channels*2, self.hidden_channels*2, heads=num_head,
                              dropout=0.5, edge_dim=self.hidden_channels, add_self_loops=False)

        self.ln = LayerNorm(num_gene)

        # self.lin2 = nn.Linear(hidden_channels*num_head, 1)
        self.lin3 = nn.Linear(num_gene, num_gene)
        self.lin4 = nn.Linear(num_gene, num_celltype)

    def forward(self, x, edge_index, batch, emb=None, edge_weight=None):
        if len(x.shape) == 1:
            x = x.view(x.shape[0], 1)
        batchsize = len(torch.unique(batch))
        x_edge = x[edge_index[0, :], :] * x[edge_index[1, :], :]
        x = self.lin1_x(x)
        # x_emb = self.lin1_emb(emb)
        x_emb = emb
        x_concat = torch.concat([x, x_emb], dim=1).relu()
        x_edge = self.lin1_edge(x_edge).relu()
        x, atten_w = self.conv1(
            x_concat, edge_index, x_edge, edge_weight, return_attention_weights=True)

        x = x.view(batchsize, self.num_nodes, -1)
        # x = self.lin2(x).squeeze(-1)
        x = torch.mean(x, dim=-1, keepdim=False)

        x_1 = x[:, -self.num_gene:]
        x_1 = self.ln(x_1)
        x_2 = self.lin3(x_1).relu()
        x_2 = F.dropout(x_2, p=0.5, training=self.training)
        x_2 = self.lin4(x_2)
        return F.log_softmax(x_2, dim=1), F.log_softmax(x_1, dim=1)


class GATLabelConcatSig(torch.nn.Module):
    def __init__(self, input_channels, emb_channels, num_head, num_gene, num_celltype, num_nodes):
        super(GATLabelConcatSig, self).__init__()
        torch.manual_seed(12345)
        self.num_nodes = num_nodes
        self.num_gene = num_gene

        self.hidden_channels = emb_channels
        self.lin1_x = nn.Linear(input_channels, self.hidden_channels)
        self.lin1_emb = nn.Linear(emb_channels, self.hidden_channels)
        self.lin1_edge = nn.Linear(input_channels, self.hidden_channels)
        self.conv1 = GATConvW(self.hidden_channels*2, self.hidden_channels*2, heads=num_head,
                              dropout=0.5, edge_dim=self.hidden_channels, add_self_loops=False)

        self.ln = LayerNorm(num_gene)

        # self.lin2 = nn.Linear(hidden_channels*num_head, 1)
        self.lin3 = nn.Linear(num_gene, num_gene)
        self.lin4 = nn.Linear(num_gene, num_celltype)

    def forward(self, x, edge_index, batch, emb=None, edge_weight=None):
        if len(x.shape) == 1:
            x = x.view(x.shape[0], 1)
        batchsize = len(torch.unique(batch))
        x_edge = x[edge_index[0, :], :] * x[edge_index[1, :], :]
        x = self.lin1_x(x)
        # x_emb = self.lin1_emb(emb)
        x_emb = emb
        x_concat = torch.concat([x, x_emb], dim=1).sigmoid()
        x_edge = self.lin1_edge(x_edge).sigmoid()
        x, atten_w = self.conv1(
            x_concat, edge_index, x_edge, edge_weight, return_attention_weights=True)

        x = x.view(batchsize, self.num_nodes, -1)
        # x = self.lin2(x).squeeze(-1)
        x = torch.mean(x, dim=-1, keepdim=False)

        x_1 = x[:, -self.num_gene:]
        x_1 = self.ln(x_1)
        x_2 = self.lin3(x_1).sigmoid()
        x_2 = F.dropout(x_2, p=0.5, training=self.training)
        x_2 = self.lin4(x_2)
        return F.log_softmax(x_2, dim=1), F.log_softmax(x_1, dim=1)


class MyLossPairwise(nn.Module):
    def __init__(self, lambda_1: float, lambda_2: float, weight_label: Tensor):
        super(MyLossPairwise, self).__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.weight_label = weight_label
        self.loss_label = torch.nn.CrossEntropyLoss(weight=weight_label)
        self.loss_exp = torch.nn.KLDivLoss(log_target=False, reduction='batchmean')
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)

    def forward(self, out_label: Tensor, out_exp: Tensor, true_label: Tensor, true_exp: Tensor):
        label_loss = self.loss_label(out_label, true_label)
        exp_loss = self.loss_exp(out_exp, true_exp)
        cos_exp = self.cos(out_exp[:, None, :], out_exp[None, :, :])
        label_mat = torch.zeros(cos_exp.shape)
        label_mat[(true_label[:, None] - true_label[None, :]) == 0] = 1
        label_mat = label_mat.to(true_label.device)
        mask_pairwise = torch.triu(torch.ones(cos_exp.shape), diagonal=1).to(true_label.device)
        loss_pairwise = \
            -label_mat * torch.log(torch.clip(cos_exp, 1e-10, 1.0)) - (1 - label_mat) * torch.log(torch.clip(1 - cos_exp, 1e-10, 1.0))
        loss_pairwise = torch.mean(mask_pairwise * loss_pairwise)
        total_loss = label_loss + self.lambda_1 * exp_loss + self.lambda_2 * loss_pairwise
        return total_loss, label_loss, exp_loss, loss_pairwise


class MyLoss(nn.Module):
    def __init__(self, lambda_1: float, weight_label: Tensor):
        super(MyLoss, self).__init__()
        self.lambda_1 = lambda_1
        self.weight_label = weight_label
        self.loss_label = torch.nn.CrossEntropyLoss(weight=weight_label)
        self.loss_exp = torch.nn.KLDivLoss(log_target=False, reduction='batchmean')

    def forward(self, out_label: Tensor, out_exp: Tensor, true_label: Tensor, true_exp: Tensor):
        label_loss = self.loss_label(out_label, true_label)
        exp_loss = self.loss_exp(out_exp, true_exp)
        total_loss = label_loss + self.lambda_1 * exp_loss
        return total_loss, label_loss, exp_loss


class MyLossExp(nn.Module):
    def __init__(self):
        super(MyLossExp, self).__init__()
        self.loss_exp = torch.nn.KLDivLoss(log_target=False, reduction='batchmean')

    def forward(self, out_exp, true_exp):
        exp_loss = self.loss_exp(out_exp, true_exp)
        return exp_loss


class MyLossExpMse(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_exp = torch.nn.MSELoss(reduction='mean')

    def forward(self, out_exp, true_exp):
        exp_loss = self.loss_exp(out_exp, true_exp)
        return exp_loss


def train(model: Module, criterion: Module, optimizer: Optimizer, 
          use_device: device, loader: DataLoader):
    model.train()
    list_loss1 = []
    list_loss2 = []
    list_loss = []
    for data in loader:  # Iterate in batches over the training dataset.
        data = data.to(use_device)
        # edge_tf_input = data.edge_tf.T
        batch_tf = []
        batchsize = len(torch.unique(data.batch))
        num_peaks = data.x.shape[0] // batchsize
        print(data.x.shape, data.edge_index_2.shape, data.batch.shape)
        # num_tfpair = edge_tf_input.shape[1] // batchsize
        # for idx_tf in range(batchsize):
        #     batch_tf.extend([idx_tf * num_peaks for _ in range(num_tfpair)])
        # tensor_batch_tf = torch.tensor([batch_tf, batch_tf]).to(use_device)
        # edge_tf_input = edge_tf_input + tensor_batch_tf
        out1, out2, out_atten, out_atten2  = \
            model(data.x, data.edge_index_2, data.batch)
        loss, loss1, loss2 = criterion(out1, out2, data.y, data.y_exp.view(out2.shape))
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
        list_loss.append(loss.cpu().detach().numpy())
        list_loss1.append(loss1.cpu().detach().numpy())
        list_loss2.append(loss2.cpu().detach().numpy())
    loss_cat = np.array(list_loss)
    loss1_cat = np.array(list_loss1)
    loss2_cat = np.array(list_loss2)
    return np.mean(loss_cat), np.mean(loss1_cat), np.mean(loss2_cat)


def train_exp(model, criterion, optimizer, device, loader):
    model.train()
    list_loss = []
    for data in tqdm(loader):  # Iterate in batches over the training dataset.
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y_exp.view(out.shape))
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
        list_loss.append(loss.cpu().detach().numpy())
    loss_cat = np.array(list_loss)
    return np.mean(loss_cat)


def test_exp(model, device, loader):
    with torch.no_grad():
        list_corr = []
        for data in tqdm(loader):  # Iterate in batches over the training/test dataset.c
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            list_corr.append(pearsonr(torch.exp(out.cpu()), data.y_exp.view(out.shape).cpu()))
        corr_cat = torch.cat(list_corr, dim=0)

    return torch.median(corr_cat)


def train_gat(path_data_root: str, dataset_atac, dir_model: str,
              use_device: device, hidwidth: int = 16, numhead: int = 8,
              learning_rate: float = 1e-3, num_epoch: int = 20,
              split_prop: float = 0.6, batch_size: int = 16, load_from_pretrained=False):
    # read data
    # file_atac_test = os.path.join(path_data_root, 'dataset_atac.pkl')
    # with open(file_atac_test, 'rb') as r_pkl:
    #     dataset_atac = pickle.loads(r_pkl.read())
    dataset_atac.generate_data_list(rna_exp=True)
    list_graph_cortex = dataset_atac.list_graph
    # path_graph_input = os.path.join(path_data_root, 'input_graph')
    # dataset_atac_graph = ATACGraphDataset(path_graph_input)

    torch.manual_seed(12345)
    random.shuffle(list_graph_cortex)
    # dataset = dataset_atac_graph.shuffle()
    # split_prop = 0.8
    num_split = int(len(list_graph_cortex) * split_prop)
    train_dataset = list_graph_cortex[:num_split]
    test_dataset = list_graph_cortex[num_split:]

    # batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # use_device = torch.use_device("cuda:0" if torch.cuda.is_available() else "cpu")
    peaks = dataset_atac.array_peak
    mask_numpy = np.array([0 if peak[:3] == 'chr' else 1 for peak in peaks])
    number_gene = int(np.sum(mask_numpy))

    list_weights = []
    for i in range(len(dataset_atac.array_celltype)):
        sub_dataset = [data for data in train_dataset if data.y == i]
        sub_len = len(sub_dataset)
        sub_weight = len(train_dataset)/sub_len
        list_weights.append(sub_weight)
    criterion = MyLossExp(1, torch.tensor(list_weights).to(use_device))

    # hidwidth, numhead = 16, 8
    num_node_features = 1
    num_nodes = peaks.shape[0]
    model = GAT(input_channels=num_node_features,
                hidden_channels=hidwidth, num_head=numhead,
                num_gene=number_gene,
                num_nodes=num_nodes).to(use_device)

    # train scReGAT
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    list_loss = []
    list_train_corr = []
    list_test_corr = []
    for epoch in range(num_epoch):
        loss_t = train_exp(model, criterion, optimizer, use_device, train_loader)
        train_corr = test_exp(model, use_device, train_loader)
        test_corr = test_exp(model, use_device, test_loader)
        list_loss.append(loss_t)
        list_train_corr.append(train_corr)
        list_test_corr.append(test_corr)
        print(f'Epoch: {epoch:03d}, Total loss: {loss_t:.4f}, \n'
              f'Train Corr: {train_corr:.4f}, Test Corr: {test_corr:.4f}')

    # dir_model = 'scReGAT_exp'
    path_model = os.path.join(path_data_root, dir_model)
    if not os.path.exists(path_model):
        os.mkdir(path_model)

    # save results
    df_res = pd.DataFrame({"loss": list_loss,
                           'train_corr': list_train_corr, 'test_corr': list_test_corr})
    file_res = \
        os.path.join(path_model,
                     f'res_batch_size_{batch_size}_hidwidth_{hidwidth}_numhead_{numhead}_'
                     f'lr_{learning_rate}_numepoch_{num_epoch}_split_{split_prop}.txt')
    df_res.to_csv(file_res, sep='\t')

    # save scReGAT
    file_atac_model = \
        os.path.join(path_model,
                     f'Model_batch_size_{batch_size}_hidwidth_{hidwidth}_numhead_{numhead}_'
                     f'lr_{learning_rate}_numepoch_{num_epoch}_split_{split_prop}.pt')
    torch.save(model, file_atac_model)

    return model, test_dataset


def test(model: Module, use_device: device, loader: DataLoader):
    with torch.no_grad():
        correct = 0
        list_corr = []
        for data in loader:  # Iterate in batches over the training/test dataset.c
            data = data.to(use_device)
            # edge_tf_input = data.edge_tf.T
            batchsize = len(torch.unique(data.batch))
            num_peaks = data.x.shape[0] // batchsize
            # batch_tf = []
            # num_tfpair = edge_tf_input.shape[1] // batchsize
            # for idx_tf in range(batchsize):
            #     batch_tf.extend([idx_tf * num_peaks for _ in range(num_tfpair)])
            # tensor_batch_tf = torch.tensor([batch_tf, batch_tf]).to(use_device)
            # edge_tf_input = edge_tf_input + tensor_batch_tf
            out1, out2, out_atten, out_atten2 = \
                model(data.x, data.edge_index_2, data.batch)
            pred = out1.argmax(dim=1)  # Use the class with highest probability.
            correct += int((pred == data.y).sum())
            list_corr.append(pearsonr(
                torch.exp(out2.cpu()), torch.exp(data.y_exp.view(out2.shape).cpu())))
        total_acc = correct / len(loader.dataset)
        corr_cat = torch.cat(list_corr, dim=0)

    return total_acc, torch.median(corr_cat)


def train_scregat(path_data_root: str, dataset_atac, dir_model: str,
                  use_device: device, hidwidth: int = 16, numhead: int = 8,
                  learning_rate: float = 1e-3, num_epoch: int = 20,
                  split_prop: float = 0.6, batch_size: int = 16):
    # read data
    # file_atac_test = os.path.join(path_data_root, 'dataset_atac.pkl')
    # with open(file_atac_test, 'rb') as r_pkl:
    #     dataset_atac = pickle.loads(r_pkl.read())
    dataset_atac.generate_data_list(rna_exp=True)
    list_graph_cortex = dataset_atac.list_graph
    # path_graph_input = os.path.join(path_data_root, 'input_graph')
    # dataset_atac_graph = ATACGraphDataset(path_graph_input)

    torch.manual_seed(12345)
    random.shuffle(list_graph_cortex)
    # dataset = dataset_atac_graph.shuffle()
    # split_prop = 0.8
    num_split = int(len(list_graph_cortex) * split_prop)
    train_dataset = list_graph_cortex[:num_split]
    test_dataset = list_graph_cortex[num_split:]

    # batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # use_device = torch.use_device("cuda:0" if torch.cuda.is_available() else "cpu")
    peaks = dataset_atac.array_peak
    mask_numpy = np.array([0 if peak[:3] == 'chr' else 1 for peak in peaks])
    number_gene = int(np.sum(mask_numpy))

    list_weights = []
    for i in range(len(dataset_atac.array_celltype)):
        sub_dataset = [data for data in train_dataset if data.y == i]
        sub_len = len(sub_dataset)
        sub_weight = len(train_dataset)/sub_len
        list_weights.append(sub_weight)
    criterion = MyLossExp(1, torch.tensor(list_weights).to(use_device))

    # hidwidth, numhead = 16, 8
    # model = SCReGAT(input_channels=dataset.num_node_features,
    #                 hidden_channels=hidwidth, num_head=numhead,
    #                 num_gene=number_gene, num_celltype=dataset.num_classes,
    #                 num_nodes=dataset_atac_graph[0].num_nodes).to(use_device)
    num_node_features = 1
    num_nodes = peaks.shape[0]
    model = GAT(input_channels=num_node_features,
                    hidden_channels=hidwidth, num_head=numhead,
                    num_gene=number_gene,
                    num_nodes=num_nodes).to(use_device)

    # train scReGAT
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    list_loss = []
    list_train_acc = []
    list_train_corr = []
    list_test_acc = []
    list_test_corr = []
    for epoch in range(num_epoch):
        loss_t, loss_1, loss_2 = train(model, criterion, optimizer, use_device, train_loader)
        train_acc, train_corr = test(model, use_device, train_loader)
        test_acc, test_corr = test(model, use_device, test_loader)
        list_loss.append(loss_t)
        list_train_acc.append(train_acc)
        list_train_corr.append(train_corr)
        list_test_acc.append(test_acc)
        list_test_corr.append(test_corr)
        print(f'Epoch: {epoch:03d}, Total loss: {loss_t:.4f}, Label loss: {loss_1:.4f}, '
              f'Expression loss: {loss_2:.4f} \n'
              f'Train Acc: {train_acc:.4f}, Train Corr: {train_corr:.4f}, \n'
              f'Test Acc: {test_acc:.4f}, Test Corr: {test_corr:.4f}')

    # dir_model = 'scReGAT'
    path_model = os.path.join(path_data_root, dir_model)
    if not os.path.exists(path_model):
        os.mkdir(path_model)

    # save results
    df_res = pd.DataFrame({"loss": list_loss,
                           'train_acc': list_train_acc, 'test_acc': list_test_acc,
                           'train_corr': list_train_corr, 'test_corr': list_test_corr})
    file_res = \
        os.path.join(path_model,
                     f'res_batch_size_{batch_size}_hidwidth_{hidwidth}_numhead_{numhead}_'
                     f'lr_{learning_rate}_numepoch_{num_epoch}_split_{split_prop}.txt')
    df_res.to_csv(file_res, sep='\t')

    # save scReGAT
    file_atac_model = \
        os.path.join(path_model,
                     f'Model_batch_size_{batch_size}_hidwidth_{hidwidth}_numhead_{numhead}_'
                     f'lr_{learning_rate}_numepoch_{num_epoch}_split_{split_prop}.pt')
    torch.save(model, file_atac_model)

    return model, test_dataset


def predict(model: Module, use_device: device, loader: DataLoader):
    with torch.no_grad():
        list_exp = []
        list_y = []
        list_y_exp = []
        list_cell = []
        list_pred = []
        for data in loader:
            data = data.to(use_device)
            out1, out2 = \
                model(data.x, data.edge_index, data.batch)
            pred = out1.argmax(dim=1)
            list_exp.append(out2.cpu().detach().numpy())
            list_y.append(data.y.cpu().detach().numpy())
            list_y_exp.append(data.y_exp.view(out2.shape).cpu().detach().numpy())
            list_cell.extend(data.cell)
            list_pred.append(pred.cpu().detach().numpy())

    return np.concatenate(list_pred), np.concatenate(list_exp), \
           np.concatenate(list_y), np.concatenate(list_y_exp), list_cell


def calculate_ari(mat_in, true_label):
    num_label = len(np.unique(true_label))
    kmeans = KMeans(n_clusters=num_label).fit(mat_in)
    cluster_label = kmeans.labels_
    ari = adjusted_rand_score(cluster_label, true_label)
    ari_norm = (ari + 1) / 2

    return ari_norm


def evaluate(dataset_atac, model, test_loader, use_device, path_out=None):
    peaks = dataset_atac.array_peak
    mask_numpy = np.array([0 if peak[:3] == 'chr' else 1 for peak in peaks])
    number_gene = np.sum(mask_numpy)

    pred_test, pred_exp, true_label, true_exp, test_cell = predict(model, use_device, test_loader)
    true_label = dataset_atac.adata.obs.loc[test_cell, 'celltype_rna'].tolist()
    # true exp
    df_true_cell = pd.DataFrame(
        true_exp,
        index=pd.MultiIndex.from_arrays([test_cell, true_label],
                                        names=['index', 'celltype']),
        columns=peaks[:number_gene])
    df_true_cell = df_true_cell.groupby('celltype').apply(lambda x: x.mean())
    # df_true_cell = np.log1p(df_true_cell*1e5)
    # df_true_cell.index = dataset_atac.array_celltype
    # pred exp
    df_pred_exp = pd.DataFrame(pred_exp,
                               index=pd.MultiIndex.from_arrays([test_cell, true_label],
                                                               names=['index', 'celltype']),
                               columns=peaks[:number_gene])
    df_pred_exp = np.exp(df_pred_exp)
    df_pred_cell = df_pred_exp.groupby('celltype').apply(lambda x: x.mean())
    # df_pred_cell.index = dataset_atac.array_celltype
    # df_pred_cell = np.log1p(df_pred_cell*1e5)

    # cell-level corr
    list_corr_cell = []
    for i, label in df_pred_exp.index:
        sub_cor = \
            stats.pearsonr(np.array(df_true_cell.loc[label, :]),
                           np.array(df_pred_exp.loc[i, :])[0])
        list_corr_cell.append(sub_cor[0])
    cell_corr = np.nanmean(list_corr_cell)
    # celltype-level corr
    list_corr_celltype = []
    for celltype_label in df_pred_cell.index:
        sub_cor = stats.pearsonr(np.array(df_true_cell.loc[celltype_label, :]),
                                 np.array(df_pred_cell.loc[celltype_label, :]))
        list_corr_celltype.append(sub_cor[0])
    celltype_corr = np.nanmean(list_corr_celltype)
    # gene-level corr
    list_corr_gene = []
    for i in df_true_cell.columns:
        sub_cor = stats.pearsonr(df_true_cell.loc[:, i], df_pred_cell.loc[:, i])
        list_corr_gene.append(sub_cor[0])
    gene_corr = np.nanmean(list_corr_gene)

    # pred
    df_pred_exp = pd.DataFrame(pred_exp, index=test_cell, columns=peaks[:number_gene])
    df_pred_exp = np.exp(df_pred_exp)
    adata_pred = ad.AnnData(
        X=df_pred_exp.copy()*1e5, obs=dataset_atac.adata.obs.loc[df_pred_exp.index, :])

    if path_out is not None:
        adata_pred.write(path_out)

    sc.pp.normalize_total(adata_pred)
    sc.pp.log1p(adata_pred)
    # sc.pp.highly_variable_genes(adata_edge, n_top_genes=30000, flavor='seurat')
    # adata = adata_edge[:, adata_edge.var.highly_variable]
    sc.pp.scale(adata_pred, max_value=10)
    # sc.pp.regress_out(adata_pred, keys='nCount_ATAC')
    sc.tl.pca(adata_pred, svd_solver='arpack', n_comps=50)
    sc.pp.neighbors(adata_pred, n_neighbors=30, n_pcs=50)
    sc.tl.umap(adata_pred, min_dist=0.5)
    sc.pl.umap(adata_pred, color=['celltype'])
    # silhouette score
    asw_norm = (silhouette_score(adata_pred.obsm['X_pca'], true_label) + 1) / 2
    ari_norm = calculate_ari(adata_pred.obsm['X_pca'], true_label)

    return cell_corr, celltype_corr, gene_corr, asw_norm, ari_norm


def ground_truth(path_data, dataset_atac, path_hic, list_celltype_hic):
    df_graph = dataset_atac.df_graph
    list_po = df_graph.apply(lambda x: (x['region1'], x['region2']), axis=1).tolist()
    path_cell_interactome = os.path.join(path_data, 'cell_interactome')
    if not os.path.exists(path_cell_interactome):
        os.mkdir(path_cell_interactome)
    file_po = os.path.join(path_cell_interactome, 'PO.bed')
    with open(file_po, 'w') as w_po:
        for pair in list_po:
            pair_1 = pair[0]
            pair_2 = pair[1]
            chrom = pair_2.split('-')[0]
            start = pair_2.split('-')[1]
            end = pair_2.split('-')[2]
            w_po.write(f"{chrom}\t{start}\t{end}\t{pair_1}\t{pair_2}\n")

    list_df_po = []
    for celltype_hic in list_celltype_hic:
        path_cell_hg38 = os.path.join(path_hic, celltype_hic)
        file_pp_cell = os.path.join(path_cell_hg38, 'PP.txt')
        file_po_cell = os.path.join(path_cell_hg38, 'PO.txt')
        path_cell = os.path.join(path_cell_interactome, celltype_hic)
        if not os.path.exists(path_cell):
            os.mkdir(path_cell)
        file_intersect = os.path.join(path_cell, 'PO_intersect.txt')
        # os.system(f"bedtools intersect -a {file_po} -b {file_po_cell} -wao > {file_intersect}")
        subprocess.run(
            f"bedtools intersect -a {file_po} -b {file_po_cell} -wao > {file_intersect}",
            shell=True
        )
        # subprocess.call(['bedtools', 'intersect', '-a', file_po, '-b', file_po_cell, '-wao',
        #                  '>', file_intersect])
        df_intersect = pd.read_csv(file_intersect, sep='\t', header=None)
        df_po_cell = df_intersect.loc[df_intersect.iloc[:, 5] != '.', [3, 4]]
        df_po_cell = df_po_cell.drop_duplicates()
        df_po_cell.columns = ['region1', 'region2']
        df_pp_cell = pd.read_csv(file_pp_cell, sep='\t', header=None)
        df_pp_cell.columns = ['region1', 'region2']
        set_po_cell = set(df_po_cell.apply(lambda x: (x['region1'], x['region2']), axis=1).tolist())
        list_df_po.append(pd.Series([pair in set_po_cell for pair in list_po], index=list_po))

    df_cell_interactome = pd.concat(list_df_po, axis=1)
    df_cell_interactome.columns = list_celltype_hic
    df_cell_interactome_sub = df_cell_interactome.loc[np.sum(df_cell_interactome, axis=1) != 0, :]
    df_cell_interactome_spec = df_cell_interactome.loc[np.sum(df_cell_interactome, axis=1) == 1, :]

    file_cell_interactome = os.path.join(path_cell_interactome, 'cell_interactome_ori.txt')
    df_cell_interactome.to_csv(file_cell_interactome, sep='\t')
    file_cell_interactome_sub = os.path.join(path_cell_interactome, 'cell_interactome.txt')
    df_cell_interactome_sub.to_csv(file_cell_interactome_sub, sep='\t')
    file_cell_interactome_spec = os.path.join(path_cell_interactome, 'cell_interactome_spec.txt')
    df_cell_interactome_spec.to_csv(file_cell_interactome_spec, sep='\t')

    return df_cell_interactome_spec


def celltype_score(path_model, df_celltype_weight, df_cell_interactome,
                   list_celltype_hic, folder_weight):
    celltypes = np.unique(df_celltype_weight.index)
    path_weight_interatome = os.path.join(path_model, folder_weight)
    if not os.path.exists(path_weight_interatome):
        os.mkdir(path_weight_interatome)
    for celltype_hic in list_celltype_hic:
        cell_pair = df_cell_interactome.index[df_cell_interactome[celltype_hic]]
        cell_pair = list(set(cell_pair).intersection(set(df_celltype_weight.columns)))
        list_scores = []
        for celltype in celltypes:
            cell_score = df_celltype_weight.loc[celltype, cell_pair]
            list_scores.append(cell_score)
        df_scores = pd.concat(list_scores, axis=1)
        file_scores = os.path.join(path_weight_interatome, f'{celltype_hic}_scores.txt')
        df_scores.to_csv(file_scores, sep='\t')

    return


if __name__ == '__main__':
    time_start = time()

    time_end = time()
    print(time_end - time_start)
