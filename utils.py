import os
import argparse
import random
import psutil
import yaml
import logging
import pickle
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
from torch import optim as optim

import dgl
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import scanpy as sc
import numpy as np
import pandas as pd
import anndata as ad
from scipy import stats
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def save_args(args, ckpt_path):
    with open(os.path.join(ckpt_path, "args.pkl"), "wb") as f:
        pickle.dump(args, f)


def calculate_ari(mat_in, true_label):
    num_label = len(np.unique(true_label))
    kmeans = KMeans(n_clusters=num_label).fit(mat_in)
    cluster_label = kmeans.labels_
    ari = adjusted_rand_score(cluster_label, true_label)
    ari_norm = (ari + 1) / 2

    return ari_norm


def evaluate(dataset_atac, pred_exp, true_label, true_exp, test_cell, path_out=None, simple=False):
    peaks = dataset_atac.array_peak
    mask_numpy = np.array([0 if peak[:3] == 'chr' else 1 for peak in peaks])
    number_gene = np.sum(mask_numpy)

    # true_label = dataset_atac.adata.obs.loc[test_cell, 'celltype_rna'].tolist()
    # true exp
    df_true_cell = pd.DataFrame(
        true_exp,
        index=pd.MultiIndex.from_arrays([test_cell, true_label],
                                        names=['index', 'celltype']),
        columns=peaks[-number_gene:])
    df_true_cell = df_true_cell.groupby('celltype').apply(lambda x: x.mean())
    # df_true_cell = np.log1p(df_true_cell*1e5)
    # df_true_cell.index = dataset_atac.array_celltype
    # pred exp
    df_pred_exp = pd.DataFrame(pred_exp,
                               index=pd.MultiIndex.from_arrays([test_cell, true_label],
                                                               names=['index', 'celltype']),
                               columns=peaks[-number_gene:])
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

    if simple:
        asw_norm, ari_norm = 0, 0
    else:
        # pred
        df_pred_exp = pd.DataFrame(pred_exp, index=test_cell, columns=peaks[-number_gene:])
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


def accuracy(y_pred, y_true):
    y_true = y_true.squeeze().long()
    preds = y_pred.max(1)[1].type_as(y_true)
    correct = preds.eq(y_true).double()
    correct = correct.sum().item()
    return correct / len(y_true)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True


def get_current_lr(optimizer):
    return optimizer.state_dict()["param_groups"][0]["lr"]


def build_yaml_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="./graph_pretrain/pretrain_conf.yaml")
    args = parser.parse_args(args=[])
    return args


def build_attribution_yaml_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, help="path to config file")
    args = parser.parse_args()
    print(args)
    return args


def build_args():
    parser = argparse.ArgumentParser(description="GAT")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    # parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max_epoch", type=int, default=1,
                        help="number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=-1)

    parser.add_argument("--num_heads", type=int, default=4,
                        help="number of hidden attention heads")
    parser.add_argument("--num_out_heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--num_dec_layers", type=int, default=1)
    parser.add_argument("--num_remasking", type=int, default=3)    
    parser.add_argument("--num_hidden", type=int, default=64,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in_drop", type=float, default=.2,
                        help="input feature dropout")
    parser.add_argument("--attn_drop", type=float, default=.1,
                        help="attention dropout")
    parser.add_argument("--norm", type=str, default=None)
    parser.add_argument("--lr", type=float, default=0.0005,
                        help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0,
                        help="weight decay")
    parser.add_argument("--negative_slope", type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument("--activation", type=str, default="prelu")
    parser.add_argument("--mask_rate", type=float, default=0.5)
    parser.add_argument("--remask_rate", type=float, default=0.5)
    parser.add_argument("--remask_method", type=str, default="random")
    parser.add_argument("--mask_type", type=str, default="mask",
                        help="`mask` or `drop`")
    parser.add_argument("--mask_method", type=str, default="random")
    parser.add_argument("--drop_edge_rate", type=float, default=0.0)
    parser.add_argument("--drop_edge_rate_f", type=float, default=0.0)

    parser.add_argument("--encoder", type=str, default="gat")
    parser.add_argument("--decoder", type=str, default="gat")
    parser.add_argument("--loss_fn", type=str, default="sce")
    parser.add_argument("--alpha_l", type=float, default=2)
    parser.add_argument("--optimizer", type=str, default="adam")
    
    # parser.add_argument("--max_epoch_f", type=int, default=300)
    # parser.add_argument("--lr_f", type=float, default=0.01)
    # parser.add_argument("--weight_decay_f", type=float, default=0.0)
    # parser.add_argument("--linear_prob", action="store_true", default=False)

    parser.add_argument("--no_pretrain", action="store_true")
    parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--logging", action="store_true")
    parser.add_argument("--scheduler", action="store_true", default=True)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--batch_size_f", type=int, default=128)
    parser.add_argument("--sampling_method", type=str, default="saint", help="sampling method, `lc` or `saint`")

    parser.add_argument("--label_rate", type=float, default=1.0)
    parser.add_argument("--ego_graph_file_path", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="data")

    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--full_graph_forward", action="store_true", default=False)
    parser.add_argument("--delayed_ema_epoch", type=int, default=0)
    parser.add_argument("--replace_rate", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.996)
    parser.add_argument("--do_feat_encoder", type=bool, default=False)
    parser.add_argument("--nonzero_mask", type=bool, default=True)

    args = parser.parse_args()
    return args


def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name == "selu":
        return nn.SELU()
    elif name == "elu":
        return nn.ELU()
    elif name == "silu":
        return nn.SiLU()
    elif name is None:
        return nn.Identity()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


def identity_norm(x):
    def func(x):
        return x
    return func


def create_norm(name):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    elif name == "identity":
        return identity_norm
    else:
        # print("Identity norm")
        return None


def create_optimizer(opt, model, lr, weight_decay, get_num_layer=None, get_layer_scale=None):
    opt_lower = opt.lower()
    parameters = model.parameters()
    opt_args = dict(lr=lr, weight_decay=weight_decay)

    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]

    if opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        return optim.SGD(parameters, **opt_args)
    else:
        raise NotImplementedError("Invalid optimizer")

    return optimizer


def show_occupied_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2


# -------------------
def mask_edge(graph, mask_prob):
    E = graph.num_edges()

    mask_rates = torch.ones(E) * mask_prob
    masks = torch.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx


def drop_edge(graph, drop_rate, return_edges=False):
    if drop_rate <= 0:
        return graph

    graph = graph.remove_self_loop()

    n_node = graph.num_nodes()
    edge_mask = mask_edge(graph, drop_rate)
    src, dst = graph.edges()

    nsrc = src[edge_mask]
    ndst = dst[edge_mask]

    ng = dgl.graph((nsrc, ndst), num_nodes=n_node)
    ng = ng.add_self_loop()

    return ng


def visualize(x, y, method="tsne"):
    if torch.is_tensor(x):
        x = x.cpu().numpy()
        
    if torch.is_tensor(y):
        y = y.cpu().numpy()
    
    if method == "tsne":
        func = TSNE(n_components=2)
    else:
        func = PCA(n_components=2)
    out = func.fit_transform(x)
    plt.scatter(out[:, 0], out[:, 1], c=y)
    plt.savefig("vis.png")
    

def load_best_configs(args):
    dataset_name = args.dataset
    config_path = os.path.join("configs", f"{dataset_name}.yaml")
    with open(config_path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)
    for k, v in configs.items():
        if "lr" in k or "weight_decay" in k:
            v = float(v)
        setattr(args, k, v)
    logging.info(f"----- Using best configs from {config_path} -----")

    return args


def load_yaml_conf(args):
    # if "other_config_path" in kwargs:
    #     conf_path_list = kwargs["other_config_path"]
    # else:
    #     conf_path_list = []

    # if os.path.exists(args.config_path):
    #     conf_path_list.append(args.config_path)
    # for path in conf_path_list:
    if os.path.exists(args.config_path):
        with open(args.config_path, "r") as f:
            conf = yaml.load(f, yaml.FullLoader)
        logging.info(f"load config from {args.config_path}")
        for k, v in conf.items():
            if "lr" in k or "weight_decay" in k:
                v = float(v)
            setattr(args, k, v)
    else:
        raise ValueError(f"{args.config_path} does not exist")
    if hasattr(args, "finetune_config_path"):
        with open(args.finetune_config_path, "r") as f:
            conf = yaml.load(f, yaml.FullLoader)
        logging.info(f"load config from {args.finetune_config_path}")
        for k, v in conf.items():
            setattr(args, k, v)


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    scheduler = np.concatenate((warmup_schedule, schedule))
    assert len(scheduler) == epochs * niter_per_ep
    return scheduler


class TBLogger(object):
    def __init__(self, log_path="./logging_data", name="run"):
        super(TBLogger, self).__init__()

        if not os.path.exists(log_path):
            os.makedirs(log_path, exist_ok=True)

        self.last_step = 0
        self.log_path = log_path
        raw_name = os.path.join(log_path, name)
        name = raw_name
        for i in range(1000):
            name = raw_name + str(f"_{i}")
            if not os.path.exists(name):
                break
        self.writer = SummaryWriter(logdir=name)

    def note(self, metrics, step=None):
        if step is None:
            step = self.last_step
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)
        self.last_step = step

    def finish(self):
        self.writer.close()
