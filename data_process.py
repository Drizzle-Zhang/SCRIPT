# _*_ coding: utf-8 _*_
# @author: Drizzle_Zhang
# @file: data_process.py
# @time: 2023/2/1 14:27


from time import time
import os
import torch
from torch_geometric.data import InMemoryDataset, Data
from collections import defaultdict
import episcanpy.api as epi
import scanpy as sc
import numpy as np
import pandas as pd
from pandas import DataFrame
import anndata as ad
from anndata import AnnData
from typing import Optional, Mapping, List, Union
from scipy import sparse
import sklearn
from statsmodels.distributions.empirical_distribution import ECDF
import pickle
import random
import cosg


def tfidf(X: Union[np.ndarray, sparse.spmatrix]) -> Union[np.ndarray, sparse.spmatrix]:
    r"""
    TF-IDF normalization (following the Seurat v3 approach)
    Parameters
    ----------
    X
        Input matrix
    Returns
    -------
    X_tfidf
        TF-IDF normalized matrix
    """
    idf = X.shape[0] / X.sum(axis=0)
    if sparse.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf


def lsi(
        adata: AnnData, n_components: int = 20,
        use_top_features: Optional[bool] = False, min_cutoff: float = 0.05, **kwargs
) -> None:
    r"""
    LSI analysis
    Parameters
    ----------
    adata
        Input dataset
    n_components
        Number of dimensions to use
    use_top_features
        Whether to find most frequently observed features and use them
    min_cutoff
        Cutoff for feature to be included in the ``adata.var['select_feature']``.
        For example, '0.05' to set the top 95% most common features as the selected features.
    **kwargs
        Additional keyword arguments are passed to
        :func:`sklearn.utils.extmath.randomized_svd`
    """
    if "random_state" not in kwargs:
        kwargs["random_state"] = 0  # Keep deterministic as the default behavior

    adata_use = adata.copy()
    if use_top_features:
        adata_use.var['featurecounts'] = np.array(np.sum(adata_use.X, axis=0))[0]
        df_var = adata_use.var.sort_values(by='featurecounts')
        ecdf = ECDF(df_var['featurecounts'])
        df_var['percentile'] = ecdf(df_var['featurecounts'])
        df_var["selected_feature"] = (df_var['percentile'] > min_cutoff)
        adata_use.var = df_var.loc[adata_use.var.index, :]

    # factor_size = int(np.median(np.array(np.sum(adata_use.X, axis=1))))
    X_norm = np.log1p(tfidf(adata_use.X) * 1e4)
    if use_top_features:
        X_norm = X_norm.toarray()[:, adata_use.var["selected_feature"]]
    else:
        X_norm = X_norm.toarray()
    svd = sklearn.decomposition.TruncatedSVD(n_components=n_components, algorithm='arpack')
    X_lsi = svd.fit_transform(X_norm)
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    adata.obsm["X_lsi"] = X_lsi


class ATACDataset(object):
    def __init__(self, data_root: str, raw_filename: str, file_chrom: str):
        self.data_root = data_root
        self.raw_filename = raw_filename
        self.adata = self.load_matrix()
        # self.adata.raw = self.adata.copy()
        self.path_process = os.path.join(data_root, 'processed_files')
        if not os.path.exists(self.path_process):
            os.mkdir(self.path_process)
        self.file_peaks_sort = os.path.join(self.path_process, 'peaks.sort.bed')
        if os.path.exists(self.file_peaks_sort):
            os.remove(self.file_peaks_sort)
        self.file_chrom = file_chrom
        # tools
        # basepath = "./"
        basepath = os.path.abspath(__file__)
        folder = os.path.dirname(basepath)
        self.bedtools = os.path.join(folder, 'tools/bedtools/bin/bedtools')
        self.liftover = os.path.join(folder, 'tools/liftOver')
        self.file_chain = os.path.join(folder, 'tools/files_liftOver/hg19ToHg38.over.chain.gz')
        self.generate_peaks_file()
        self.all_promoter_genes = None
        self.all_promoter_peaks = None
        self.gene_cre_idx = None
        self.all_proximal_genes = None
        self.adata_merge = None
        self.other_peaks = None
        self.df_graph = None
        self.list_graph = None
        self.list_graph_train_idx = None
        self.list_graph_val_idx = None
        self.array_peak = None
        self.array_celltype = None
        self.df_rna = None
        self.dict_promoter = None
        self.df_gene_peaks = None
        self.df_proximal = None
        self.df_distal = None
        self.df_eqtl = None
        self.df_tf = None
        self.df_graph_index = None
        self.df_graph_index_cre = None
        self.tensor_merge = None
        self.list_gene_peak = None
        self.df_graph_cre = None

    def load_matrix(self):
        if self.raw_filename[-5:] == '.h5ad':
            adata_atac = sc.read_h5ad(self.raw_filename)
            # print(adata_atac)
        elif self.raw_filename[-4:] == '.tsv':
            adata_atac = ad.read_text(self.raw_filename,
                delimiter='\t', first_column_names=True, dtype='int')
            epi.pp.sparse(adata_atac)
        else:
            raise ImportError("Input format error!")
        return adata_atac

    def generate_peaks_file(self):
        df_chrom = pd.read_csv(self.file_chrom, sep='\t', header=None, index_col=0)
        df_chrom = df_chrom.iloc[:24]
        file_peaks_atac = os.path.join(self.path_process, 'peaks.bed')
        fmt_peak = "{chrom_peak}\t{start_peak}\t{end_peak}\t{peak_id}\n"
        with open(file_peaks_atac, 'w') as w_peak:
            for one_peak in self.adata.var.index:
                chrom_peak = one_peak.strip().split('-')[0]
                # locs = one_peak.strip().split(':')[1]
                if chrom_peak in df_chrom.index:
                    start_peak = one_peak.strip().split('-')[1]
                    end_peak = one_peak.strip().split('-')[2]
                    peak_id = one_peak
                    w_peak.write(fmt_peak.format(**locals()))

        os.system(f"{self.bedtools} sort -i {file_peaks_atac} > {self.file_peaks_sort}")

    def hg19tohg38(self):
        path_peak = os.path.join(self.data_root, 'peaks_process')
        if not os.path.exists(path_peak):
            os.mkdir(path_peak)

        file_ummap = os.path.join(path_peak, 'unmap.bed')
        file_peaks_hg38 = os.path.join(path_peak, 'peaks_hg38.bed')
        os.system(f"{self.liftover} {self.file_peaks_sort} {self.file_chain} "
                  f"{file_peaks_hg38} {file_ummap}")

        df_hg19 = pd.read_csv(self.file_peaks_sort, sep='\t', header=None)
        df_hg19['length'] = df_hg19.iloc[:, 2] - df_hg19.iloc[:, 1]
        len_down = np.min(df_hg19['length']) - 20
        len_up = np.max(df_hg19['length']) + 100

        df_hg38 = pd.read_csv(file_peaks_hg38, sep='\t', header=None)
        df_hg38['peak_hg38'] = df_hg38.apply(lambda x: f"{x[0]}-{x[1]}-{x[2]}", axis=1)
        df_hg38['length'] = df_hg38.iloc[:, 2] - df_hg38.iloc[:, 1]
        df_hg38 = df_hg38.loc[df_hg38['length'] < len_up, :]
        df_hg38 = df_hg38.loc[df_hg38['length'] > len_down, :]

        sel_peaks_hg19 = df_hg38.iloc[:, 3]
        adata_atac_out = self.adata[:, sel_peaks_hg19]
        adata_atac_out.var['peaks_hg19'] = adata_atac_out.var.index
        adata_atac_out.var.index = df_hg38['peak_hg38']

        self.adata = adata_atac_out

    def quality_control(self, min_features: int = 1000, max_features: int = 60000,
                        min_percent: Optional[float] = None, min_cells: Optional[int] = None):
        adata_atac = self.adata
        epi.pp.filter_cells(adata_atac, min_features=min_features)
        epi.pp.filter_cells(adata_atac, max_features=max_features)
        # print("-"*10, adata_atac.obs.head())
        if min_percent is not None:
            by = adata_atac.obs['celltype']
            agg_idx = pd.Index(by.cat.categories) \
                if isinstance(by, pd.CategoricalDtype) \
                else pd.Index(np.unique(by))
            agg_sum = sparse.coo_matrix((
                np.ones(adata_atac.shape[0]), (
                    agg_idx.get_indexer(by),
                    np.arange(adata_atac.shape[0])
                )
            )).tocsr()
            sum_x = agg_sum @ (adata_atac.X != 0)
            df_percent = pd.DataFrame(
                sum_x.toarray(), index=agg_idx, columns=adata_atac.var.index
            ) / adata_atac.obs.value_counts('celltype').loc[agg_idx].to_numpy()[:, np.newaxis]
            df_percent_max = np.max(df_percent, axis=0)
            sel_peaks = df_percent.columns[df_percent_max > min_percent]
            self.adata = self.adata[:, sel_peaks]
        elif min_cells is not None:
            epi.pp.filter_features(adata_atac, min_cells=min_cells)

    def deepen_atac(self, num_pc: int = 50, num_cell_merge: int = 10):
        random.seed(1234)
        adata_atac_sample_cluster = self.adata.copy()
        lsi(adata_atac_sample_cluster, n_components=num_pc)
        adata_atac_sample_cluster.obsm["X_lsi"] = adata_atac_sample_cluster.obsm["X_lsi"][:, 1:]
        sc.pp.neighbors(adata_atac_sample_cluster, use_rep="X_lsi", metric="cosine",
                        n_neighbors=int(num_cell_merge), n_pcs=num_pc-1)

        list_atac_index = []
        list_neigh_index = []
        for cell_atac in list(adata_atac_sample_cluster.obs.index):
            cell_atac = [cell_atac]
            cell_atac_index = np.where(adata_atac_sample_cluster.obs.index == cell_atac[0])[0]
            cell_neighbor_idx = \
                np.nonzero(
                    adata_atac_sample_cluster.obsp['connectivities'].getcol(
                        cell_atac_index).toarray())[0]
            if num_cell_merge >= len(cell_neighbor_idx):
                cell_sample_atac = np.hstack([cell_atac_index, cell_neighbor_idx])
            else:
                cell_sample_atac = np.hstack([cell_atac_index,
                                              np.random.choice(cell_neighbor_idx, num_cell_merge,
                                                               replace=False)])
            list_atac_index.extend([cell_atac_index[0] for _ in range(len(cell_sample_atac))])
            list_neigh_index.append(cell_sample_atac)

        agg_sum = sparse.coo_matrix((
            np.ones(len(list_atac_index)), (np.array(list_atac_index), np.hstack(list_neigh_index))
        )).tocsr()
        array_atac = agg_sum @ self.adata.X

        # self.adata = self.adata.copy()
        self.adata.X = None
        self.adata.X = array_atac

    def add_promoter(self, file_tss: str, flank_proximal: int = 2000):
        sc.pp.normalize_total(self.adata)
        sc.pp.log1p(self.adata)

        df_tss = pd.read_csv(file_tss, sep='\t', header=None)
        df_tss.columns = ['chrom', 'tss', 'symbol', 'ensg_id', 'strand']
        df_tss = df_tss.drop_duplicates(subset='symbol')
        df_tss.index = df_tss['symbol']
        df_tss['tss_start'] = df_tss['tss'] - 2000
        df_tss['tss_end'] = df_tss['tss'] + 2000
        df_tss['proximal_start'] = df_tss['tss'] - flank_proximal
        df_tss['proximal_end'] = df_tss['tss'] + flank_proximal
        file_promoter = os.path.join(self.path_process, 'promoter.txt')
        file_proximal = os.path.join(self.path_process, 'proximal.txt')
        df_promoter = \
            df_tss.loc[:, ['chrom', 'tss_start', 'tss_end', 'symbol', 'ensg_id', 'strand']]
        df_promoter.to_csv(file_promoter, sep='\t', header=False, index=False)
        df_proximal = \
            df_tss.loc[:, ['chrom', 'proximal_start', 'proximal_end',
                           'symbol', 'ensg_id', 'strand']]
        df_proximal.to_csv(file_proximal, sep='\t', header=False, index=False)

        self.generate_peaks_file()

        # add promoter to adata
        file_peaks_promoter = os.path.join(self.path_process, 'peaks_promoter.txt')
        os.system(f"{self.bedtools} intersect -a {self.file_peaks_sort} -b {file_promoter} -wao "
                  f"> {file_peaks_promoter}")
        dict_promoter = defaultdict(list)
        with open(file_peaks_promoter, 'r') as w_pro:
            for line in w_pro:
                list_line = line.strip().split('\t')
                if list_line[4] == '.':
                    continue
                gene_symbol = list_line[7]
                peak = list_line[3]
                gene_tss = df_tss.loc[gene_symbol, 'tss']
                coor_cre = (int(list_line[2]) + int(list_line[1]))/2
                dist_gene_cre = abs(gene_tss - coor_cre)
                dict_promoter[gene_symbol].append((peak, dist_gene_cre))

        all_genes = dict_promoter.keys()
        list_peaks_promoter = []
        list_genes_promoter = []
        for gene_symbol in all_genes:
            sub_peaks = dict_promoter[gene_symbol]
            sel_peak = ''
            min_dist = 2000
            for sub_peak in sub_peaks:
                if sub_peak[1] < min_dist:
                    sel_peak = sub_peak[0]
                    min_dist = sub_peak[1]
            if sel_peak != '':
                list_peaks_promoter.append(sel_peak)
                list_genes_promoter.append(gene_symbol)
        self.all_promoter_genes = list_genes_promoter
        self.all_promoter_peaks = list_peaks_promoter

        adata_gene_promoter = self.adata[:, list_peaks_promoter]
        adata_promoter = \
            ad.AnnData(X=adata_gene_promoter.X,
                       var=pd.DataFrame(data={'cRE_type': np.full(len(list_genes_promoter),
                                                                  'Promoter')},
                                        index=list_genes_promoter),
                       obs=pd.DataFrame(index=adata_gene_promoter.obs.index))

        adata_peak = self.adata.copy()
        adata_peak.obs = pd.DataFrame(index=self.adata.obs.index)
        adata_peak.var = pd.DataFrame(data={'node_type': np.full(adata_peak.var.shape[0], 'cRE')},
                                      index=adata_peak.var.index)
        # adata_merge = ad.concat([adata_promoter, adata_peak], axis=1)
        adata_merge = ad.concat([adata_peak, adata_promoter], axis=1)
        self.adata_merge = adata_merge

        # proximal regulation
        file_peaks_proximal = os.path.join(self.path_process, 'peaks_proximal.txt')
        os.system(f"{self.bedtools} intersect -a {self.file_peaks_sort} -b {file_proximal} -wao "
                  f"> {file_peaks_proximal}")
        dict_proximal = defaultdict(list)
        with open(file_peaks_proximal, 'r') as w_pro:
            for line in w_pro:
                list_line = line.strip().split('\t')
                if list_line[4] == '.':
                    continue
                gene_symbol = list_line[7].strip().split('<-')[0]
                peak = list_line[3]
                dict_proximal[gene_symbol].append(peak)
        self.dict_promoter = dict_proximal

        all_genes = dict_proximal.keys()
        list_peaks_proximal = []
        list_genes_proximal = []
        for gene_symbol in all_genes:
            sub_peaks = dict_proximal[gene_symbol]
            list_genes_proximal.extend([gene_symbol for _ in range(len(sub_peaks))])
            list_peaks_proximal.extend(sub_peaks)
        self.all_proximal_genes = set(list_genes_proximal)

        self.df_gene_peaks = \
            pd.DataFrame({'gene': list_genes_proximal, 'peak': list_peaks_proximal})
        self.df_proximal = \
            pd.DataFrame({'region1': list_genes_proximal, 'region2': list_peaks_proximal,
                          'type': ['proximal']*len(list_peaks_proximal)})
        set_gene = set(self.df_rna.columns).intersection(self.all_promoter_genes)
        self.df_proximal = \
            self.df_proximal.loc[self.df_proximal["region1"].apply(lambda x: x in set_gene), :]

        return

    def build_graph(self, path_interaction: str, sel_interaction: str = 'PO'):
        file_pp = os.path.join(path_interaction, 'PP.txt')
        file_po = os.path.join(path_interaction, 'PO.txt')
        if sel_interaction == 'PP' or sel_interaction == 'ALL':
            df_pp_pre = pd.read_csv(file_pp, sep='\t', header=None)
            df_pp_pre = \
                df_pp_pre.loc[df_pp_pre.apply(
                    lambda x: x.iloc[0] in self.all_promoter_genes and x.iloc[1] in self.all_promoter_genes, axis=1), :]
            df_pp_pre.columns = ['region1', 'gene']
            df_gene_peaks = self.df_gene_peaks.copy()
            df_gene_peaks.columns = ['gene', 'region2']
            df_pp = pd.merge(left=df_pp_pre, right=df_gene_peaks, on='gene')
            df_pp = df_pp.loc[:, ['region1', 'region2']]
        if sel_interaction == 'PO' or sel_interaction == 'ALL':
            file_po_peaks = os.path.join(self.path_process, 'peaks_PO.bed')
            os.system(f"{self.bedtools} intersect -a {self.file_peaks_sort} -b {file_po} -wao "
                      f"> {file_po_peaks}")
            list_dict = []
            with open(file_po_peaks, 'r') as r_po:
                for line in r_po:
                    list_line = line.strip().split('\t')
                    peak = list_line[3]
                    gene_symbol = list_line[8]
                    if gene_symbol in self.all_promoter_genes:
                        list_dict.append({"region1": gene_symbol, "region2": peak})
            df_po = pd.DataFrame(list_dict)
        if sel_interaction == 'PP':
            df_interaction = df_pp
        elif sel_interaction == 'PO':
            df_interaction = df_po
        elif sel_interaction == 'ALL':
            df_interaction = pd.concat([df_pp, df_po])
        else:
            print("Error: please set correct parameter 'sel_interaction'! ")
            return

        self.df_distal = df_interaction.drop_duplicates()
        self.df_distal['type'] = ['distal']*self.df_distal.shape[0]
        set_gene = set(self.df_rna.columns)
        self.df_distal = \
            self.df_distal.loc[self.df_distal["region1"].apply(lambda x: x in set_gene), :]
        self.df_graph = pd.concat([self.df_proximal, self.df_distal], axis=0)

        return

    def generate_data_tensor(self):
        graph_data = self.df_graph
        df_graph_all = graph_data
        df_pro = df_graph_all.loc[df_graph_all['type'] == 'proximal', :]
        df_pro = df_pro.loc[:, ['region1', 'region2']]
        df_distal = df_graph_all.loc[df_graph_all['type'] != 'proximal', :]
        df_graph_new = pd.merge(df_pro, df_distal, on='region1')
        df_graph_new = df_graph_new.loc[:, ['region2_x', 'region2_y', 'type']]
        df_graph_new.columns = ['region1', 'region2', 'type']
        self.df_graph_cre = df_graph_new
        graph_cre = self.df_graph_cre
        adata_merge = self.adata_merge
        all_cre_gene = set(graph_data['region1']).union(set(graph_data['region2']))
        all_peaks = all_cre_gene
        adata_merge_peak = adata_merge[:, [one_peak for one_peak in adata_merge.var.index
                                           if one_peak in all_peaks]]
        array_peak = np.array(adata_merge_peak.var.index)
        list_gene_peak = [one_peak for one_peak in array_peak if one_peak[:3] != 'chr']
        peak_dict = {val: idx for idx, val in enumerate(array_peak)}

        # cRE-Gene
        array_region1 = graph_data['region1'].map(peak_dict.get)
        array_region2 = graph_data['region2'].map(peak_dict.get)
        df_graph_index = torch.tensor([np.array(array_region2), np.array(array_region1)],
                                      dtype=torch.int64)

        # cRE-cRE
        array_cre_region1 = graph_cre['region1'].map(peak_dict.get)
        array_cre_region2 = graph_cre['region2'].map(peak_dict.get)
        df_graph_index_1 = torch.tensor([np.array(array_cre_region2), np.array(array_cre_region1)],
                                        dtype=torch.int64)
        df_graph_index_2 = torch.tensor([np.array(array_cre_region1), np.array(array_cre_region2)],
                                        dtype=torch.int64)
        df_graph_index_cre = torch.concat([df_graph_index_1, df_graph_index_2], dim=1)

        df_merge_peak = adata_merge_peak.to_df()
        tensor_merge = torch.Tensor(np.array(df_merge_peak))

        # gene peaks
        self.gene_cre_idx = pd.Series(self.all_promoter_peaks).map(peak_dict.get)

        self.df_graph_index = df_graph_index
        self.df_graph_index_cre = df_graph_index_cre
        self.tensor_merge = tensor_merge
        self.array_peak = array_peak
        self.list_gene_peak = list_gene_peak

        return

    def generate_data_list(self, rna_exp=False):
        df_graph_index = self.df_graph_index
        df_graph_index_cre = self.df_graph_index_cre
        tensor_merge = self.tensor_merge
        adata_atac = self.adata

        array_celltype = np.unique(np.array(adata_atac.obs['celltype']))
        self.array_celltype = array_celltype
        label_dict = {val: idx for idx, val in enumerate(array_celltype)}

        if rna_exp:
            self.df_rna = self.df_rna.loc[:, self.list_gene_peak]
            self.df_rna = self.df_rna / np.array(np.sum(self.df_rna, axis=1))[:, np.newaxis]
        list_graph = []
        for i_cell in range(0, adata_atac.n_obs):
            one_cell = adata_atac.obs.index[i_cell]
            label = adata_atac.obs.loc[one_cell, 'celltype']
            label_idx = torch.tensor([label_dict[label]], dtype=torch.int16)
            if rna_exp:
                label_rna = adata_atac.obs.loc[one_cell, 'celltype_rna']
                label_exp = self.df_rna.loc[label_rna, self.list_gene_peak].tolist()
                cell_data = Data(
                    x=tensor_merge[i_cell, :],
                    edge_index_cre=df_graph_index_cre,
                    edge_index=df_graph_index,
                    y=label_idx,
                    y_exp=torch.tensor(label_exp),
                    cell=one_cell
                )
            else:
                cell_data = Data(
                    x=tensor_merge[i_cell, :],
                    edge_index_cre=df_graph_index_cre,
                    edge_index=df_graph_index,
                    y=label_idx,
                    cell=one_cell
                )
            list_graph.append(cell_data)

        self.list_graph = list_graph
        return


def process_rna(file_rna, num_gene_percelltype):
    adata_rna = sc.read_h5ad(file_rna)
    # protein coding gene
    basepath = os.path.abspath(__file__)
    folder = os.path.dirname(basepath)
    file_gene_hg38 = os.path.join(folder, 'data/genes.protein.tss.tsv')
    df_gene_hg38 = pd.read_csv(file_gene_hg38, sep='\t', header=None)
    pretein_genes = df_gene_hg38.iloc[:, 2]
    sel_pretein_genes = sorted(list(set(pretein_genes.tolist()).intersection(adata_rna.var.index)))
    adata_rna_cosg = adata_rna[:, sel_pretein_genes].copy()
    # find marker genes
    cosg.cosg(adata_rna_cosg,
              key_added='cosg',
              mu=1,
              remove_lowly_expressed=True,
              expressed_pct=0.2,
              n_genes_user=num_gene_percelltype,
              use_raw=True,
              groupby='celltype')
    cosg_genes = []
    for i in range(adata_rna_cosg.uns['cosg']['names'].shape[0]):
        cosg_genes.extend(list(adata_rna_cosg.uns['cosg']['names'][i]))
    cosg_genes = list(set(cosg_genes))
    adata_rna = adata_rna[:, cosg_genes]
    df_rna = pd.DataFrame(adata_rna.X.toarray(),
                          index=adata_rna.obs.index, columns=adata_rna.var.index)
    df_rna_cell = df_rna
    df_rna_cell['celltype'] = adata_rna.obs.loc[:, 'celltype']
    df_rna_cell = df_rna_cell.groupby('celltype').apply(lambda x: x.sum())
    df_rna_celltype = df_rna_cell.iloc[:, :-1]

    return df_rna_celltype


def prepare_model_input(path_data_root: str, file_atac: str, df_rna_celltype: DataFrame,
                        min_features: Optional[float] = None, max_features: Optional[float] = None,
                        min_percent: Optional[float] = 0.05,
                        hg19tohg38: bool = False, deepen_data: bool = True):
    if not os.path.exists(path_data_root):
        os.mkdir(path_data_root)

    basepath = os.path.abspath(__file__)
    folder = os.path.dirname(basepath)
    # basepath = "./"
    file_chrom_hg38 = os.path.join(folder, 'data/hg38.chrom.sizes')

    dataset_ATAC = ATACDataset(data_root=path_data_root, raw_filename=file_atac,
                               file_chrom=file_chrom_hg38)
    # dataset_ATAC.adata.obs['celltype'] = dataset_ATAC.adata.obs['seurat_annotations']
    if hg19tohg38:
        dataset_ATAC.hg19tohg38()
    vec_num_feature = np.array(np.sum(dataset_ATAC.adata.X != 0, axis=1))
    if min_features is None:
        default_min = int(np.percentile(vec_num_feature, 1))
        min_features = default_min
    if max_features is None:
        default_max = int(np.percentile(vec_num_feature, 99))
        max_features = default_max
    dataset_ATAC.quality_control(min_features=min_features, max_features=max_features,
                                 min_percent=min_percent)
    # dataset_ATAC.quality_control(min_features=3000, min_cells=5)

    # deep atac
    if deepen_data:
        dataset_ATAC.deepen_atac(num_cell_merge=10)

    # add RNA-seq data
    dataset_ATAC.df_rna = df_rna_celltype

    file_gene_hg38 = os.path.join(folder, 'data/genes.protein.tss.tsv')
    dataset_ATAC.add_promoter(file_gene_hg38, flank_proximal=2_000)

    # Hi-C
    path_hic = os.path.join(folder, 'data')
    dataset_ATAC.build_graph(path_hic, sel_interaction='ALL')

    dataset_ATAC.generate_data_tensor()

    # save data
    file_atac_test = os.path.join(path_data_root, 'dataset_atac.pkl')
    with open(file_atac_test, 'wb') as w_pkl:
        str_pkl = pickle.dumps(dataset_ATAC)
        w_pkl.write(str_pkl)

    return dataset_ATAC


if __name__ == '__main__':
    time_start = time()

    time_end = time()
    print(time_end - time_start)
