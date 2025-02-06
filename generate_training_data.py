import os
from time import time

import pandas as pd
import scanpy as sc
import pickle

from data_process import prepare_model_input

num_gene_percelltype = 300


def generate_data(
        file_atac,
        file_rna,
        path_out,
        cell_aggregation=1,
        hg19_to_hg38=0,
        min_features=0,
        max_features=10e8,
        min_percent=0,
):
    # cortex
    time_start = time()

    df_rna_cortex = pd.read_csv(file_rna, sep='\t', index_col=0)

    cell_aggregation = True if cell_aggregation > 0 else False
    hg19_to_hg38 = True if hg19_to_hg38 > 0 else False
    # normalization = True if normalization > 0 else False
    print(f"file_atac_in: {file_atac}, \nfile_rna_in: {file_rna}, \npath_out: {path_out}")
    print(f"cell_aggregation: {cell_aggregation}, hg19_to_hg38: {hg19_to_hg38}")
    if min_features < 0:
        min_features = None
    if max_features < 0:
        max_features = None
    prepare_model_input(
        path_out, file_atac, df_rna_cortex,
        min_features=min_features, max_features=max_features, min_percent=min_percent,
        deepen_data=cell_aggregation, hg19tohg38=hg19_to_hg38
    )

    time_end = time()
    print(time_end - time_start)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_atac", type=str, help="h5ad file path of scATAC-seq data")
    parser.add_argument("--file_rna", type=str, help="h5ad file path of scRNA-seq data")
    parser.add_argument("--path_out", type=str, help="output path")
    parser.add_argument("--cell_aggregation", type=int, default=1, help="do cell aggregation")
    parser.add_argument("--hg19tohg38", type=int, default=0, help="transform hg19 to hg38")
    # parser.add_argument("--normalization", type=int, default=0, help="do normalization")
    parser.add_argument("--min_features", type=int, default=-1, help="min number of features (For QC)")
    parser.add_argument("--max_features", type=int, default=-1, help="max number of features (For QC)")
    parser.add_argument("--min_percent", type=float, default=0.05, help="min percent of features (For QC)")
    args = parser.parse_args()
    generate_data(
        file_atac=args.file_atac,
        file_rna=args.file_rna,
        path_out=args.path_out,
        cell_aggregation=args.cell_aggregation,
        hg19_to_hg38=args.hg19tohg38,
        min_features=args.min_features,
        max_features=args.max_features,
        min_percent=args.min_percent
    )
