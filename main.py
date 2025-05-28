# _*_ coding: utf-8 _*_
# @author: Drizzle_Zhang
# @file: main.py.py
# @time: 2025/3/20 10:41

from time import time
import os


def main(file_atac, file_rna, path_tmp, cell_aggregation, hg19_to_hg38,
         min_features, max_features, min_percent, all_genes, num_gene,
         hidden_size, num_head, lr, epoch, patience, batch_size,
         train_val_split, pretrain_model_path, pretrain_file_name, lambda_loss,
         path_out, num_sample_cell, device, random_seed):

    if not os.path.exists(path_tmp):
        os.mkdir(path_tmp)
    if not os.path.exists(path_out):
        os.mkdir(path_out)
    path_data_process = os.path.join(path_tmp, "output_data")
    if not os.path.exists(path_data_process):
        os.mkdir(path_data_process)

    # run data preprocessing
    status = os.system(
        f"python generate_training_data.py --file_atac {file_atac} "
        f"--file_rna {file_rna} --path_out {path_data_process} "
        f"--cell_aggregation {cell_aggregation} --hg19tohg38 {hg19_to_hg38} "
        f"--min_features {min_features} --max_features {max_features} --min_percent {min_percent} "
        f"--use_all_gene {all_genes} --num_gene {num_gene}"
    )
    if status == 0:
        print('Data Preprocessing is Completed!')
    else:
        print('Error in Data Preprocessing!')
        return

    # run transcriptional simulation
    folder_ckpt = "simulation_model"
    status = os.system(
        f"python transcription_simulation.py "
        f"--path_in {path_data_process} --folder_ckpt {path_tmp}/{folder_ckpt} "
        f"--hidden_size {hidden_size} --num_head {num_head} --lr {lr} --max_epoch {epoch} "
        f"--patience {patience} --batch_size {batch_size} --train_val_split {train_val_split} "
        f"--pretrain_model_path {pretrain_model_path} --pretrain_file_name {pretrain_file_name} "
        f"--num_sample_cell {num_sample_cell} --lambda_loss {lambda_loss} "
        f"--device {device} --random_seed {random_seed}"
    )
    if status == 0:
        print('Transcription Simulation is Completed!')
    else:
        print('Error in Transcription Simulation!')
        return

    # predict single-cell regulation scores
    status = os.system(
        f"python attribution.py --path_in {path_data_process} --path_out {path_out} "
        f"--config_simulation {path_tmp}/{folder_ckpt}/args.pkl "
        f"--model_simulation {path_tmp}/{folder_ckpt}/best_model.pt"
    )
    if status == 0:
        print('Single-cell regulation score prediction is completed!')
    else:
        print('Error in Single-cell regulation score prediction!')
        return

    return


if __name__ == '__main__':
    time_start = time()

    import argparse
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument("--file_atac", type=str, help="h5ad file path of scATAC-seq data")
    parser.add_argument("--file_rna", type=str, help="h5ad file path of scRNA-seq data")
    parser.add_argument("--path_out", type=str, default='./res_path',
                        help="The result path of single-cell CRRs predicted by SCRIPT")
    parser.add_argument("--path_tmp", type=str, default='./temp_path',
                        help="The path of data preparation and intermediate results")

    # parameters of preprocessing
    parser.add_argument("--cell_aggregation", type=int, default=5, help="do cell aggregation")
    parser.add_argument("--hg19tohg38", type=int, default=0, help="transform hg19 to hg38")
    parser.add_argument("--min_features", type=int, default=-1, help="min number of features (For QC)")
    parser.add_argument("--max_features", type=int, default=-1, help="max number of features (For QC)")
    parser.add_argument("--min_percent", type=float, default=0.05, help="min percent of features (For QC)")
    parser.add_argument("--use_all_gene", type=int, default=0, help="use all genes in scRNA-seq data")
    parser.add_argument("--num_gene", type=int, default=300, help="number of marker genes per cell type")

    # parameters of Transcription Simulation
    parser.add_argument("--hidden_size", type=int, default=64, help="Hidden size of Graph Neural Networks")
    parser.add_argument("--num_head", type=int, default=4, help="Number of heads for Graph Attention Networks")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--max_epoch", type=int, default=300, help="Maximum epoches")
    parser.add_argument("--patience", type=int, default=30, help="Early stop patience. Model training stops when best validation correlation does not decrease for a consecutive ``patience`` epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--train_val_split", type=float, default=0.6, help="Proportion of training data")
    parser.add_argument("--pretrain_model_path", type=str, default='./graph_pretrain', help="Path of checkpoint of pretrain model")
    parser.add_argument("--pretrain_file_name", type=str, default='ssgae_save.pt', help="File name of checkpoint of pretrain model")
    parser.add_argument("--num_sample_cell", type=int, default=-1, help="Number of cells selected to train model")
    parser.add_argument("--lambda_loss", type=int, default=1, help="Weight of expression loss")
    parser.add_argument("--device", type=int, default=0, help="GPU number")
    parser.add_argument("--random_seed", type=int, default=2024, help="Random seed")

    args = parser.parse_args()
    main(
        file_atac=args.file_atac,
        file_rna=args.file_rna,
        path_out=args.path_out,
        path_tmp=args.path_tmp,
        cell_aggregation=args.cell_aggregation,
        hg19_to_hg38=args.hg19tohg38,
        min_features=args.min_features,
        max_features=args.max_features,
        min_percent=args.min_percent,
        all_genes=args.use_all_gene,
        num_gene=args.num_gene,
        hidden_size=args.hidden_size,
        num_head=args.num_head,
        lr=args.lr,
        epoch=args.max_epoch,
        patience=args.patience,
        batch_size=args.batch_size,
        train_val_split=args.train_val_split,
        pretrain_model_path=args.pretrain_model_path,
        pretrain_file_name=args.pretrain_file_name,
        num_sample_cell=args.num_sample_cell,
        lambda_loss=args.lambda_loss,
        device=args.device,
        random_seed=args.random_seed,
    )

    time_end = time()
    print(time_end - time_start)
