from torch_geometric.data import DataLoader as GDataLoader, Dataset
from tqdm import tqdm

import time
from model import PretrainGAT, GATLabelConcat, GAT
from trainer_base import TrainerBase
import torch
import numpy as np
import logging
import os
from utils import save_args
from graph_pretrain import build_model
import random
import captum.attr as attr
import pandas as pd
import datatable as dt
import argparse
from pretrain import load_graph_data
import yaml
import pickle

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


class AttributeDataset(Dataset):
    def __init__(self, data, transform=None):
        super().__init__(transform)
        self.data = data
        # self.gene = gene

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class Attribution(TrainerBase):
    def __init__(
            self,
            dataset,
            path_out,
            hidden_size,
            num_head,
            lr,
            epoch,
            batch_size,
            train_val_split,
            load_pretrain_emb,
            large_emb_init,
            device,
            random_seed,
            pretrain_model=None,
            pretrain_model_path=None,
            finetune_model_path=None,
            label_loss=True
    ):
        super().__init__(f"{int(time.time())}")
        self.dataset = dataset
        self.path_out = path_out
        self.in_dim = 1
        self.hidden_size = hidden_size
        self.num_head = num_head
        self.lr = lr
        self.train_val_split = train_val_split
        self.device = device
        self.epoch = epoch
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.load_pretrain_emb = load_pretrain_emb
        self.number_gene, self.number_node, self.number_celltype = self._get_args_from_dataset()
        self.pretrain_model = pretrain_model
        self.pretrain_model_path = pretrain_model_path
        self.finetune_model = None
        self.finetune_model_path = finetune_model_path
        self.df_attr = None
        self.do_edge_attr = True
        self.label_loss = label_loss
        self.large_emb_init = large_emb_init
        self.use_pretrain = self.load_pretrain_emb or self.large_emb_init

        model = GATLabelConcat(
            input_channels=self.in_dim,
            emb_channels=self.hidden_size,
            num_head=self.num_head,
            num_gene=self.number_gene,
            num_nodes=self.number_node,
            num_celltype=self.number_celltype
        )
        self.finetune_model = model.to(self.device)
        logging.info(f"get finetune model from {self.finetune_model_path}")
        finetune_model = \
            torch.load(self.finetune_model_path, map_location=lambda storage, loc: storage).to(self.device)
        if isinstance(finetune_model, GATLabelConcat) or isinstance(finetune_model, PretrainGAT) or isinstance(finetune_model, GAT):
            self.finetune_model.load_state_dict(finetune_model.state_dict(), strict=False)
        else:
            self.finetune_model.load_state_dict(finetune_model, strict=False)
        # self.finetune_model = torch.nn.DataParallel(self.finetune_model)
        self.finetune_model.to(self.device)
        self.finetune_model.eval()

        logging.info(f"get pretrain emb from model {self.pretrain_model_path}")
        self.pretrain_model.load_state_dict(torch.load(self.pretrain_model_path))
        # self.pretrain_model = torch.nn.DataParallel(self.pretrain_model)
        self.pretrain_model.to(self.device)
        self.pretrain_model.eval()

    def _get_args_from_dataset(self):
        mask_numpy = np.array([0 if peak[:3] == 'chr' else 1 for peak in self.dataset.array_peak])
        number_gene = int(np.sum(mask_numpy))
        number_node = self.dataset.array_peak.shape[0]
        number_celltype = len(np.unique(self.dataset.array_celltype))
        return number_gene, number_node, number_celltype

    def _get_data_loader(self):
        random.seed(self.random_seed)
        random.shuffle(self.dataset.list_graph)
        finetune_split_idx = int(len(self.dataset.list_graph) * self.train_val_split)
        train_dataset = AttributeDataset(
            self.dataset.list_graph[:finetune_split_idx]
        )
        val_dataset = AttributeDataset(
            self.dataset.list_graph[finetune_split_idx:]
        )
        train_data_loader = GDataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        val_data_loader = GDataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        return train_data_loader, val_data_loader

    def model_forward_edge(self, edge_mask, data, finetune_model, gene_idx):
        batchsize = len(torch.unique(data.batch))
        node_tensor = data.x.view(batchsize, self.number_node)
        list_emb = []
        for idx, graph in enumerate(data.graph):
            list_emb.append(self.pretrain_model.embed(graph, node_tensor[idx, :].unsqueeze(-1)))
        emb = torch.cat(list_emb, dim=0)
        _, out_exp  = finetune_model(data.x, data.edge_index, data.batch, emb=emb, edge_weight=edge_mask)
        return out_exp[:, gene_idx]

    def _do_attribution(self, data, ixg, idx_gene, batch_size):
        edge_mask = torch.ones(data.edge_index.shape[1])
        edge_mask.requires_grad = True
        mask = ixg.attribute(
            edge_mask.to(self.device),
            additional_forward_args=(
                data.to(self.device),
                self.finetune_model,
                idx_gene
            )
        )
        num_col = mask.shape[0] // batch_size
        mask = mask.view(batch_size, num_col)
        edge_mask = mask.cpu().detach().numpy().astype('float32')
        return edge_mask

    def attribution_process(self):
        train_data_loader, val_data_loader = self._get_data_loader()

        peaks = self.dataset.array_peak
        mask_numpy = np.array([1 if peak[:3] == 'chr' else 0 for peak in peaks])
        number_cre = int(np.sum(mask_numpy))
        whole_array_merge = []
        list_pairs = []
        # idx_gene = 305
        ixg = attr.InputXGradient(self.model_forward_edge)
        idx_edge = self.dataset.list_graph[0].edge_index.numpy().astype('int32')

        batch_num = 0
        for data in tqdm(val_data_loader):
            list_array = []
            sub_pairs = []
            unique_batch = torch.unique(data.batch)
            batch_size = len(unique_batch)
            for idx_gene in range(self.dataset.df_rna.shape[1]):
                peak_array_idx = number_cre + idx_gene
                idx_peak = idx_edge[0, idx_edge[1, :] == peak_array_idx]
                gene = peaks[peak_array_idx]
                sub_pairs.extend((gene, peaks[idx]) for idx in idx_peak)

                attribution_res = self._do_attribution(data, ixg, idx_gene, batch_size)

                sub_array = attribution_res[:, idx_edge[1, :] == peak_array_idx]
                list_array.append(sub_array)
            array_merge = np.concatenate(list_array, axis=1)
            whole_array_merge.append(array_merge)
                # if idx_gene % 50 == 0:
                #     print(f"Calculating progress: {idx_gene + 1} genes completed")
            batch_num = batch_num + 1
            # if batch_num > 1:
            #     break

        whole_array_merge = np.concatenate(whole_array_merge, axis=0)
        list_cell = [item for data in val_data_loader for item in data.cell]
        print('-' * 10, whole_array_merge.shape, len(list_cell), len(sub_pairs), list_cell[0])
        df_merge = pd.DataFrame(whole_array_merge, index=list_cell, columns=sub_pairs)

        file_weight = os.path.join(
            self.path_out, "attribution_scores.csv"
        )
        # df_merge.to_csv(file_weight, sep='\t')

        # use datatable to save (faster)
        df_merge.insert(loc=0, column='Cells', value=list_cell)
        df_merge = dt.Frame(df_merge)
        df_merge.to_csv(file_weight)

        self.df_attr = df_merge

        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_in", type=str, help="The result path of training data generation")
    parser.add_argument("--path_out", type=str, help="The result path of single-cell CRRs predicted by SCRIPT")
    parser.add_argument("--config_simulation", type=str, help="The configure file of transcription simulation model")
    parser.add_argument("--model_simulation", type=str, help="The model file of transcription simulation model")
    # parser.add_argument("--pretrain_model_path", type=str, default='./graph_pretrain', help="Path of checkpoint of pretrain model")
    # parser.add_argument("--pretrain_file_name", type=str, default='ssgae_save.pt', help="File name of checkpoint of pretrain model")
    parser.add_argument("--device", type=int, default=0, help="GPU number")

    args = parser.parse_args()

    graph_dataset = load_graph_data(args.path_in, sample_graph=False)
    attr_device = args.device if torch.cuda.is_available() else torch.device("cpu")
    # pretrain config
    with open('./graph_pretrain/pretrain_conf.yaml', 'r') as file:
        dict_pretrain = yaml.load(file, Loader=yaml.FullLoader)
    args_pretrain = argparse.Namespace(**dict_pretrain)
    f_pretrain_model = build_model(args_pretrain)

    # simulation configure
    with open(args.config_simulation, 'rb') as file:
        args_simulation = pickle.load(file)

    class_attr = Attribution(
        dataset=graph_dataset,
        path_out=args.path_out,
        hidden_size=args_simulation.hidden_size,
        num_head=args_simulation.num_head,
        lr=args_simulation.lr,
        epoch=args_simulation.max_epoch,
        batch_size=args_simulation.batch_size,
        train_val_split=0.6,
        load_pretrain_emb=True,
        large_emb_init=False,
        device=attr_device,
        random_seed=args_simulation.random_seed,
        pretrain_model=f_pretrain_model,
        pretrain_model_path=os.path.join(
            args_simulation.pretrain_model_path, args_simulation.pretrain_file_name),
        finetune_model_path=args.model_simulation,
        label_loss=True
    )
    save_args(args, class_attr.ckpt_path)
    class_attr.attribution_process()
