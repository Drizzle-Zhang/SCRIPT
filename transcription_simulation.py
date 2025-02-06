import gc
import time

import pandas as pd
from torch_geometric.data import DataLoader as GDataLoader
from audtorch.metrics.functional import pearsonr

import dgl
from model import MyLoss, MyLossExp
from model import PretrainGAT, GATLabelConcat, GAT
from trainer_base import TrainerBase
import torch
import numpy as np
import logging
import os
from utils import save_args, evaluate, accuracy
from tqdm import tqdm
from graph_pretrain import build_model
import random
from sklearn.metrics import accuracy_score
from pretrain import load_graph_data
import argparse
import yaml


class FinetuneTrainerLabel(TrainerBase):
    def __init__(
            self,
            dataset,
            hidden_size,
            num_head,
            lr,
            epoch,
            patience,
            batch_size,
            train_val_split,
            load_pretrain_emb,
            device,
            random_seed,
            pretrain_model=None,
            pretrain_model_path=None,
            pretrain_file_name=None,
            large_emb_init=True,
            lambda_loss=1,
            label_loss=True
    ):
        super().__init__(f"{int(time.time())}")
        self.dataset = dataset
        self.in_dim = 1
        self.hidden_size = hidden_size
        self.num_head = num_head
        self.lr = lr
        self.train_val_split = train_val_split
        self.device = device
        self.epoch = epoch
        self.patience = patience
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.load_pretrain_emb = load_pretrain_emb
        self.number_gene, self.number_node, self.number_celltype = self._get_args_from_dataset()
        self.pretrain_model = pretrain_model
        self.pretrain_model_path = pretrain_model_path
        self.pretrain_file_name = pretrain_file_name
        self.large_emb_init = large_emb_init
        self.lambda_loss = lambda_loss
        self.label_loss = label_loss
        self.use_pretrain = self.load_pretrain_emb or self.large_emb_init

    def _get_args_from_dataset(self):
        mask_numpy = np.array([0 if peak[:3] == 'chr' else 1 for peak in self.dataset.array_peak])
        number_gene = int(np.sum(mask_numpy))
        number_node = self.dataset.array_peak.shape[0]
        number_celltype = len(np.unique(self.dataset.array_celltype))
        return number_gene, number_node, number_celltype

    def _init_pretrain_model(self):
        logger.info(f"get pretrain emb from model {self.pretrain_model_path}")
        self.pretrain_model.load_state_dict(
            torch.load(os.path.join(
                str(self.pretrain_model_path),
                self.pretrain_file_name
            ))
        )
        self.pretrain_model.to(self.device)

    def _get_pretrain_emb(self, graph_list):
        self.pretrain_model.eval()
        with torch.no_grad():
            for batch in tqdm(graph_list):
                if self.large_emb_init:
                    graph = dgl.batch(batch.graph).to(self.device)
                else:
                    graph = batch.graph.to(self.device)
                node_tensor = batch.x.to(self.device).unsqueeze(-1)
                batch.emb = self.pretrain_model.embed(graph, node_tensor)

    def _get_pretrain_emb_batch(self, data_batch):
        self.pretrain_model.eval()
        with torch.no_grad():
            graph = dgl.batch(data_batch.graph).to(self.device)
            node_tensor = data_batch.x.unsqueeze(-1).to(self.device)
            emb = self.pretrain_model.embed(graph, node_tensor)
        return emb

    def train_exp_label_from_pretrain(self, model, criterion, optimizer, loader):
        model.train()
        list_loss1 = []
        list_loss2 = []
        list_loss = []
        for data in tqdm(loader):  # Iterate in batches over the training dataset.
            if self.load_pretrain_emb:
                data.x = data.x.to(self.device)
                data.emb = data.emb.to(self.device)
            elif self.large_emb_init:
                data.x = data.x.to(self.device)
                data.emb = self._get_pretrain_emb_batch(data)
            else:
                data.x = data.x.to(self.device)
                data.emb = None
            data.edge_index = data.edge_index.to(self.device)
            data.batch = data.batch.to(self.device)
            out1, out2 = model(data.x, data.edge_index, data.batch, emb=data.emb)
            y_exp = data.y_exp.view(out2.shape).to(self.device)
            loss, loss1, loss2 = criterion(out1, out2, data.y.long().to(self.device), y_exp)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            list_loss.append(loss.cpu().detach().numpy())
            list_loss1.append(loss1.cpu().detach().numpy())
            list_loss2.append(loss2.cpu().detach().numpy())

        loss_cat = np.array(list_loss)
        loss1_cat = np.array(list_loss1)
        loss2_cat = np.array(list_loss2)
        return {'loss': np.mean(loss_cat), 'loss_label': np.mean(loss1_cat),
                'loss_exp': np.mean(loss2_cat)}

    def test_exp_label_from_pretrain(self, model, loader):
        total_acc = 0
        with torch.no_grad():
            list_exp = []
            list_pred = []
            list_y = []
            list_y_exp = []
            list_cell = []
            for data in tqdm(loader):
                if self.load_pretrain_emb:
                    data.x = data.x.to(self.device)
                    data.emb = data.emb.to(self.device)
                elif self.large_emb_init:
                    data.x = data.x.to(self.device)
                    data.emb = self._get_pretrain_emb_batch(data)
                else:
                    data.x = data.x.to(self.device)
                    data.emb = None
                edge_index = data.edge_index.to(self.device)
                data_batch = data.batch.to(self.device)
                out1, out2 = model(data.x, edge_index, data_batch, emb=data.emb)
                pred = out1.argmax(dim=1)
                list_pred.append(pred.cpu().detach().numpy())
                list_exp.append(out2.cpu().detach().numpy())
                list_y.append(data.y.cpu().detach().numpy())
                list_y_exp.append(data.y_exp.view(out2.shape).cpu().detach().numpy())
                list_cell.extend(data.cell)

        pred_exp, pred_label, true_label, true_exp = \
            np.concatenate(list_exp), np.concatenate(list_pred), np.concatenate(list_y), np.concatenate(list_y_exp)
        if self.label_loss:
            total_acc = accuracy_score(pred_label, true_label)
        cell_corr, celltype_corr, gene_corr, asw_norm, ari_norm = evaluate(
            self.dataset, pred_exp, true_label, true_exp, list_cell, os.path.join(self.ckpt_path, "pred_exp.h5ad"))

        return {'acc': total_acc, 'cell_corr': cell_corr, 'celltype_corr': celltype_corr,
                'gene_corr': gene_corr, 'asw_norm': asw_norm, 'ari_norm': ari_norm}

    def _get_data_loader(self):
        random.seed(self.random_seed)
        random.shuffle(self.dataset.list_graph)
        finetune_split_idx = int(len(self.dataset.list_graph) * self.train_val_split)
        train_data_loader = GDataLoader(
            self.dataset.list_graph[:finetune_split_idx],
            batch_size=self.batch_size,
            shuffle=True
        )
        val_data_loader = GDataLoader(
            self.dataset.list_graph[finetune_split_idx:],
            batch_size=self.batch_size,
            shuffle=False
        )
        return train_data_loader, val_data_loader

    def train_process(self):
        model = GATLabelConcat(
            input_channels=self.in_dim,
            emb_channels=self.hidden_size,
            num_head=self.num_head,
            num_gene=self.number_gene,
            num_nodes=self.number_node,
            num_celltype=self.number_celltype
        ).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)
        list_loss = []
        list_eval_train = []
        list_eval_test = []
        list_best_epoch = []

        if self.load_pretrain_emb:
            self._init_pretrain_model()
            self._get_pretrain_emb(self.dataset.list_graph)

        if self.large_emb_init:
            self._init_pretrain_model()

        train_data_loader, val_data_loader = self._get_data_loader()
        if self.label_loss:
            list_weights = []
            for i in range(len(self.dataset.array_celltype)):
                sub_dataset = [data for data in trainer.dataset.list_graph if data.y == i]
                sub_len = len(sub_dataset)
                sub_weight = 1/sub_len
                list_weights.append(sub_weight)
            criterion = MyLoss(self.lambda_loss, (torch.tensor(list_weights)/sum(list_weights)).to(self.device))
        else:
            criterion = MyLossExp()

        best_corr = 0.0
        best_epoch = 0
        for epoch in range(self.epoch):
            logger.info(f"begin epoch {epoch}")
            loss_dict = self.train_exp_label_from_pretrain(model, criterion, optimizer, train_data_loader)
            eval_dict_train = self.test_exp_label_from_pretrain(model, train_data_loader)
            eval_dict_test = self.test_exp_label_from_pretrain(model, val_data_loader)
            list_loss.append(loss_dict)
            list_eval_train.append(eval_dict_train)
            list_eval_test.append(eval_dict_test)

            test_corr = eval_dict_test['cell_corr'] + eval_dict_test['gene_corr']
            if test_corr > best_corr:
                torch.save(
                    model, os.path.join(
                        self.ckpt_path, "best_model.pt"
                    )
                )
                best_corr = test_corr
                best_epoch = epoch
            # torch.save(model, os.path.join(self.ckpt_path, f"epoch_{epoch}.pt"))
            list_best_epoch.append(best_epoch)
            logger.info(
                f"Epoch: {epoch:03d}, "
                f"Total loss: {loss_dict['loss']:.4f}, Label loss: {loss_dict['loss_label']:.4f}, "
                f"Expression loss: {loss_dict['loss_exp']:.4f},  \n"
                f"Train Acc: {eval_dict_train['acc']:.4f}, Train Corr: {eval_dict_train['cell_corr']:.4f}, "
                f"Train Gene Corr: {eval_dict_train['gene_corr']:.4f},  \n"
                f"Test Acc: {eval_dict_test['acc']:.4f}, Test Corr: {eval_dict_test['cell_corr']:.4f}, "
                f"Test Gene Corr: {eval_dict_test['gene_corr']:.4f},   \n"
                f"Best Corr: {best_corr:.4f}"
            )

            # res record
            df_res = pd.concat(
                [pd.DataFrame(list_loss), pd.DataFrame(list_eval_train),
                 pd.DataFrame(list_eval_test)], axis=1)
            df_res['best_epoch'] = list_best_epoch
            df_res.to_csv(os.path.join(self.ckpt_path, "finetune_res.tsv"), sep='\t')

            # early stop
            if epoch - best_epoch >= self.patience:
                print("Model have been saved in 'best_model.pt'")
                # torch.save(model, os.path.join(self.ckpt_path, f"model_param_epoch_{epoch}.pt"))
                break

        # torch.save(
        #     model, os.path.join(
        #         self.ckpt_path, "finetune_model.pt"
        #     )
        # )

        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_in", type=str, help="The result path of training data generation")
    parser.add_argument("--hidden_size", type=int, default=64, help="Hidden size of Graph Neural Networks")
    parser.add_argument("--num_head", type=int, default=4, help="Number of heads for Graph Attention Networks")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--max_epoch", type=int, default=400, help="Maximum epoches")
    parser.add_argument("--patience", type=int, default=30, help="Early stop patience. Model training stops when best validation correlation does not decrease for a consecutive ``patience`` epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--train_val_split", type=float, default=0.8, help="Proportion of training data")
    # parser.add_argument("--use_pretrain", type=str, default='true', help="Whether to use pretrain model")
    parser.add_argument("--pretrain_model_path", type=str, default='./graph_pretrain', help="Path of checkpoint of pretrain model")
    parser.add_argument("--pretrain_file_name", type=str, default='ssgae_save.pt', help="File name of checkpoint of pretrain model")
    # parser.add_argument("--label_loss", type=str, default='true', help="Whether use label loss")
    parser.add_argument("--lambda_loss", type=int, default=1, help="Weight of expression loss")
    parser.add_argument("--device", type=int, default=0, help="GPU number")
    parser.add_argument("--random_seed", type=int, default=2024, help="Random seed")

    args = parser.parse_args()

    graph_dataset = load_graph_data(args.path_in, sample_graph=False)
    train_device = args.device if torch.cuda.is_available() else torch.device("cpu")
    with open('./graph_pretrain/pretrain_conf.yaml', 'r') as file:
        dict_pretrain = yaml.load(file, Loader=yaml.FullLoader)
    args_pretrain = argparse.Namespace(**dict_pretrain)
    f_pretrain_model = build_model(args_pretrain)

    trainer = FinetuneTrainerLabel(
        dataset=graph_dataset,
        hidden_size=args.hidden_size,
        num_head=args.num_head,
        lr=args.lr,
        epoch=args.max_epoch,
        patience=args.patience,
        batch_size=args.batch_size,
        train_val_split=args.train_val_split,
        load_pretrain_emb=True,
        device=train_device,
        random_seed=args.random_seed,
        pretrain_model=f_pretrain_model,
        pretrain_model_path=args.pretrain_model_path,
        pretrain_file_name=args.pretrain_file_name,
        large_emb_init=False,
        lambda_loss=args.lambda_loss,
        label_loss=True
    )

    logger = logging.getLogger(f'{trainer.ckpt_path}')
    logger.setLevel(logging.INFO)
    log_file = os.path.join(trainer.ckpt_path, 'log_file.log')
    filehandler = logging.FileHandler(log_file)
    streamhandler = logging.StreamHandler()
    filehandler.setLevel(logging.INFO)
    streamhandler.setLevel(logging.INFO)
    logger.addHandler(streamhandler)
    logger.addHandler(filehandler)
    logger.info(f"args: {args}")

    save_args(args, trainer.ckpt_path)
    trainer.train_process()

