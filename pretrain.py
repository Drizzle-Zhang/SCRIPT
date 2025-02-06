import os
import pickle
import time
import random

import dgl

import logging
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from audtorch.metrics.functional import pearsonr
from model import GAT
from pretrain_loader import PretrainLoader
import sys

from utils import (
    create_optimizer,
    set_random_seed,
    TBLogger,
    build_yaml_args,
    load_yaml_conf
)
from graph_pretrain import build_model
from graph_pretrain.finetune_model import FinetuneModel

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
random.seed(2024)


def load_graph(root_path):
    edge_index = torch.load(os.path.join(root_path, "edge_tensor.pt"))
    node_tensor = torch.load(os.path.join(root_path, "node_tensor.pt"))
    logging.info(f"load graph from {root_path}, edge shape: {edge_index.shape}, node shape: {node_tensor.shape}")
    graph = dgl.graph((edge_index[0].numpy().tolist(), edge_index[1].numpy().tolist()))
    node_list = [torch.unsqueeze(i, dim=-1) for i in node_tensor]

    # graph.ndata['feat'] = tmp_node
    # num_features = node_list[0].shape[1]
    # print(graph)
    # num_classes = dataset.num_classes
    return node_list, graph


def load_graph_gene(root_path, max_gene_idx, gene_first=False):
    edge_index_gene = torch.load(os.path.join(root_path, "edge_tensor_gene.pt"))
    edge_index_cre = torch.load(os.path.join(root_path, "edge_tensor_cre.pt"))
    node_tensor = torch.load(os.path.join(root_path, "node_tensor.pt"))
    logging.info(
        f"load graph from {root_path}, gene edge shape: {edge_index_gene.shape}, "
        f"cre edge shape: {edge_index_cre.shape}, node shape: {node_tensor.shape}"
    )
    if gene_first:
        edge_index_cre = edge_index_cre - max_gene_idx
        graph = dgl.graph((edge_index_cre[0].numpy().tolist(), edge_index_cre[1].numpy().tolist()))
        cre_node_list = [torch.unsqueeze(i[max_gene_idx:], dim=-1) for i in node_tensor]
    else:
        graph = dgl.graph((edge_index_cre[0].numpy().tolist(), edge_index_cre[1].numpy().tolist()))
        cre_node_list = [torch.unsqueeze(i[max_gene_idx:], dim=-1) for i in node_tensor]
    return cre_node_list, edge_index_cre, graph


def load_graph_data(root_path, sample_graph=False):
    """
    Load dataset pickle file
    Parameters
    ----------
    sample_graph
    root_path

    Returns
    -------

    """
    with open(os.path.join(root_path, "dataset_atac.pkl"), 'rb') as f:
        dataset = pickle.load(f)
    dataset.generate_data_list(rna_exp=True)

    if sample_graph:
        random.shuffle(dataset.list_graph)
        dataset.list_graph = dataset.list_graph[:10000]

    first_edge_index = torch.cat([dataset.list_graph[0].edge_index_cre[0],
                                  dataset.list_graph[0].edge_index[0]])
    second_edge_index = torch.cat([dataset.list_graph[0].edge_index_cre[1],
                                   dataset.list_graph[0].edge_index[1]])
    graph = dgl.graph((first_edge_index.numpy().tolist(), second_edge_index.numpy().tolist()))
    graph = dgl.add_self_loop(graph)

    logging.info(f"generate dgl graph from path: {root_path}")
    for data in tqdm(dataset.list_graph):
            data.graph = graph
    return dataset


def load_graph_data_prune(root_path, add_gene=False, edge_prune=True, sample_graph=False):
    """
    Load dataset pickle file
    Parameters
    ----------
    sample_graph
    edge_prune
    add_gene
    root_path

    Returns
    -------

    """
    with open(os.path.join(root_path, "dataset_atac.pkl"), 'rb') as f:
        dataset = pickle.load(f)
    dataset.generate_data_list(rna_exp=True)

    if sample_graph:
        random.shuffle(dataset.list_graph)
        dataset.list_graph = dataset.list_graph[:200]

    # before_time = time.time()
    # with multiprocessing.Pool(16) as pool:
    #     list_graph = pool.map(map_graph, dataset.list_graph)
    # after_time = time.time()
    # logging.info(f"generate graph time: {after_time - before_time} seconds")
    # for idx, g in enumerate(list_graph):
    #     dataset.list_graph[idx].graph = g
    logging.info(f"generate dgl graph from path: {root_path}")
    for data in tqdm(dataset.list_graph):
        if edge_prune:
            edge_index_cre_x_scr = torch.index_select(data.x, 0, data.edge_index_cre[0])
            edge_index_cre_x_target = torch.index_select(data.x, 0, data.edge_index_cre[1])
            nonzero_idx = torch.nonzero(edge_index_cre_x_scr + edge_index_cre_x_target)
            data.edge_index_cre = torch.index_select(data.edge_index_cre, 1, nonzero_idx.squeeze())
        if not add_gene:
            data.graph = dgl.graph((data.edge_index_cre[0].numpy().tolist(), data.edge_index_cre[1].numpy().tolist()))
            data.graph = dgl.add_self_loop(data.graph)
        else:
            # generate dgl graph for pretrain model, finetune model only uses data.edge_index
            first_edge_index = torch.cat([data.edge_index_cre[0], data.edge_index[0]])
            second_edge_index = torch.cat([data.edge_index_cre[1], data.edge_index[1]])
            data.graph = dgl.graph((first_edge_index.numpy().tolist(), second_edge_index.numpy().tolist()))
            data.graph = dgl.add_self_loop(data.graph)
    return dataset


def do_finetune(model, finetune_train_graph, finetune_val_graph, finetune_args, finetune_device):
    finetune_train_list = process_finetune_data(
        model, finetune_train_graph, finetune_args["max_gene_node"], finetune_args["do_pretrain"], finetune_device)
    finetune_train_data_loader = DataLoader(
        finetune_train_list, batch_size=16, shuffle=True
    )
    finetune_eval_list = process_finetune_data(
        model, finetune_val_graph, finetune_args["max_gene_node"], finetune_args["do_pretrain"], finetune_device)
    finetune_eval_data_loader = DataLoader(
        finetune_eval_list, batch_size=16, shuffle=False
    )
    eval_loss, eval_corr = train_finetune_model(
        finetune_train_data_loader, finetune_eval_data_loader, finetune_device, finetune_args)
    return eval_loss, eval_corr


def downstream(dataset, model, finetune_args, finetune_device):
    random.shuffle(dataset.list_graph)
    finetune_split_idx = int(len(dataset.list_graph) * 0.9)
    finetune_train_graph = dataset.list_graph[:finetune_split_idx]
    finetune_val_graph = dataset.list_graph[finetune_split_idx:]
    if finetune_args["do_pretrain"]:
        eval_loss, eval_corr = do_finetune(
            model, finetune_train_graph, finetune_val_graph, finetune_args, finetune_device
        )
        logging.info(
            f"[Downstream] loss: {eval_loss:.6f}, corr: {eval_corr:.6f} "
        )


def pretrain(
        model,
        pretrain_loader,
        optimizer,
        max_epoch,
        device,
        finetune_device,
        scheduler,
        checkpoint_path,
        logger=None,
        evaluate_step=200,
        finetune_args=None
):
    logging.info("start training..")
    # graph = graph.to(device)
    # edge_index = edge_index.to(device)
    # x = feat.to(device)

    # random.shuffle(node_list)
    # train_val_split = int(len(node_list) * 0.95)
    # train_loader = DataLoader(node_list[:train_val_split], batch_size=1, shuffle=True)
    # val_loader = DataLoader(node_list[train_val_split:], batch_size=1, shuffle=False)

    graph = pretrain_loader.graph.to(device)
    idx_range = list(range(0, pretrain_loader.node_matrix.shape[0]))
    train_loader = DataLoader(idx_range, batch_size=1, shuffle=True)
    # val_loader = DataLoader(dataset.list_graph, batch_size=1, shuffle=False)
    # random.shuffle(list_graph)
    # finetune_split_idx = int(len(list_graph) * 0.9)
    # finetune_train_graph = list_graph[:finetune_split_idx]
    # finetune_val_graph = list_graph[finetune_split_idx:]

    total_step = 0
    train_loss_of_eval = []
    for epoch in range(max_epoch):
        logging.info(f"start epoch {epoch}")
        train_loss_of_epoch = []
        for idx in tqdm(train_loader):
            # edge_index_cre = batch.edge_index_cre.to(device)
            # node_tensor = batch.x.to(device)
            node_tensor = torch.FloatTensor(pretrain_loader.node_matrix[idx].todense()).squeeze().to(device)
            # graph = batch.graph[0].to(device)
            # cre node only
            node_tensor = node_tensor[:(torch.max(pretrain_loader.edge_index_cre).detach().numpy() + 1)].unsqueeze(-1)
            target_nodes = torch.arange(node_tensor.shape[0], device=device, dtype=torch.long)
            model.train()

            loss = model(graph, node_tensor, targets=target_nodes)
            train_loss_of_eval.append(loss.detach().cpu().numpy())
            train_loss_of_epoch.append(loss.detach().cpu().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            if total_step % evaluate_step == 0 and total_step != 0:
                logging.info(f"[Train] epoch {epoch}, total step: {total_step}, loss: {np.mean(train_loss_of_eval)}")
                # train finetune model
                # logging.info(f"begin processing finetune data in train epoch {epoch}, step: {total_step}")
                # eval_loss, eval_corr = do_finetune(
                #     model, finetune_train_graph, finetune_val_graph, finetune_args, finetune_device
                # )
                # logging.info(
                #     f"[Finetune] epoch {epoch}, total step: {total_step}, finetune loss: {eval_loss:.6f}, "
                #     f"finetune corr: {eval_corr:.6f}"
                # )
                # if logger is not None:
                #     logger.note(
                #         {
                #             "step/finetune_loss": eval_loss,
                #             "step/train_loss": np.mean(train_loss_of_eval),
                #             "step/eval_corr": eval_corr
                #         },
                #         step=total_step
                #     )
                train_loss_of_eval = []
            total_step += 1

        # logging.info(f"begin processing finetune data after epoch {epoch}")
        # eval_loss, eval_corr = do_finetune(
        #     model, finetune_train_graph, finetune_val_graph, finetune_args, finetune_device
        # )
        # logging.info(
        #     f"[Train] epoch {epoch}, train loss: {np.mean(train_loss_of_epoch):.6f}, "
        #     f"[Finetune] epoch {epoch}, finetune loss: {eval_loss:.6f}, finetune corr: {eval_corr:.6f}"
        # )
        # if logger is not None:
        #     loss_dict = {
        #         "epoch/train_loss": np.mean(train_loss_of_epoch),
        #         "epoch/val_loss": eval_loss,
        #         "epoch/lr": get_current_lr(optimizer)
        #     }
        #     logger.note(loss_dict, step=epoch)

    torch.save(model, os.path.join(checkpoint_path, "latest_epoch_model.pt"))
    return model


def process_finetune_data(pretrain_model, eval_data_loader, max_gene_node, do_pretrain, device):
    with torch.no_grad():
        pretrain_model.eval()
        eval_data_list = []

        # pretrain_model_copy = copy.deepcopy(pretrain_model)
        # pretrain_model_copy.to(device)

        for val_batch in tqdm(eval_data_loader):
            graph = val_batch.graph.to(device)
            if device > 0:
                val_batch.y_exp.to(device)

            node_tensor = val_batch.x.to(device)
            node_tensor = node_tensor[:(torch.max(val_batch.edge_index_cre).detach().numpy() + 1)].unsqueeze(-1)

            # edge_index_cre = val_batch.edge_index_cre.to(device)
            edge_index = val_batch.edge_index.to(device)
            # all node tensor
            if do_pretrain:
                emb = pretrain_model.embed(graph, node_tensor)
            else:
                emb = node_tensor
            # node tensor selected by gene rel
            emb_select = torch.index_select(emb, 0, edge_index[0].long())
            emb_list_of_gene = []
            mask_list_of_gene = []
            # iterate all gene
            for idx in range(torch.min(edge_index[1]), torch.max(edge_index[1]) + 1):
                # select cre node by gene rel
                idx_of_emb = torch.nonzero(edge_index[1] == idx).squeeze()
                # indefinite cre node embedding selected by gene rel
                emb_select_per_graph = torch.index_select(emb_select, 0, idx_of_emb)

                # padding
                if emb_select_per_graph.size(0) >= max_gene_node:
                    emb_select_per_graph = emb_select_per_graph[:max_gene_node]
                    mask = [1] * emb_select_per_graph.size(0)
                else:
                    emb_select_per_graph_padding = torch.cat(
                        [
                            emb_select_per_graph,
                            torch.zeros(
                                (max_gene_node - emb_select_per_graph.size(0)), emb_select_per_graph.size(1),
                                device=emb_select_per_graph.device
                            )
                        ],
                        dim=0
                    )
                    mask = [1] * emb_select_per_graph.size(0) + [0] * (max_gene_node - emb_select_per_graph.size(0))
                emb_list_of_gene.append(torch.unsqueeze(emb_select_per_graph_padding, dim=0).to(val_batch.y_exp.device))
                mask = torch.tensor(mask, device=val_batch.y_exp.device)
                mask_list_of_gene.append(torch.unsqueeze(mask, dim=0))
            # emb_list_of_gene: 1500, [?, 64]
            emb_list_of_gene = torch.cat(emb_list_of_gene, dim=0)
            emb_list_of_gene.to(device)
            mask_list_of_gene = torch.cat(mask_list_of_gene, dim=0)
            assert emb_list_of_gene.shape[0] == val_batch.y_exp.shape[0]
            eval_data_list.append((emb_list_of_gene, val_batch.y_exp, mask_list_of_gene))
    return FinetuneDataset(eval_data_list)


class FinetuneDataset(Dataset):
    def __init__(self, eval_data_list):
        super().__init__()
        self.eval_data_list = eval_data_list

    def __getitem__(self, index):
        # ?, 1574, [1574]
        return self.eval_data_list[index][0], self.eval_data_list[index][1], self.eval_data_list[index][2]

    def __len__(self):
        return len(self.eval_data_list)


def finetune_model_forward(ft_batch, ft_mask, ft_label, device, finetune_model):
    ft_label = ft_label.to(device)
    ft_batch = ft_batch.to(device)
    ft_mask = ft_mask.to(device)
    gene_output = finetune_model(ft_batch, ft_mask)

    loss = finetune_model.get_loss(gene_output, ft_label)
    corr = pearsonr(
        gene_output.squeeze().detach().cpu(),
        ft_label.detach().cpu()
    )
    # corr = pearsonr(
    #     gene_output.squeeze().detach().cpu().numpy(), ft_label.detach().cpu().numpy()
    # )
    return loss, corr


def get_args_from_dataset(dataset):
    peaks = dataset.array_peak
    mask_numpy = np.array([0 if peak[:3] == 'chr' else 1 for peak in peaks])
    number_gene = int(np.sum(mask_numpy))
    number_node = peaks.shape[0]
    return number_gene, number_node


def train_finetune_model_raw(train_data_loader, eval_data_loader, device, finetune_args):
    from model import train_exp
    from model import MyLossExp
    from model import test_exp
    criterion = MyLossExp()
    model = GAT(
        input_channels=finetune_args["in_dim"],
        hidden_channels=finetune_args["hidden_size"],
        num_head=4,
        num_gene=finetune_args["number_gene"],
        num_nodes=finetune_args["number_node"]
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=finetune_args["lr"], weight_decay=1e-4)
    list_loss = []
    list_train_corr = []
    list_test_corr = []

    for epoch in range(finetune_args["epoch"]):
        loss_t = train_exp(model, criterion, optimizer, device, train_data_loader)
        train_corr = test_exp(model, device, train_data_loader)
        test_corr = test_exp(model, device, eval_data_loader)
        list_loss.append(loss_t)
        list_train_corr.append(train_corr)
        list_test_corr.append(test_corr)
        logging.info(
            f'Epoch: {epoch:03d}, Total loss: {loss_t:.4f}, '
            f'Train Corr: {train_corr:.4f}, Test Corr: {test_corr:.4f}')
    torch.save(
        model, os.path.join(
            finetune_args["ckpt_path"], "finetune_model.pt"
        )
    )


def train_finetune_model(train_data_loader, eval_data_loader, device, finetune_args):
    finetune_model = FinetuneModel(
        pooling_method=finetune_args["pooling_method"],
        projection_method=finetune_args["projection_method"],
        hidden_size=finetune_args["hidden_size"],
        in_dim=finetune_args["in_dim"],
        out_dim=finetune_args["out_dim"],
        loss_fn=finetune_args["loss_fn"],
        device=device
    )
    finetune_model.to(device)
    optimizer = create_optimizer(
        finetune_args["optimizer"],
        finetune_model,
        finetune_args["lr"],
        finetune_args["weight_decay"]
    )
    logging.info(
        f"begin finetune model, finetune args: {finetune_args}, "
        f"train data size: {len(train_data_loader)}, eval data size: {len(eval_data_loader)}"
    )

    finetune_model.train()
    for epoch in range(finetune_args["epoch"]):
        loss_of_epoch = []
        corr_of_epoch = []
        for ft_batch, ft_label, ft_mask in tqdm(train_data_loader):
            loss, corr = finetune_model_forward(ft_batch, ft_mask, ft_label, device, finetune_model)
            loss_of_epoch.append(loss.detach().cpu().numpy())
            corr_of_epoch.extend(corr.squeeze().numpy().tolist())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        logging.info(
            f"[Finetune] epoch {epoch}, loss: {np.mean(loss_of_epoch):.6f}, corr: {np.mean(corr_of_epoch)}"
        )

    with torch.no_grad():
        finetune_model.eval()
        loss_list = []
        corr_list = []
        for ft_batch, ft_label, ft_mask in tqdm(eval_data_loader):
            loss, corr = finetune_model_forward(ft_batch, ft_mask, ft_label, device, finetune_model)
            loss_list.append(loss.detach().cpu().numpy())
            corr_list.extend(corr.squeeze().numpy().tolist())
    return np.mean(loss_list), np.mean(corr_list)


def main(args):
    device = 0 if torch.cuda.is_available() else torch.device("cpu")
    finetune_device = 0 if torch.cuda.is_available() else torch.device("cpu")
    seeds = args.seeds
    # dataset_name = args.dataset
    max_epoch = args.max_epoch
    # max_epoch_f = args.max_epoch_f
    num_hidden = args.num_hidden
    # num_layers = args.num_layers
    # encoder_type = args.encoder
    # decoder_type = args.decoder
    # replace_rate = args.replace_rate

    optim_type = args.optimizer
    # loss_fn = args.loss_fn

    lr = args.lr
    weight_decay = args.weight_decay
    # lr_f = args.lr_f
    # weight_decay_f = args.weight_decay_f
    # linear_prob = args.linear_prob
    load_model = args.load_model
    logs = args.logging
    use_scheduler = args.scheduler

    # graph, (num_features, num_classes) = load_small_dataset(dataset_name)
    # args.data_root_path = "/cpfs01/projects-HDD/cfff-282dafecea22_HDD/public/regFoundation_zylyc/scATAC/test/cortex"
    # node_list, graph = load_graph(args.data_root_path)
    # node_list, edge_index_cre, graph = load_graph_gene(args.data_root_path, args.max_gene_idx)
    # graph_dataset = load_graph_data(args.data_root_path, False, args.edge_prune)
    # num_classes = 0
    # args.num_features = num_hidden if args.do_feat_encoder else graph_dataset.tensor_merge[0].shape[1]
    # args.num_features = args.num_features

    # number_gene, number_node = get_args_from_dataset(graph_dataset)

    loader = PretrainLoader(
        args.data_root_path, args.data_file_name, args.config_path
    )

    finetune_args = {
        "pooling_method": args.f_pooling_method,
        "projection_method": args.f_projection_method,
        "hidden_size": args.f_hidden_size,
        "in_dim": args.num_hidden if args.f_do_pretrain else 1,
        "out_dim": 1,
        "optimizer": args.f_optimizer,
        "lr": args.f_lr,
        "weight_decay": args.f_weight_decay,
        "loss_fn": args.f_loss_fn,
        "epoch": args.f_epoch,
        "max_gene_node": args.f_max_gene_node,
        "do_pretrain": args.f_do_pretrain,
        "number_gene": loader.number_gene,
        "number_node": loader.number_node
    }

    # acc_list = []
    # estp_acc_list = []

    # write args
    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")
    ckpt_path = os.path.join("checkpoints", f"{int(time.time())}")
    os.makedirs(ckpt_path)
    finetune_args.update({"ckpt_path": ckpt_path})
    if args.config_path:
        os.system(f"cp {args.config_path} {ckpt_path}")
    else:
        with open(os.path.join(ckpt_path, "args.pkl"), "wb") as f:
            pickle.dump(args, f)

    for i, seed in enumerate(seeds):
        logging.info(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)

        if logs:
            logger = TBLogger(
                log_path=ckpt_path,
                name="tensorboard")
        else:
            logger = None

        model = build_model(args)
        model.to(device)
        optimizer = create_optimizer(optim_type, model, lr, weight_decay)

        if use_scheduler:
            logging.info("Use schedular")
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda epoch: (1 + np.cos(epoch * np.pi / max_epoch)) * 0.5
            )
        else:
            scheduler = None

        if not load_model:
            # if finetune_args["do_pretrain"]:
            model = pretrain(
                model,
                loader,
                optimizer,
                max_epoch,
                device,
                finetune_device,
                scheduler,
                ckpt_path,
                logger,
                args.evaluate_step,
                finetune_args
            )
            # else:
            #     downstream(
            #         graph_dataset,
            #         model,
            #         finetune_args,
            #         finetune_device
            #     )
            model = model.cpu()

        if load_model:
            logging.info("Loading Model ... ")
            model.load_state_dict(torch.load("checkpoint.pt"))

        model = model.to(device)
        model.eval()

        # final_acc, estp_acc = linear_probing_full_batch(model, graph, x, num_classes, lr_f, weight_decay_f, max_epoch_f,
        #                                                 device, linear_prob)
        # acc_list.append(final_acc)
        # estp_acc_list.append(estp_acc)

        if logger is not None:
            logger.finish()

    # final_acc, final_acc_std = np.mean(acc_list), np.std(acc_list)
    # estp_acc, estp_acc_std = np.mean(estp_acc_list), np.std(estp_acc_list)
    # print(f"# final_acc: {final_acc:.4f}±{final_acc_std:.4f}")
    # print(f"# early-stopping_acc: {estp_acc:.4f}±{estp_acc_std:.4f}")


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    # args_ = build_args()
    # if args_.use_cfg:
    #     args_ = load_best_configs(args_)
    # print(args_)
    # main(args_)

    args_ = build_yaml_args()
    load_yaml_conf(args_)
    print(args_)
    main(args_)
    # root_path = "/cpfs01/projects-HDD/cfff-282dafecea22_HDD/public/regFoundation_zylyc/scATAC/test/cortex"
    # load_graph_data(root_path)
