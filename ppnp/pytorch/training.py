from typing import Type
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..data.simple.sparsegraph import SparseGraph
from ..preprocessing import gen_seeds, gen_splits,gen_splits2, normalize_attributes
from .earlystopping import EarlyStopping, stopping_args
from .utils import matrix_to_torch, sparse_matrix_to_torch, get_dataloaders, get_predictions, SparseMM
from sklearn.metrics import f1_score
from .propagation import process_graph
import os
import numpy as np
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model_AdaGCN(
        name: str, model_name: str, model_class: Type[nn.Module], graph: SparseGraph, model_args: dict, reg_lambda: float,
        idx_split_args: dict = {'ntrain_per_class': 20, 'nstopping': 500, 'nknown': 1500, 'seed': 2413340114},
        stopping_args: dict = stopping_args,
        test: bool = False, device: str = 'cuda',
        torch_seed: int = None, print_interval: int = 10, early: bool = True,  split_idx = None) -> nn.Module:
    # <editor-fold desc="(1) set up logging">
    print('--------------------------------------------------train', model_name, 'on', name,
          '---------------------------------------------------')
    labels_all = graph.labels.astype(np.int)
    idx_np = {}
    # (1) split labels to train, stopping, val/test
    # idx_np['train'], idx_np['stopping'], idx_np['valtest']  = gen_splits2(split_idx, labels_all)
    idx_np['train'], idx_np['stopping'], idx_np['valtest'] = gen_splits(labels_all, idx_split_args, test=test)
    idx_all = {key: torch.LongTensor(val) for key, val in idx_np.items()}

    logging.log(21, f"{model_class.__name__}: {model_args}")
    if torch_seed is None:
        torch_seed = gen_seeds()
    torch.manual_seed(seed=torch_seed)  # random
    logging.log(22, f"PyTorch seed: {torch_seed}")

    nfeatures = graph.attr_matrix.shape[1]  # cora: 2879
    nclasses = max(labels_all) + 1  # cora: 7
    # define model
    print('device:', device)
    model = model_class(nfeat=nfeatures,
                        nhid=model_args['hid_AdaGCN'],
                        nclass=nclasses,
                        dropout=model_args['drop_prob'],
                        dropout_adj=model_args['dropoutadj_AdaGCN']).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=model_args['lr'], weight_decay=model_args['weight_decay'])

    # (2) optimizer, dataloader, early_stopping
    dataloaders = get_dataloaders(idx_all, labels_all)  # random index to dataloaders

    # nomalize features and then to tensor
    attr_mat_norm_np = normalize_attributes(graph.attr_matrix)
    attr_mat_norm = torch.FloatTensor(np.array(attr_mat_norm_np.todense())).to(device)  # DenseTensor

    # %%(3) define variables:
    if early:
        epoch_stats = {'train': {}, 'stopping': {}}
    else:
        epoch_stats = {'train': {}}
    start_time = time.time()
    last_time = start_time

    # %%load data
    graph_processor = process_graph(graph.adj_matrix)
    graph_processor.features = attr_mat_norm.cpu().numpy()
    adj = sparse_matrix_to_torch(graph_processor.calc_A_hat()).cuda()  # caculate normalized H
    hyperedge_index = graph.hyperedge_index
    temp_feature = graph_processor.features
    simple_adj = graph_processor.calc_hgcn_adj(adj.shape[0], hyperedge_index, temp_feature ,False).to_dense()
    simple_adj = torch.FloatTensor(simple_adj).to(device)

    # %%initial sample_weights
    sample_weights = torch.ones(graph.adj_matrix.shape[0])
    sample_weights = sample_weights[idx_all['train']]
    sample_weights = sample_weights / sample_weights.sum()
    sample_weights = sample_weights.to(device)

    # %%initial results saver
    results = torch.zeros(graph.adj_matrix.shape[0], nclasses).to(device)
    vali_weight = torch.zeros(model_args['layers'], 1).to(device)
    ALL_epochs = 0

    # %%(4) train HyperAdaGCN

    features = attr_mat_norm.clone()
    simple_features = attr_mat_norm.clone()

    for layer in range(model_args['layers']):  # (First contribution) aggregated different hop H i an AdaBoost way
        early_stopping = EarlyStopping(model, **stopping_args)
        new_adj = adj.to_dense()
        features = SparseMM.apply(new_adj, features).detach()  # hypergraph information propagation (ith layer trained by ith hop)
        features = attr_mat_norm + features  # add res
        simple_features = SparseMM.apply(simple_adj, simple_features).detach()  # simplegraph information propagation (ith layer trained by ith hop)
        simple_features = attr_mat_norm + simple_features

        model.simple_features = simple_features
        logging.info(f"|This is the {layer + 1}th layer!")
        # (5) train each classifier:  each epoch: training + early stopping
        if layer == 0:  #becasue the next layer inherit the prameters of perious layer, hence only the first layer need more iteration
            num_epoch = early_stopping.max_epochs
        else:
            num_epoch = 200

        for epoch in range(num_epoch):  # 10000
            for phase in epoch_stats.keys():  # 2 phases: train, stopping, train 1 epoch, evaluate on stopping dataset
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0
                running_corrects = 0
                error_rate = 0
                i = 1

                for idx, labels in dataloaders[phase]:  # training set / early stopping set
                    idx = idx.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):  # train: True
                        log_preds = model(features, idx)
                        loss = F.nll_loss(log_preds, labels, reduction='none')  # each loss
                        # core 1: weighted loss
                        if phase == 'train':
                            loss = loss * sample_weights
                        preds = torch.argmax(log_preds, dim=1)
                        loss = loss.sum()
                        l2_reg = sum((torch.sum(param ** 2) for param in model.reg_params))
                        loss = loss + reg_lambda / 2 * l2_reg  # cross loss + L2 regularization
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                        # Collect statistics
                        running_loss += loss.item()
                        running_corrects += torch.sum(preds == labels)
                        error_rate += (torch.sum(preds == labels)) / len(labels)
                        i += 1
                error_rate = error_rate / i
                # Collect statistics (current epoch)
                epoch_stats[phase]['loss'] = running_loss / len(dataloaders[phase].dataset)
                epoch_stats[phase]['acc'] = running_corrects.item() / len(dataloaders[phase].dataset)

            # print logging each interval
            if epoch % print_interval == 0:
                duration = time.time() - last_time  # each interval including training and early-stopping
                last_time = time.time()
                if early:
                    logging.info(f"Epoch {epoch}: "
                                 f"Train loss = {epoch_stats['train']['loss']:.2f}, "
                                 f"train acc = {epoch_stats['train']['acc'] * 100:.1f}, "
                                 f"early stopping loss = {epoch_stats['stopping']['loss']:.2f}, "
                                 f"early stopping acc = {epoch_stats['stopping']['acc'] * 100:.1f} "
                                 f"({duration:.3f} sec)")
                else:
                    logging.info(f"Epoch {epoch}: "
                                 f"Train loss = {epoch_stats['train']['loss']:.2f}, "
                                 f"train acc = {epoch_stats['train']['acc'] * 100:.1f}, "
                                 f"({duration:.3f} sec)")
            # (4) check whether it stops on some epoch
            if early:
                if len(early_stopping.stop_vars) > 0:
                    stop_vars = [epoch_stats['stopping'][key] for key in early_stopping.stop_vars]  # 'acc', 'loss'
                    if early_stopping.check(stop_vars, epoch):  # whether exist improvement for patience times
                        break
        # (6) SAMME.R
        ALL_epochs += epoch
        runtime = time.time() - start_time
        logging.log(22,
                    f"Last epoch: {epoch}, best epoch: {early_stopping.best_epoch}, best vals:{early_stopping.best_vals[0]} ({runtime:.3f} sec)")
        # Load best model weights
        if early:
            model.load_state_dict(early_stopping.best_state)
            print('best w1 :', model.w1)
        model.eval()
        # %% obtain predictions in multi-classification AdaBoost way
        output = model(features, torch.arange(graph.adj_matrix.shape[0])).detach()
        output_logp = torch.log(F.softmax(output, dim=1))
        h = (nclasses - 1) * (output_logp - torch.mean(output_logp, dim=1).view(-1, 1))

        results += h * early_stopping.best_vals[0]  # weighted voting
        # %%
        vali_weight[layer] = early_stopping.best_vals[0]

        # adjust weights
        temp = F.nll_loss(output_logp[idx_all['train']],
                          torch.LongTensor(labels_all[idx_all['train']].astype(np.int32)).to(device),
                          reduction='none')  # 140*1
        weight = sample_weights * torch.exp(((- (nclasses - 1)) / (nclasses)) * temp)  # update weights
        weight = weight / weight.sum()
        sample_weights = weight.detach()
        simple_adj = graph_processor.calc_hgcn_adj(adj.shape[0], hyperedge_index, simple_features.cpu().numpy() ,False).to_dense()
        a = simple_adj.nonzero()
        simple_adj = torch.FloatTensor(simple_adj).to(device)
    # <editor-fold desc=" # (5) evaluate the best model from early stopping ">
    runtime = time.time() - start_time

    stopping_preds = torch.argmax(results[idx_all['stopping']], dim=1).cpu().numpy()
    stopping_acc = (stopping_preds == labels_all[idx_all['stopping']]).mean()
    stopping_f1 = f1_score(stopping_preds, labels_all[idx_all['stopping']], average='micro')
    logging.log(21, f"Early stopping accuracy: {stopping_acc * 100:.1f}%")

    valtest_preds = torch.argmax(results[idx_all['valtest']], dim=1).cpu().numpy()
    valtest_acc = (valtest_preds == labels_all[idx_all['valtest']]).mean()
    valtest_f1 = f1_score(valtest_preds, labels_all[idx_all['valtest']], average='micro')
    valtest_name = 'Test' if test else 'Validation'
    logging.log(22, f"{valtest_name} accuracy: {valtest_acc * 100:.1f}%")

    # (6) return result
    result = {}
    result['early_stopping'] = {'accuracy': stopping_acc, 'f1_score': stopping_f1}
    result['valtest'] = {'accuracy': valtest_acc, 'f1_score': valtest_f1}
    result['runtime'] = runtime
    result['runtime_perepoch'] = runtime / (ALL_epochs + 1)
    # </editor-fold>
    return result

