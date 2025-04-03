import yaml
import scipy
import anndata as ad
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_sparse import transpose
from torch_geometric.loader import DataLoader
from torch_geometric.utils import subgraph, is_undirected
from ogb.graphproppred import Evaluator
from sklearn.metrics import roc_auc_score
from rdkit import Chem

from utils import Writer, Criterion, MLP, visualize_a_graph, save_checkpoint, load_checkpoint, get_preds, get_lr, \
    set_seed, process_data
from utils import get_local_config_name, get_model, get_data_loaders, write_stat_from_metric_dicts, reorder_like, \
    init_metric_dict


class GE(nn.Module):

    def __init__(self, clf, extractor, optimizer, scheduler, writer, device, model_dir, dataset_name, num_class,
                 multi_label, random_state,
                 method_config, shared_config):
        super().__init__()
        self.clf = clf
        self.extractor = extractor
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.writer = writer
        self.device = device
        self.model_dir = model_dir
        self.dataset_name = dataset_name
        self.random_state = random_state
        self.method_name = method_config['method_name']

        self.learn_edge_att = shared_config['learn_edge_att']
        self.k = shared_config['precision_k']
        self.num_viz_samples = shared_config['num_viz_samples']
        self.viz_interval = shared_config['viz_interval']
        self.viz_norm_att = shared_config['viz_norm_att']

        self.epochs = method_config['epochs']
        self.pred_loss_coef = method_config['pred_loss_coef']
        self.info_loss_coef = method_config['info_loss_coef']

        self.fix_r = method_config.get('fix_r', None)
        self.decay_interval = method_config.get('decay_interval', None)
        self.decay_r = method_config.get('decay_r', None)
        self.final_r = method_config.get('final_r', 0.1)
        self.init_r = method_config.get('init_r', 0.9)

        self.multi_label = multi_label
        self.criterion = Criterion(num_class, multi_label)

    def __loss__(self, att, clf_logits, clf_labels, epoch):
        pred_loss = self.criterion(clf_logits, clf_labels)

        r = self.fix_r if self.fix_r else self.get_r(self.decay_interval, self.decay_r, epoch, final_r=self.final_r,
                                                     init_r=self.init_r)
        info_loss = (att * torch.log(att / r + 1e-6) + (1 - att) * torch.log((1 - att) / (1 - r + 1e-6) + 1e-6)).mean()

        pred_loss = pred_loss * self.pred_loss_coef
        info_loss = info_loss * self.info_loss_coef
        loss = pred_loss + info_loss
        loss_dict = {'loss': loss.item(), 'pred': pred_loss.item(), 'info': info_loss.item()}
        return loss, loss_dict

    def forward_pass(self, data, epoch, training):
        emb = self.clf.get_emb(data.x, data.edge_index, batch=data.batch, edge_attr=data.edge_attr)
        att_log_logits = self.extractor(emb, data.edge_index, data.batch)
        att = self.sampling(att_log_logits, epoch, training)

        if self.learn_edge_att:
            if is_undirected(data.edge_index):
                trans_idx, trans_val = transpose(data.edge_index, att, None, None,
                                                 coalesced=False)
                trans_val_perm = reorder_like(trans_idx, data.edge_index, trans_val)
                edge_att = (att + trans_val_perm) / 2
            else:
                edge_att = att
        else:
            edge_att = self.lift_node_att_to_edge_att(att, data.edge_index)

        clf_logits = self.clf(data.x, data.edge_index, data.batch, edge_attr=data.edge_attr, edge_atten=edge_att)
        loss, loss_dict = self.__loss__(att, clf_logits, data.y, epoch)
        return edge_att, loss, loss_dict, clf_logits

    @torch.no_grad()
    def eval_one_batch(self, data, epoch):
        self.extractor.eval()
        self.clf.eval()

        att, loss, loss_dict, clf_logits = self.forward_pass(data, epoch, training=False)
        return att.data.cpu().reshape(-1), loss_dict, clf_logits.data.cpu()

    def train_one_batch(self, data, epoch):
        self.extractor.train()
        self.clf.train()

        att, loss, loss_dict, clf_logits = self.forward_pass(data, epoch, training=True)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return att.data.cpu().reshape(-1), loss_dict, clf_logits.data.cpu()

    def run_one_epoch(self, data_loader, epoch, phase, use_edge_attr):
        loader_len = len(data_loader)

        run_one_batch = self.train_one_batch if phase == 'train' else self.eval_one_batch
        phase = 'test ' if phase == 'test' else phase  # align tqdm desc bar

        all_loss_dict = {}
        all_exp_labels, all_att, all_clf_labels, all_clf_logits, all_precision_at_k = ([] for i in range(5))

        size_info = data_loader.dataset.size_info
        subgraph_size = size_info.get('subgraph_size')
        gene_length = size_info.get('gene_length')
        subgraph_node_label = size_info.get('subgraph_node_label')

        normal_size = subgraph_size
        cancer_size = subgraph_size

        matrix_list = [
            np.zeros((normal_size, normal_size)),  # total_normal_matrix
            np.zeros((cancer_size, cancer_size))  # total_cancer_matrix
        ]

        gene_expression_list = [
            np.zeros((normal_size, gene_length)),  # total_normal_gene_expression_matrix
            np.zeros((cancer_size, gene_length))  # total_cancer_gene_expression_matrix
        ]

        count_list = [0, 0]  # normal_count, cancer_count

        pbar = tqdm(data_loader)
        for idx, data in enumerate(pbar):
            subgraph_enconode_label = data[0].node_label.cpu().numpy()
            data = process_data(data, use_edge_attr)
            att, loss_dict, clf_logits = run_one_batch(data.to(self.device), epoch)

            exp_labels = data.edge_label.data.cpu()
            precision_at_k = self.get_precision_at_k(att, exp_labels, self.k, data.batch,
                                                     data.edge_index)

            desc, _, _, _, _, _, = self.log_epoch(epoch, phase, loss_dict, exp_labels, att, precision_at_k,
                                                  data.y.data.cpu(), clf_logits, data.edge_index.cpu(),
                                                  data.batch.cpu(), data.x.cpu(), matrix_list,
                                                  gene_expression_list, count_list, batch=True)

            for k, v in loss_dict.items():
                all_loss_dict[k] = all_loss_dict.get(k, 0) + v

            all_exp_labels.append(exp_labels)
            all_att.append(att)
            all_precision_at_k.extend(precision_at_k)
            all_clf_labels.append(data.y.data.cpu())
            all_clf_logits.append(clf_logits)

            if idx == loader_len - 1:
                all_exp_labels, all_att = torch.cat(all_exp_labels), torch.cat(all_att)
                all_clf_labels, all_clf_logits = torch.cat(all_clf_labels), torch.cat(all_clf_logits)

                for k, v in all_loss_dict.items():
                    all_loss_dict[k] = v / loader_len

                desc, att_auroc, precision, clf_acc, clf_roc, avg_loss = self.log_epoch(epoch, phase, all_loss_dict,
                                                                                        all_exp_labels, all_att,
                                                                                        all_precision_at_k,
                                                                                        all_clf_labels, all_clf_logits,
                                                                                        data.edge_index.cpu(),
                                                                                        data.batch.cpu(), None, None,
                                                                                        None, None, batch=False)
            pbar.set_description(desc)

        normal_matrix_avg = np.divide(matrix_list[0], count_list[0], where=count_list[0] != 0)
        cancer_matrix_avg = np.divide(matrix_list[1], count_list[1], where=count_list[1] != 0)
        normal_matrix_avg_df = pd.DataFrame(normal_matrix_avg, index=subgraph_enconode_label,
                                            columns=subgraph_enconode_label)
        cancer_matrix_avg_df = pd.DataFrame(cancer_matrix_avg, index=subgraph_enconode_label,
                                            columns=subgraph_enconode_label)

        normal_gene_expression_matrix_avg = np.divide(gene_expression_list[0], count_list[0], where=count_list[0] != 0)
        cancer_gene_expression_matrix_avg = np.divide(gene_expression_list[1], count_list[1], where=count_list[1] != 0)

        return att_auroc, precision, clf_acc, clf_roc, avg_loss, normal_matrix_avg_df, cancer_matrix_avg_df, normal_gene_expression_matrix_avg, cancer_gene_expression_matrix_avg,

    def train(self, loaders, test_set, metric_dict, use_edge_attr):
        viz_set = self.get_viz_idx(test_set, self.dataset_name)
        for epoch in range(self.epochs):
            train_res = self.run_one_epoch(loaders['train'], epoch, 'train', use_edge_attr)
            valid_res = self.run_one_epoch(loaders['valid'], epoch, 'valid', use_edge_attr)
            test_res = self.run_one_epoch(loaders['test'], epoch, 'test', use_edge_attr)
            self.writer.add_scalar('ge_train/lr', get_lr(self.optimizer), epoch)

            assert len(train_res) == 9
            main_metric_idx = 3 if self.dataset_name == 'sc' else 2  # 3 clf_roc or 2 clf_acc
            if self.scheduler is not None:
                self.scheduler.step(valid_res[main_metric_idx])

            r = self.fix_r if self.fix_r else self.get_r(self.decay_interval, self.decay_r, epoch, final_r=self.final_r,
                                                         init_r=self.init_r)
            if (r == self.final_r or self.fix_r) and epoch > 10 and (
                    (valid_res[main_metric_idx] > metric_dict['metric/best_clf_valid'])
                    or (valid_res[main_metric_idx] == metric_dict['metric/best_clf_valid']
                        and valid_res[4] < metric_dict['metric/best_clf_valid_loss'])):

                metric_dict = {'metric/best_clf_epoch': epoch,
                               'metric/best_clf_valid_loss': valid_res[4],
                               'metric/best_clf_train': train_res[main_metric_idx],
                               'metric/best_clf_valid': valid_res[main_metric_idx],
                               'metric/best_clf_test': test_res[main_metric_idx],
                               'metric/best_x_roc_train': train_res[0],
                               'metric/best_x_roc_valid': valid_res[0],
                               'metric/best_x_roc_test': test_res[0],
                               'metric/best_x_precision_train': train_res[1],
                               'metric/best_x_precision_valid': valid_res[1],
                               'metric/best_x_precision_test': test_res[1]}

                save_checkpoint(self.clf, self.model_dir, model_name='ge_clf_epoch_' + str(epoch))
                save_checkpoint(self.extractor, self.model_dir, model_name='ge_att_epoch_' + str(epoch))

                if self.dataset_name == 'sc':
                    best_edge_weights_normal = self.model_dir / f'best_edge_weights_normal_epoch_{epoch}.csv'
                    best_edge_weights_cancer = self.model_dir / f'best_edge_weights_cancer_epoch_{epoch}.csv'

                    test_res[5].to_csv(best_edge_weights_normal, index=True, header=True)
                    test_res[6].to_csv(best_edge_weights_cancer, index=True, header=True)

                    best_gene_expression_normal = self.model_dir / f'best_gene_expression_normal_epoch_{epoch}.h5ad'
                    best_gene_expression_cancer = self.model_dir / f'best_gene_expression_cancer_epoch_{epoch}.h5ad'

                    gene_expression_normal_adata = ad.AnnData(test_res[7])
                    gene_expression_cancer_adata = ad.AnnData(test_res[8])

                    gene_expression_normal_adata.write_h5ad(best_gene_expression_normal)
                    gene_expression_cancer_adata.write_h5ad(best_gene_expression_cancer)

                    print(f"The relationship matrix of the best model has been saved as {best_edge_weights_normal} and {best_edge_weights_cancer}")
                    print(
                        f"The gene expression matrix of the best model has been saved as {best_gene_expression_normal} and{best_gene_expression_cancer}")

            for metric, value in metric_dict.items():
                metric = metric.split('/')[-1]
                self.writer.add_scalar(f'ge_best/{metric}', value, epoch)

            if self.num_viz_samples != 0 and (epoch % self.viz_interval == 0 or epoch == self.epochs - 1):
                if self.multi_label:
                    raise NotImplementedError
                for idx, tag in viz_set:
                    self.visualize_results(test_set, idx, epoch, tag, use_edge_attr)

            if epoch == self.epochs - 1:
                save_checkpoint(self.clf, self.model_dir, model_name='ge_clf_epoch_' + str(epoch))
                save_checkpoint(self.extractor, self.model_dir, model_name='ge_att_epoch_' + str(epoch))

                if self.dataset_name == 'sc':
                    edge_weights_normal = self.model_dir / f'edge_weights_normal_epoch_{epoch}.csv'
                    edge_weights_cancer = self.model_dir / f'edge_weights_cancer_epoch_{epoch}.csv'

                    test_res[5].to_csv(edge_weights_normal, index=True, header=True)  # index=True 保留行名，header=True 保留列名
                    test_res[6].to_csv(edge_weights_cancer, index=True, header=True)

                    gene_expression_normal = self.model_dir / f'gene_expression_normal_epoch_{epoch}.h5ad'
                    gene_expression_cancer = self.model_dir / f'gene_expression_cancer_epoch_{epoch}.h5ad'

                    gene_expression_normal_adata = ad.AnnData(test_res[7])
                    gene_expression_cancer_adata = ad.AnnData(test_res[8])

                    gene_expression_normal_adata.write_h5ad(gene_expression_normal)
                    gene_expression_cancer_adata.write_h5ad(gene_expression_cancer)

                    print(f"The relationship matrix of the model has been saved as {edge_weights_normal} and {edge_weights_cancer}")
                    print(f"The gene expression matrix of the model has been saved as {gene_expression_normal} and {gene_expression_cancer}")

            print(f'[Seed {self.random_state}, Epoch: {epoch}]: Best Epoch: {metric_dict["metric/best_clf_epoch"]}, '
                  f'Best Val Pred ACC/ROC: {metric_dict["metric/best_clf_valid"]:.3f}, Best Test Pred ACC/ROC: {metric_dict["metric/best_clf_test"]:.3f}, '
                  f'Best Test X AUROC: {metric_dict["metric/best_x_roc_test"]:.3f}')
            print('====================================')
            print('====================================')
        return metric_dict

    def log_epoch(self, epoch, phase, loss_dict, exp_labels, att, precision_at_k, clf_labels, clf_logits, edge_index,
                  sub_gra, sub_gra_data, matrix_list, gene_expression_list, count_list, batch):

        desc = f'[Seed {self.random_state}, Epoch: {epoch}]: ge_{phase}........., ' if batch else f'[Seed {self.random_state}, Epoch: {epoch}]: ge_{phase} finished, '

        for k, v in loss_dict.items():
            if not batch:
                self.writer.add_scalar(f'ge_{phase}/{k}', v, epoch)
            desc += f'{k}: {v:.3f}, '

        eval_desc, att_auroc, precision, clf_acc, clf_roc = self.get_eval_score(epoch, phase, exp_labels, att,
                                                                                precision_at_k, clf_labels, clf_logits,
                                                                                edge_index, sub_gra, sub_gra_data,
                                                                                matrix_list, gene_expression_list,
                                                                                count_list, batch
                                                                                )
        desc += eval_desc

        return desc, att_auroc, precision, clf_acc, clf_roc, loss_dict['pred']

    def get_eval_score(self, epoch, phase, exp_labels, att, precision_at_k, clf_labels, clf_logits, edge_index,
                       sub_gra, sub_gra_data, matrix_list, gene_expression_list, count_list, batch):

        clf_preds = get_preds(clf_logits, self.multi_label)
        clf_acc = 0 if self.multi_label else (clf_preds == clf_labels).sum().item() / clf_labels.shape[0]

        if batch:
            if self.dataset_name == 'sc' and phase == 'test ':
                for i in range(sub_gra.max().item() + 1):
                    nodes_for_graph_i = torch.where(sub_gra == i)[0]
                    global_to_local_idx_map = {global_idx.item(): local_idx for local_idx, global_idx in
                                               enumerate(nodes_for_graph_i)}
                    gene_expression_for_graph_i = sub_gra_data[nodes_for_graph_i]

                    if clf_preds[i] == 1:  #  normal (1)
                        gene_expression_list[0][
                        :len(nodes_for_graph_i)] += gene_expression_for_graph_i.cpu().numpy()
                        count_list[0] += 1
                    elif clf_preds[i] == 0:  #  cancer (0)
                        gene_expression_list[1][
                        :len(nodes_for_graph_i)] += gene_expression_for_graph_i.cpu().numpy()
                        count_list[1] += 1

                    source_nodes = edge_index[0]
                    target_nodes = edge_index[1]

                    edges_for_graph_i = (torch.isin(source_nodes, nodes_for_graph_i)) & (
                        torch.isin(target_nodes, nodes_for_graph_i))
                    edge_pairs = edge_index[:, edges_for_graph_i].T

                    att_for_graph_i = att[edges_for_graph_i]

                    for j, (source_global, target_global) in enumerate(edge_pairs):
                        source_local = global_to_local_idx_map[source_global.item()]
                        target_local = global_to_local_idx_map[target_global.item()]

                        if source_local <= target_local:
                            if clf_preds[i] == 1:  #  normal
                                matrix_list[0][source_local, target_local] += att_for_graph_i[j].item()
                            elif clf_preds[i] == 0:  #  cancer
                                matrix_list[1][source_local, target_local] += att_for_graph_i[j].item()

            return f'clf_acc: {clf_acc:.3f}', None, None, None, None

        precision_at_k = np.mean(precision_at_k)
        clf_roc = 0
        if self.dataset_name == 'sc':
            clf_roc = roc_auc_score(clf_labels, clf_logits)

        att_auroc, bkg_att_weights, signal_att_weights = 0, att, att
        if np.unique(exp_labels).shape[0] > 1:
            att_auroc = roc_auc_score(exp_labels, att)
            bkg_att_weights = att[exp_labels == 0]
            signal_att_weights = att[exp_labels == 1]

        self.writer.add_histogram(f'ge_{phase}/bkg_att_weights', bkg_att_weights, epoch)
        self.writer.add_histogram(f'ge_{phase}/signal_att_weights', signal_att_weights, epoch)
        self.writer.add_scalar(f'ge_{phase}/clf_acc/', clf_acc, epoch)
        self.writer.add_scalar(f'ge_{phase}/clf_roc/', clf_roc, epoch)
        self.writer.add_scalar(f'ge_{phase}/att_auroc/', att_auroc, epoch)
        self.writer.add_scalar(f'ge_{phase}/precision@{self.k}/', precision_at_k, epoch)
        self.writer.add_scalar(f'ge_{phase}/avg_bkg_att_weights/', bkg_att_weights.mean(), epoch)
        self.writer.add_scalar(f'ge_{phase}/avg_signal_att_weights/', signal_att_weights.mean(), epoch)
        self.writer.add_pr_curve(f'PR_Curve/ge_{phase}/', exp_labels, att, epoch)

        desc = f'clf_acc: {clf_acc:.3f}, clf_roc: {clf_roc:.3f}, ' + \
               f'att_roc: {att_auroc:.3f}, att_prec@{self.k}: {precision_at_k:.3f}'
        return desc, att_auroc, precision_at_k, clf_acc, clf_roc

    def get_precision_at_k(self, att, exp_labels, k, batch, edge_index):
        precision_at_k = []
        for i in range(batch.max() + 1):
            nodes_for_graph_i = batch == i
            edges_for_graph_i = nodes_for_graph_i[edge_index[0]] & nodes_for_graph_i[edge_index[1]]
            labels_for_graph_i = exp_labels[edges_for_graph_i]
            mask_log_logits_for_graph_i = att[edges_for_graph_i]
            precision_at_k.append(labels_for_graph_i[np.argsort(-mask_log_logits_for_graph_i)[:k]].sum().item() / k)
        return precision_at_k

    def get_viz_idx(self, test_set, dataset_name):
        y_dist = test_set.data.y.numpy().reshape(-1)
        num_nodes = np.array([each.x.shape[0] for each in test_set])
        classes = np.unique(y_dist)
        res = []
        for each_class in classes:
            tag = 'class_' + str(each_class)
            candidate_set = np.nonzero(y_dist == each_class)[0]
            idx = np.random.choice(candidate_set, self.num_viz_samples, replace=False)
            res.append((idx, tag))
        return res

    def visualize_results(self, test_set, idx, epoch, tag, use_edge_attr):
        viz_set = test_set[idx]
        data = next(iter(DataLoader(viz_set, batch_size=len(idx), shuffle=False)))
        data = process_data(data, use_edge_attr)
        batch_att, _, clf_logits = self.eval_one_batch(data.to(self.device), epoch)
        imgs = []
        for i in tqdm(range(len(viz_set))):
            mol_type, coor = None, None

            node_subset = data.batch == i
            _, edge_att = subgraph(node_subset, data.edge_index, edge_attr=batch_att)

            node_label = viz_set[i].node_label.reshape(-1) if viz_set[i].get('node_label',
                                                                             None) is not None else torch.zeros(
                viz_set[i].x.shape[0])
            fig, img = visualize_a_graph(viz_set[i].edge_index, edge_att, node_label, self.dataset_name,
                                         norm=self.viz_norm_att, mol_type=mol_type, coor=coor)
            imgs.append(img)
        imgs = np.stack(imgs)
        self.writer.add_images(tag, imgs, epoch, dataformats='NHWC')

    def get_r(self, decay_interval, decay_r, current_epoch, init_r=0.9, final_r=0.5):
        r = init_r - current_epoch // decay_interval * decay_r
        if r < final_r:
            r = final_r
        return r

    def sampling(self, att_log_logits, epoch, training):
        att = self.concrete_sample(att_log_logits, temp=1, training=training)
        return att

    @staticmethod
    def lift_node_att_to_edge_att(node_att, edge_index):
        src_lifted_att = node_att[edge_index[0]]
        dst_lifted_att = node_att[edge_index[1]]
        edge_att = src_lifted_att * dst_lifted_att
        return edge_att

    @staticmethod
    def concrete_sample(att_log_logit, temp, training):
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = (att_log_logit).sigmoid()
        return att_bern


class ExtractorMLP(nn.Module):

    def __init__(self, hidden_size, shared_config):
        super().__init__()
        self.learn_edge_att = shared_config['learn_edge_att']
        dropout_p = shared_config['extractor_dropout_p']

        if self.learn_edge_att:
            self.feature_extractor = MLP([hidden_size * 2, hidden_size * 4, hidden_size, 1], dropout=dropout_p)
        else:
            self.feature_extractor = MLP([hidden_size * 1, hidden_size * 2, hidden_size, 1], dropout=dropout_p)

    def forward(self, emb, edge_index, batch):
        if self.learn_edge_att:
            col, row = edge_index
            f1, f2 = emb[col], emb[row]
            f12 = torch.cat([f1, f2], dim=-1)
            att_log_logits = self.feature_extractor(f12, batch[col])
        else:
            att_log_logits = self.feature_extractor(emb, batch)
        return att_log_logits


def train_ge_one_seed(local_config, data_dir, log_dir, model_name, dataset_name, method_name, device, random_state):
    print('====================================')
    print('====================================')
    print(f'[INFO] Using device: {device}')
    print(f'[INFO] Using random_state: {random_state}')
    print(f'[INFO] Using dataset: {dataset_name}')
    print(f'[INFO] Using model: {model_name}')

    set_seed(random_state)

    model_config = local_config['model_config']
    data_config = local_config['data_config']
    method_config = local_config[f'{method_name}_config']
    shared_config = local_config['shared_config']
    assert model_config['model_name'] == model_name
    assert method_config['method_name'] == method_name

    batch_size, splits = data_config['batch_size'], data_config.get('splits', None)
    loaders, test_set, x_dim, edge_attr_dim, num_class, aux_info = get_data_loaders(data_dir, dataset_name, batch_size,
                                                                                    splits, random_state)

    model_config['deg'] = aux_info['deg']
    model = get_model(x_dim, edge_attr_dim, num_class, aux_info['multi_label'], model_config, device)
    print('====================================')
    print('====================================')

    log_dir.mkdir(parents=True, exist_ok=True)
    print('[INFO] Training both the model and the attention from scratch...')

    extractor = ExtractorMLP(model_config['hidden_size'], shared_config).to(device)
    lr, wd = method_config['lr'], method_config.get('weight_decay', 0)
    optimizer = torch.optim.Adam(list(extractor.parameters()) + list(model.parameters()), lr=lr, weight_decay=wd)

    scheduler_config = method_config.get('scheduler', {})
    scheduler = None if scheduler_config == {} else ReduceLROnPlateau(optimizer, mode='max', **scheduler_config)

    writer = Writer(log_dir=log_dir)
    hparam_dict = {**model_config, **data_config}
    hparam_dict = {k: str(v) if isinstance(v, (dict, list)) else v for k, v in hparam_dict.items()}
    metric_dict = deepcopy(init_metric_dict)
    writer.add_hparams(hparam_dict=hparam_dict, metric_dict=metric_dict)

    print('====================================')
    print('[INFO] Training GE...')
    #
    ge = GE(model, extractor, optimizer, scheduler, writer, device, log_dir, dataset_name, num_class,
            aux_info['multi_label'], random_state, method_config, shared_config)
    metric_dict = ge.train(loaders, test_set, metric_dict, model_config.get('use_edge_attr', True))
    writer.add_hparams(hparam_dict=hparam_dict, metric_dict=metric_dict)
    return hparam_dict, metric_dict


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train GE')
    parser.add_argument('--dataset', type=str, help='dataset used')
    parser.add_argument('--backbone', type=str, help='backbone model used')
    parser.add_argument('--cuda', type=int, help='cuda device id, -1 for cpu')
    args = parser.parse_args()
    dataset_name = args.dataset
    cuda_id = args.cuda

    torch.set_num_threads(5)
    config_dir = Path('./configs')
    model_name = args.backbone
    method_name = 'GE'

    print('====================================')
    print('====================================')
    print(f'[INFO] Running {method_name} on {dataset_name} with {model_name}')
    print('====================================')

    global_config = yaml.safe_load((config_dir / 'global_config.yml').open('r'))
    local_config_name = get_local_config_name(model_name, dataset_name)
    local_config = yaml.safe_load((config_dir / local_config_name).open('r'))

    data_dir = Path(global_config['data_dir'])
    num_seeds = global_config['num_seeds']

    time = datetime.now().strftime("%m_%d_%Y-%H_%M_%S")
    device = torch.device(f'cuda:{cuda_id}' if cuda_id >= 0 else 'cpu')

    metric_dicts = []
    for random_state in range(num_seeds):
        log_dir = data_dir / dataset_name / 'logs' / (
                time + '-' + dataset_name + '-' + model_name + '-seed' + str(random_state) + '-' + method_name)
        hparam_dict, metric_dict = train_ge_one_seed(local_config, data_dir, log_dir, model_name, dataset_name,
                                                     method_name, device, random_state)
        metric_dicts.append(metric_dict)

    log_dir = data_dir / dataset_name / 'logs' / (
            time + '-' + dataset_name + '-' + model_name + '-seed99-' + method_name + '-stat')
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = Writer(log_dir=log_dir)
    write_stat_from_metric_dicts(hparam_dict, metric_dicts, writer)


if __name__ == '__main__':
    main()
