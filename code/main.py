import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import numpy
import numpy as np
import torch
from dgl.dataloading import DataLoader
from dgl.dataloading import NeighborSampler
from module import HetDualCL
import time
import random
from self_tools.data_tools import load_data, get_batch_pos
from self_tools.evaluate import evaluate_for_test, evaluate_for_train
from self_tools.params import set_params
import itertools


args = set_params('acm')
if torch.cuda.is_available() and args.device > -1:
    device = torch.device("cuda:0")
    torch.cuda.set_device(args.device)
else:
    device = torch.device("cpu")

## name of intermediate document ##
own_str = args.dataset
exp_num = 2

def get_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / (1024 ** 2)
    return 0


class EfficiencyStats:
    def __init__(self):
        self.train_times = []
        self.infer_times = []
        self.gpu_memories = []
        self.silhouette_scores = []

    def add_experiment(self, avg_epoch_time, infer_time, peak_mem):
        self.train_times.append(avg_epoch_time)
        self.infer_times.append(infer_time)
        self.gpu_memories.append(peak_mem)
        print(self.gpu_memories)

    def add_silhouette_score(self, silhouette_avg):
        self.silhouette_scores.append(silhouette_avg)

    def get_summary(self):
        return {
            'train_mean': np.mean(self.train_times) * 1000,
            'train_std': np.std(self.train_times) * 1000,
            'infer_mean': np.mean(self.infer_times) * 1000,
            'infer_std': np.std(self.infer_times) * 1000,
            'gpu_mean': np.mean(self.gpu_memories),
            'gpu_std': np.std(self.gpu_memories),
            'gpu_max': np.max(self.gpu_memories),
            'silhouette_score':self.silhouette_scores
        }


# 初始化全局统计对象
efficiency_stats = EfficiencyStats()

def make(config, dgl_graph, feats_dim_list, P, h_dict, category, all_node_idx,
         num_classes, mini_batch_flag=True):
    """
    the fuction of building the model, train_loader and optimizer
    :param config:
    :param dgl_graph:
    :param feats_dim_list:
    :param P:
    :param meta_path_adj:
    :param h_dict:
    :param category:
    :param all_node_idx:
    :param num_classes:
    :param mini_batch_flag:
    :return: model，train_loader,optimizer
    """
    print("seed ", config.seed)
    print("Dataset: ", config.dataset)
    print("The number of gnn_branch_num: ", config.gnn_branch_layer_num)
    # build the model
    model = HetDualCL(config.hidden_dim, feats_dim_list, config.feat_drop, P, config.tau, config.lam,
                t_hops=config.t_hops, t_n_class=num_classes, t_input_dim=h_dict[category].shape[1],
                t_pe_dim=config.t_pe_dim, t_n_layers=config.t_n_layers, t_num_heads=config.t_n_heads,
                t_dropout_rate=config.t_dropout,
                t_attention_dropout_rate=config.t_attention_dropout, rel_names=dgl_graph.etypes, category=category,
                gnn_branch_layer_num=config.gnn_branch_layer_num)
    # build the optimizer for
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.l2_coef)

    # NeighborSampler and corresponding graph DataLoader for mini_batch training~
    # for more details for NeighborSampler and DataLoader, please see https://docs.dgl.ai/guide/minibatch.html#guide-minibatch
    fanouts = [20]  # first hop sample 20 neighbors for every node
    for i in range(1, config.gnn_branch_layer_num):
        fanouts.append(10)  # 2-gnn_branch_layer_num hop sample 10 neighbors for every node
    sampler = NeighborSampler(fanouts=fanouts)
    all_idx_dict = {category: all_node_idx}

    train_dataloader_4GTC = DataLoader(graph=dgl_graph, indices=all_idx_dict, graph_sampler=sampler,
                                       batch_size=config.batch_size,
                                       shuffle=True)

    return model, train_dataloader_4GTC, optimizer


def train_flow(model, train_loader, optimizer, config, category, pos, own_str, exp=0):
    cnt_wait = 0
    best = 1e9
    best_t = 0
    print('-' * 60)
    print('train_flow for exp-{}'.format(exp))

    epoch_times = []
    batch_times = []
    mem_usages = []

    starttime = time.time()
    for epoch in range(config.nb_epochs):
        epoch_start = time.time()
        model.train()
        loss_epoch = 0

        for batch_id, (input_nodes, output_nodes, blocks) in enumerate(train_loader):
            batch_start = time.time()

            blocks = [block.to(config.device) for block in blocks]
            # for GNN_branch batch data
            if 'h' in blocks[0].srcdata:
                input_fea4GNN = blocks[0].srcdata['h']
            elif 'feature' in blocks[0].srcdata:
                input_fea4GNN = blocks[0].srcdata['feature']
            else:
                print('please specify the feature key!')
                return
            if not isinstance(input_fea4GNN, dict):
                input_fea4GNN = {category: input_fea4GNN}
            # deal with pos for mini-batch
            pos_batch = get_batch_pos(pos=pos, batch_node_id_x=output_nodes[category].numpy()).to(config.device)
            # [num_meta-paths,num_nodes,num_hops,feature_dim}
            multi_hop_features = blocks[-1].dstnodes[category].data['multi_hop_feature'].permute(1, 0, 2, 3)

            mem_before = get_gpu_memory()

            loss = model(g=blocks, feats=input_fea4GNN, multi_hop_features=multi_hop_features, pos=pos_batch,
                         mini_batch_flag=True)

            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            mem_after = get_gpu_memory()

            loss_epoch = loss_epoch + loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mem_usages.append(mem_after)
            print("exp={}; epoch: {};batch-{}; loss {}; time: {}s; mem: {:.4f}MB".format(
                exp, epoch, batch_id, loss.data.cpu(), batch_time, mem_usages[-1]))

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        print("epoch: {}; epoch_loss {}; epoch_time: {}s".format(
            epoch, loss_epoch.data.cpu(), epoch_time))

        if loss_epoch < best:
            print('best loss: {}->{}'.format(best, loss_epoch))
            best = loss_epoch
            best_t = epoch
            cnt_wait = 0
            # save better checkpoint~
            torch.save(model.state_dict(), '../data/HetDualCL_' + own_str + '.pkl')
        else:
            cnt_wait += 1
            print('lost not improved~ {}'.format(cnt_wait))
        if cnt_wait >= config.patience:
            print('Early stopping at {} epoch!'.format(epoch))
            break

    avg_epoch_time = np.mean(batch_times)
    peak_mem = max(mem_usages)

    print('\nTraining Statistics:')
    print('Average epoch time: {}s'.format(np.mean(epoch_times)))
    print('Average batch time: {}s'.format(np.mean(batch_times)))
    print('Peak GPU memory: {:.4f}MB'.format(max(mem_usages)))

    print('best epoch is {} !'.format(best_t))
    endtime = time.time()
    duration = endtime - starttime
    print('Total train time {} s'.format(duration))
    print('-' * 40)

    return best_t, avg_epoch_time, peak_mem

def test(model, config, train_idx_list, val_idx_list, test_idx_list, labels, num_classes, fea_evalue, ma_dic_list,
         mi_dic_list, auc_dic_list, full_graph=None, feats_dict=None):
    starttime = time.time()
    model.eval()

    infer_start = time.time()
    #Multi-hop views as the final embedding
    emb = model.get_embeds(multi_hop_features=fea_evalue.permute(1, 0, 2, 3))

    #network schema view as the final embedding
    # emb = model.get_gnn_embeds(g=full_graph, feat=feats_dict, mini_batch_flag=False).detach()

    infer_time = time.time() - infer_start
    infer_mem = get_gpu_memory()

    print('Full graph inference time: {:.4f}ms'.format(infer_time * 1000))
    print('Inference GPU memory: {:.4f}MB'.format(infer_mem))

    for i in range(len(train_idx_list)):  # for different data splits for testing~
        ma, mi, auc = evaluate_for_train(config.hidden_dim, train_idx_list[i], val_idx_list[i], test_idx_list[i],
                                         labels, num_classes, config.device, config.dataset, config.eva_lr,
                                         config.eva_wd, batch_size=500, patience=config.patience, emb=emb)
        # record the result of this exp
        ma_dic_list['ma_{}'.format(config.ratio[i])].append(ma)
        mi_dic_list['mi_{}'.format(config.ratio[i])].append(mi)
        auc_dic_list['auc_{}'.format(config.ratio[i])].append(auc)

    endtime = time.time()
    duration = endtime - starttime
    print("Total evaluate time: ", duration, "s")
    print('-' * 40)

    return infer_time, infer_mem

def model_train(args):
    # record the result of each exp
    ma_dic_list = dict.fromkeys(['ma_20', 'ma_40', 'ma_60'])
    for key in ma_dic_list.keys():
        ma_dic_list[key] = []
    mi_dic_list = dict.fromkeys(['mi_20', 'mi_40', 'mi_60'])
    for key in mi_dic_list.keys():
        mi_dic_list[key] = []
    auc_dic_list = dict.fromkeys(['auc_20', 'auc_40', 'auc_60'])
    for key in auc_dic_list.keys():
        auc_dic_list[key] = []

    for exp in range(exp_num):  # every exp
        print('-' * 60)
        print('exp:{}'.format(exp))
        print('-' * 60)
        starttime = time.time()
        if torch.cuda.is_available() and args.device > -1:
            device = torch.device("cuda:0")
            torch.cuda.set_device(args.device)
        else:
            device = torch.device("cpu")

        # name of intermediate document
        own_str = args.dataset + '_' + str(exp)

        # random seed
        seed = args.seed
        numpy.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # load data~
        dgl_graph, category, all_node_idx, train_idx_list, val_idx_list, test_idx_list, \
        h_dict, labels, P, num_classes, pos = load_data(
            data_name=args.dataset, data_dir='../data/', t_hops=args.t_hops,
            cache_sub_dir='cache-opensource')

        feats_dim_list = [h_dict[key].shape[-1] for key in h_dict.keys()]

        from self_tools.theta import compute_pos_stats
        avg, med, maxv, minv, std = compute_pos_stats(pos)
        print(f"Dataset {args.dataset}: pos stats: avg={avg:.2f}, median={med}, max={maxv}, min={minv}, std={std:.2f}")

        # build the model, train_loader and optimizer
        model, train_loader, optimizer = make(args, dgl_graph, feats_dim_list, P, h_dict,
                                              category, all_node_idx, dgl_graph.etypes, num_classes)
        print(model)

        if torch.cuda.is_available() and args.device > -1:
            print('Using CUDA~')
            model.to(device)
            labels = labels.cuda()
            for index in range(len(train_idx_list)):
                train_idx_list[index] = train_idx_list[index].long().cuda()
                val_idx_list[index] = val_idx_list[index].long().cuda()
                test_idx_list[index] = test_idx_list[index].long().cuda()

        # train the model~
        best_t, avg_epoch_time, peak_mem_train = train_flow(model, train_loader, optimizer, args, category, pos, own_str, exp=exp)
        # test the model~
        print('-' * 40)
        print('test paradigm~')
        print('Loading {}th epoch'.format(best_t))
        # load checkpoint
        model.load_state_dict(torch.load('../data/HetDualCL_' + own_str + '.pkl'))
        fea_evalue = dgl_graph.nodes[category].data['multi_hop_feature'].to(device)
        # test flow
        dgl_graph_gpu = dgl_graph.to(device)

        h_dict_gpu = {k: v.to(device) for k, v in h_dict.items()}
        infer_time, infer_mem = test(model, args, train_idx_list, val_idx_list, test_idx_list, labels, num_classes,
                                     fea_evalue, ma_dic_list, mi_dic_list, auc_dic_list, full_graph=dgl_graph_gpu, feats_dict=h_dict_gpu)


        efficiency_stats.add_experiment(
            avg_epoch_time=avg_epoch_time,
            infer_time=infer_time,
            peak_mem=max(peak_mem_train, infer_mem)
        )

        endtime = time.time()
        duration = endtime - starttime
        print("Total time: ", duration, "s")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # print the result
    for key in ma_dic_list.keys():
        lst = ma_dic_list[key]
        print('{}_mean:{:.4f},{}_var:{:.4f}'.format(key, np.mean(lst), key, np.std(lst)))
        # print('{}:{}'.format(key, lst))

    for key in mi_dic_list.keys():
        lst = mi_dic_list[key]
        print('{}_mean:{:.4f},{}_var:{:.4f}'.format(key, np.mean(lst), key, np.std(lst)))
        # print('{}:{}'.format(key, lst))

    for key in auc_dic_list.keys():
        lst = auc_dic_list[key]
        print('{}_mean:{:.4f},{}_var:{:.4f}'.format(key, np.mean(lst), key, np.std(lst)))
        # print('{}:{}'.format(key, lst))


    eff_summary = efficiency_stats.get_summary()
    print("\n=== Efficiency Statistics (10 runs) ===")
    print("Avg Train Time per Epoch: {:.2f} ± {:.2f} ms".format(eff_summary['train_mean'], eff_summary['train_std']))
    print("Avg Inference Time: {:.2f} ± {:.2f} ms".format(eff_summary['infer_mean'], eff_summary['infer_std']))
    print("Avg Peak GPU Memory: {:.2f} ± {:.2f} MB".format(eff_summary['gpu_mean'], eff_summary['gpu_std']))
    print("Peal GPU Memory: {:.2f} MB".format(eff_summary['gpu_max']))

    os.makedirs(f'../result/final_result/', exist_ok=True)
    filename = f'../result/final_result/GTC_final_{args.dataset}.txt'

    run_info = f"\n=== Result ({time.strftime('%Y-%m-%d %H:%M:%S')}) ==="

    divider = "+-----+----------+----------+---------+---------+---------+"
    content = f"""{run_info}
{divider}
[Statistics]
Macro-F1_20: {np.mean(ma_dic_list['ma_20']):.4f} ± {np.std(ma_dic_list['ma_20']):.4f}
Macro-F1_40: {np.mean(ma_dic_list['ma_40']):.4f} ± {np.std(ma_dic_list['ma_40']):.4f}
Macro-F1_60: {np.mean(ma_dic_list['ma_60']):.4f} ± {np.std(ma_dic_list['ma_60']):.4f}
Micro-F1_20: {np.mean(mi_dic_list['mi_20']):.4f} ± {np.std(mi_dic_list['mi_20']):.4f}
Micro-F1_40: {np.mean(mi_dic_list['mi_40']):.4f} ± {np.std(mi_dic_list['mi_40']):.4f}
Micro-F1_60: {np.mean(mi_dic_list['mi_60']):.4f} ± {np.std(mi_dic_list['mi_60']):.4f}
AUC_20:      {np.mean(auc_dic_list['auc_20']):.4f} ± {np.std(auc_dic_list['auc_20']):.4f}
AUC_40:      {np.mean(auc_dic_list['auc_40']):.4f} ± {np.std(auc_dic_list['auc_40']):.4f}
AUC_60:      {np.mean(auc_dic_list['auc_60']):.4f} ± {np.std(auc_dic_list['auc_60']):.4f}
{divider}
"""
    with open(filename, 'a') as f:
        f.write(content)


import matplotlib.pyplot as plt
import numpy as np

def plot_stability_curves(losses_list, val_ma_list, dataset_name):
    """
    losses_list: list of lists, each inner list is epoch losses for one experiment
    val_ma_list: list of lists, each inner list is validation Macro-F1 for one experiment
    """
    if len(losses_list) == 0:
        print("No stability data to plot.")
        return

    max_len = max(len(l) for l in losses_list)
    losses_padded = []
    for l in losses_list:
        pad_len = max_len - len(l)
        if pad_len > 0:
            l = l + [l[-1]] * pad_len
        losses_padded.append(l)
    losses_array = np.array(losses_padded)  # shape: (num_exp, max_len)

    val_ma_padded = []
    for v in val_ma_list:
        pad_len = max_len - len(v)
        if pad_len > 0:
            v = v + [v[-1]] * pad_len
        val_ma_padded.append(v)
    val_ma_array = np.array(val_ma_padded)

    loss_mean = np.mean(losses_array, axis=0)
    loss_std = np.std(losses_array, axis=0)
    ma_mean = np.mean(val_ma_array, axis=0)
    ma_std = np.std(val_ma_array, axis=0)

    epochs = np.arange(1, max_len + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, loss_mean, 'b-', label='Mean Training Loss')
    ax1.fill_between(epochs, loss_mean - loss_std, loss_mean + loss_std, color='b', alpha=0.2, label='Std')
    ax1.set_xlabel('Epoch', fontsize=20)
    ax1.set_ylabel('Loss', fontsize=20)
    ax1.set_title('(a)Training Loss', fontsize=24)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.plot(epochs, ma_mean, 'r-', label='Mean Validation Ma-F1')
    ax2.fill_between(epochs, ma_mean - ma_std, ma_mean + ma_std, color='r', alpha=0.2, label='Std')
    ax2.set_xlabel('Epoch', fontsize=20)
    ax2.set_ylabel('Macro-F1 (%)', fontsize=20)
    ax2.set_title('(b)Validation Performance', fontsize=24)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    save_dir = "./result_image"
    os.makedirs(save_dir, exist_ok=True)
    # save_path = os.path.join(save_dir, f"hyperparameter_lambda.png")
    save_path = os.path.join(save_dir, f'stability_{dataset_name}.pdf')

    # plt.suptitle(f'Training Stability Analysis on {dataset_name} (over {len(losses_list)} runs)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=1200, bbox_inches='tight')
    plt.show()
    print(f'Stability plot saved as stability_{dataset_name}.pdf')

if __name__ == '__main__':
    # if args.load_from_pretrained:  # test the pretrained model
    #     test_pre_trained_model(args)
    # else:  # train new model
    model_train(args)





