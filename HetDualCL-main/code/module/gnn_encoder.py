#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time : 2023/3/15 10:04
# @Author : syd 
# @Site :  
# @File : gnn_encoder.py 
# @Software: PyCharm


# import torch
# from dgl.nn.pytorch import HeteroGraphConv, GraphConv

# class rgcn_layer(torch.nn.Module):
#
#     def __init__(self, in_feats, out_feats, rel_names):
#         """
#         RGCN layer
#         support mini-batch training
#         :param in_feats:
#         :param out_feats:
#         :param rel_names: hg.etypes
#         """
#         super(rgcn_layer, self).__init__()
#         self.conv1 = HeteroGraphConv(
#             {rel: GraphConv(in_feats, out_feats, norm='right', weight=True, bias=True) for rel in rel_names},
#             aggregate='sum')
#
#     def forward(self, g, feature):
#         assert isinstance(feature, (dict))
#         h = self.conv1(g, feature)
#         h = {k: torch.nn.functional.relu(v) for k, v in h.items()}
#         return h
#
#
# class GNN_encoder(torch.nn.Module):
#     """
#     the GNN_encoder branch of GTC model
#     """
#
#     def __init__(self, in_feats, hid_feats, out_feats, rel_names, layer_nums=2, category=None):
#         """
#
#         :param in_feats:
#         :param hid_feats:
#         :param out_feats:
#         :param rel_names:
#         :param layer_nums:
#         :param category:
#         """
#         super(GNN_encoder, self).__init__()
#         self.in_feats = in_feats
#         self.hid_feats = hid_feats
#         self.out_feats = out_feats
#         self.rel_names = rel_names
#         self.layer_nums = layer_nums
#         self.category = category
#         self.gcn_layer_list = torch.nn.ModuleList()
#
#         # 动态构建多层 R-GCN
#         for i in range(layer_nums):
#             if i == 0:  # the first layer
#                 if layer_nums == 1:  # only one layer
#                     self.gcn_layer_list.append(
#                         rgcn_layer(in_feats=self.in_feats, out_feats=self.out_feats,
#                                    rel_names=self.rel_names))
#                 else:
#                     self.gcn_layer_list.append(
#                         rgcn_layer(in_feats=self.in_feats, out_feats=self.hid_feats,
#                                    rel_names=self.rel_names))
#
#             elif i == layer_nums - 1:  # the last layer
#                 self.gcn_layer_list.append(
#                     rgcn_layer(in_feats=self.hid_feats, out_feats=self.out_feats,
#                                rel_names=self.rel_names))
#             else:
#                 self.gcn_layer_list.append(
#                     rgcn_layer(in_feats=self.hid_feats, out_feats=self.hid_feats,
#                                rel_names=self.rel_names))
#
#     def forward(self, g, feat, mini_batch_flag=False):
#         """
#         the data flow~
#         :param mini_batch_flag:
#         :param g:
#         :param feat:
#         :return:
#         """
#         h = feat
#         for layer_index in range(self.layer_nums):
#             #  mini-batch
#             if mini_batch_flag:
#                 h = self.gcn_layer_list[layer_index](g=g[layer_index], feature=h)
#             else:
#                 h = self.gcn_layer_list[layer_index](g=g, feature=h)
#         if self.category is not None:
#             out = h[self.category]
#
#         return out

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, HeteroGraphConv

# class rgcn_layer(nn.Module):
#     def __init__(self, in_feats, out_feats, rel_names):
#         super(rgcn_layer, self).__init__()
#         self.conv1 = HeteroGraphConv(
#             {rel: GraphConv(in_feats, out_feats, norm='right', weight=True, bias=True) for rel in rel_names},
#             aggregate='sum'
#         )
#         # 不再预先定义归一化层
#
#     def forward(self, g, feature):
#         h = self.conv1(g, feature)
#         # 动态归一化：对每个节点类型临时创建LayerNorm（第一次调用时初始化）
#         if not hasattr(self, 'norm_dict'):
#             self.norm_dict = nn.ModuleDict({
#                 k: nn.LayerNorm(h[k].shape[-1]).to(h[k].device)
#                 for k in h.keys()  # 自动根据输入特征键创建
#             })
#         h = {k: self.norm_dict[k](v) for k, v in h.items()}
#         h = {k: F.relu(v) for k, v in h.items()}
#         return h
#
# class GNN_encoder(nn.Module):
#     def __init__(self, in_feats, hid_feats, out_feats, rel_names, layer_nums, category=None):
#         super(GNN_encoder, self).__init__()
#         self.in_feats = in_feats
#         self.hid_feats = hid_feats
#         self.out_feats = out_feats
#         self.rel_names = rel_names
#         self.layer_nums = layer_nums
#         self.category = category
#         self.gcn_layer_list = nn.ModuleList()
#
#         for i in range(layer_nums):
#             if i == 0:
#                 if layer_nums == 1:
#                     self.gcn_layer_list.append(rgcn_layer(in_feats, out_feats, rel_names))
#                 else:
#                     self.gcn_layer_list.append(rgcn_layer(in_feats, hid_feats, rel_names))
#             elif i == layer_nums - 1:
#                 self.gcn_layer_list.append(rgcn_layer(hid_feats, out_feats, rel_names))
#             else:
#                 self.gcn_layer_list.append(rgcn_layer(hid_feats, hid_feats, rel_names))
#
#     def forward(self, g, feat, mini_batch_flag=False):
#         h = feat
#         for layer_index in range(self.layer_nums):
#             if mini_batch_flag:
#                 h = self.gcn_layer_list[layer_index](g[layer_index], h)
#             else:
#                 h = self.gcn_layer_list[layer_index](g, h)
#         return h[self.category] if self.category else h
# #
# class EnhancedGNN(nn.Module):
#     def __init__(self, original_gnn, hidden_dim, out_dim, use_residual=True, use_ffn=True, dropout_rate=0.2):
#         super().__init__()
#         self.original_gnn = original_gnn  # 此时original_gnn已包含归一化
#         self.use_residual = use_residual
#         self.use_ffn = use_ffn
#         self.dropout = nn.Dropout(dropout_rate)
#
#         if use_ffn:
#             self.ffn = nn.Sequential(
#                 nn.Linear(out_dim, out_dim * 2),
#                 nn.ReLU(),
#                 nn.Linear(out_dim * 2, out_dim)
#             )
#
#     def forward(self, g, feat):
#         # 原始GNN（现在内部已包含：卷积 → 归一化 → 激活）
#         h = self.original_gnn(g, feat)
#
#         # Dropout
#         h = {k: self.dropout(v) for k, v in h.items()}
#
#         # 残差连接（需维度匹配）
#         if self.use_residual and isinstance(feat, dict) and self.original_gnn.category:
#             residual = feat[self.original_gnn.category]
#             if h[self.original_gnn.category].shape == residual.shape:
#                 h[self.original_gnn.category] = h[self.original_gnn.category] + residual
#
#         # FFN
#         if self.use_ffn:
#             h = {k: self.ffn(v) for k, v in h.items()}
#
#         return h

#先归一化再激活
# class rgcn_layer(nn.Module):
#     def __init__(self, in_feats, out_feats, rel_names, dropout_rate, use_residual=True, use_ffn=True):
#         super(rgcn_layer, self).__init__()
#         self.use_residual = use_residual
#         self.use_ffn = use_ffn
#
#         # 1. 原始RGCN卷积
#         self.conv1 = HeteroGraphConv(
#             {rel: GraphConv(in_feats, out_feats, norm='right', weight=True, bias=True) for rel in rel_names},
#             aggregate='sum'
#         )
#
#         # 2. 动态归一化（LayerNorm）
#         self.norm_dict = None  # 延迟初始化
#
#         # 3. Dropout
#         self.dropout = nn.Dropout(dropout_rate)
#
#         # 4. FFN（可选）
#         if use_ffn:
#             self.ffn = nn.Sequential(
#                 nn.Linear(out_feats, out_feats * 2),
#                 nn.ReLU(),
#                 nn.Linear(out_feats * 2, out_feats)
#             )
#
#     def forward(self, g, feature):
#         # 1. 图卷积
#         h = self.conv1(g, feature)
#
#         # 2. 动态初始化LayerNorm
#         # 确保 norm_dict 存在并初始化
#         if not hasattr(self, 'norm_dict') or self.norm_dict is None:
#             self.norm_dict = nn.ModuleDict({
#                 k: nn.LayerNorm(h[k].shape[-1]).to(h[k].device)
#                 for k in h.keys()
#             })
#         # 检查是否所有需要的 keys 都存在
#         for k in h.keys():
#             if k not in self.norm_dict:
#                 self.norm_dict[k] = nn.LayerNorm(h[k].shape[-1]).to(h[k].device)
#
#         # 3. 归一化 + ReLU
#         h = {k: F.relu(self.norm_dict[k](v)) for k, v in h.items()}
#         # h = {k: F.relu(v) for k, v in h.items()}
#
#         # 4. Dropout
#         # h = {k: self.dropout(v) for k, v in h.items()}
#
#         # 5. 残差连接（如果维度匹配）
#         if self.use_residual:
#             h = {
#                 k: (h[k] + feature[k]) if (k in feature and h[k].shape == feature[k].shape) else h[k]
#                 for k in h.keys()
#             }
#
#         # 6. FFN（可选）
#         if self.use_ffn:
#             h = {k: self.ffn(v) for k, v in h.items()}
#
#         return h
#
# class GNN_encoder(nn.Module):
#     def __init__(self, in_feats, hid_feats, out_feats, rel_names, layer_nums, category=None,
#                  use_residual=True, use_ffn=True, dropout_rate=0.0):
#         super(GNN_encoder, self).__init__()
#         self.in_feats = in_feats
#         self.hid_feats = hid_feats
#         self.out_feats = out_feats
#         self.rel_names = rel_names
#         self.layer_nums = layer_nums
#         self.category = category
#         self.gcn_layer_list = nn.ModuleList()
#
#         for i in range(layer_nums):
#             if i == 0:
#                 if layer_nums == 1:
#                     self.gcn_layer_list.append(rgcn_layer(in_feats, out_feats, rel_names, dropout_rate, use_residual, use_ffn))
#                 else:
#                     self.gcn_layer_list.append(rgcn_layer(in_feats, hid_feats, rel_names, dropout_rate, use_residual, use_ffn))
#             elif i == layer_nums - 1:
#                 self.gcn_layer_list.append(rgcn_layer(hid_feats, out_feats, rel_names, dropout_rate, use_residual, use_ffn))
#             else:
#                 self.gcn_layer_list.append(rgcn_layer(hid_feats, hid_feats, rel_names, dropout_rate, use_residual, use_ffn))
#
#     def forward(self, g, feat, mini_batch_flag=False):
#         h = feat
#         for layer_index in range(self.layer_nums):
#             if mini_batch_flag:
#                 h = self.gcn_layer_list[layer_index](g[layer_index], h)
#             else:
#                 h = self.gcn_layer_list[layer_index](g, h)
#         return h[self.category] if self.category else h


class rgcn_layer(nn.Module):
    def __init__(self, in_feats, out_feats, rel_names, dropout_rate, use_residual=True, use_ffn=True):
        super(rgcn_layer, self).__init__()
        self.use_residual = use_residual
        self.use_ffn = use_ffn

        # 1. 原始RGCN卷积（禁用内置归一化）
        self.conv1 = HeteroGraphConv(
            {rel: GraphConv(in_feats, out_feats, norm='none') for rel in rel_names},
            aggregate='sum'
        )

        # 2. 新增自连接权重 W_0
        self.self_weights = nn.ParameterDict({
            nt: nn.Parameter(torch.Tensor(in_feats, out_feats))
            for nt in set([rel[0] for rel in rel_names])  # 所有源节点类型
        })
        self._init_weights()

        # 3. 动态归一化（保持原实现）
        self.norm_dict = None

        # 4. Dropout（保持原实现）
        self.dropout = nn.Dropout(dropout_rate)

        # 5. FFN（保持原实现）
        if use_ffn:
            self.ffn = nn.Sequential(
                nn.Linear(out_feats, out_feats * 2),
                nn.ReLU(),
                nn.Linear(out_feats * 2, out_feats)
            )

    def _init_weights(self):
        for weight in self.self_weights.values():
            nn.init.xavier_uniform_(weight)

    def forward(self, g, feature):
        # 1. 关系聚合（保持原调用）
        h = self.conv1(g, feature)

        # 2. 加自连接 W_0 h_i^(l)
        h = {
            k: torch.matmul(feature[k], self.self_weights[k]) + v
            if k in self.self_weights and k in feature
            else v
            for k, v in h.items()
        }

        # 3. 动态初始化LayerNorm（保持原逻辑）
        if not hasattr(self, 'norm_dict') or self.norm_dict is None:
            self.norm_dict = nn.ModuleDict({
                k: nn.LayerNorm(h[k].shape[-1]).to(h[k].device)
                for k in h.keys()
            })
        for k in h.keys():
            if k not in self.norm_dict:
                self.norm_dict[k] = nn.LayerNorm(h[k].shape[-1]).to(h[k].device)

        # 4. LayerNorm + ReLU + Dropout（新顺序）
        # h = {k: self.dropout(F.relu(self.norm_dict[k](v))) for k, v in h.items()}
        h = {k: F.relu(self.norm_dict[k](v)) for k, v in h.items()}
        # h = {k: F.relu(v) for k, v in h.items()}


        # 5. 残差连接（保持原判断逻辑）
        if self.use_residual:
            h = {
                k: (h[k] + feature[k]) if (k in feature and h[k].shape == feature[k].shape) else h[k]
                for k in h.keys()
            }

        # 6. FFN（保持原调用）
        if self.use_ffn:
            h = {k: self.ffn(v) for k, v in h.items()}

        return h


class GNN_encoder(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names, layer_nums, category=None,
                 use_residual=True, use_ffn=True, dropout_rate=0.0):
        super(GNN_encoder, self).__init__()
        # 保持原初始化逻辑完全不变
        self.in_feats = in_feats
        self.hid_feats = hid_feats
        self.out_feats = out_feats
        self.rel_names = rel_names
        self.layer_nums = layer_nums
        self.category = category
        self.gcn_layer_list = nn.ModuleList()
        print(self.category)

        for i in range(layer_nums):
            if i == 0:
                if layer_nums == 1:
                    self.gcn_layer_list.append(
                        rgcn_layer(in_feats, out_feats, rel_names, dropout_rate, use_residual, use_ffn))
                else:
                    self.gcn_layer_list.append(
                        rgcn_layer(in_feats, hid_feats, rel_names, dropout_rate, use_residual, use_ffn))
            elif i == layer_nums - 1:
                self.gcn_layer_list.append(
                    rgcn_layer(hid_feats, out_feats, rel_names, dropout_rate, use_residual, use_ffn))
            else:
                self.gcn_layer_list.append(
                    rgcn_layer(hid_feats, hid_feats, rel_names, dropout_rate, use_residual, use_ffn))

    def forward(self, g, feat, mini_batch_flag=False):
        # 保持原前向传播逻辑完全不变
        h = feat
        for layer_index in range(self.layer_nums):
            if mini_batch_flag:
                h = self.gcn_layer_list[layer_index](g[layer_index], h)
            else:
                h = self.gcn_layer_list[layer_index](g, h)
        return h[self.category] if self.category else h