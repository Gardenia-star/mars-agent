"""模型定义与图数据加载模块。"""
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import GATConv


class mGraphDataset(Dataset):
    """从 data 目录读取 graph1_node.csv / graph2_node.csv / edges1.csv / edges2.csv / labels.csv。"""

    def __init__(self, data_dir, edge_index1=None, edge_index2=None):
        super().__init__()
        self.data_dir = data_dir

        x1 = pd.read_csv(os.path.join(data_dir, "graph1_node.csv"), header=None).values.astype(np.float32)
        x2 = pd.read_csv(os.path.join(data_dir, "graph2_node.csv"), header=None).values.astype(np.float32)
        y = pd.read_csv(os.path.join(data_dir, "labels.csv"), header=None).values.squeeze().astype(np.int64)

        # 一次性转换为 Tensor，避免 __getitem__ 中重复 pandas/numpy 转换
        self.x1 = torch.from_numpy(x1)          # (N, node1)
        self.x2 = torch.from_numpy(x2)          # (N, node2)
        self.y = torch.from_numpy(y)            # (N,)

        # 支持外部传入自定义邻接矩阵（用于图构造策略对比实验）；默认读取 CSV 边文件
        self.edge_index1 = edge_index1 if edge_index1 is not None else self._load_edge_index(
            os.path.join(data_dir, "edges1.csv")
        )
        self.edge_index2 = edge_index2 if edge_index2 is not None else self._load_edge_index(
            os.path.join(data_dir, "edges2.csv")
        )

        print(f"数据加载完成！样本数: {len(self.y)}")
        print(f" - 图1节点特征: {self.x1.shape}, 图2节点特征: {self.x2.shape}")
        print(f" - 图1边数: {self.edge_index1.shape[1]}, 图2边数: {self.edge_index2.shape[1]}")

    @staticmethod
    def _load_edge_index(edge_file):
        edge_df = pd.read_csv(edge_file)
        return torch.tensor(edge_df[["source", "target"]].values.T, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        graph1 = Data(
            x=self.x1[idx].unsqueeze(1),  # (node1, 1)
            edge_index=self.edge_index1,
            y=self.y[idx],
        )
        graph2 = Data(
            x=self.x2[idx].unsqueeze(1),  # (node2, 1)
            edge_index=self.edge_index2,
            y=self.y[idx],
        )
        return graph1, graph2




class FeatureAugmenter(nn.Module):
    """一维节点特征扩维。"""

    def __init__(self, input_dim=1, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class GATClassifier(nn.Module):
    """四视图 GAT 分类器（不含跨视图注意力）。"""

    def __init__(
        self,
        num_features=1,
        hidden_dim=64,
        heads1=8,
        heads2=4,
        num_classes=5,
        num_expend=32,
        mask_nodes1=None,
        mask_nodes2=None,
        num_view=4,
        node_num1=11,
        node_num2=41,
    ):
        super().__init__()
        self.mask_nodes1 = mask_nodes1 if mask_nodes1 is not None else [[0, 1]] * num_view
        self.mask_nodes2 = mask_nodes2 if mask_nodes2 is not None else [[0, 1]] * num_view
        self.num_view = num_view
        self.node_num1 = node_num1
        self.node_num2 = node_num2

        self.augmenter = FeatureAugmenter(num_features, num_expend)

        self.graph1_layers = nn.ModuleList()
        self.graph2_layers = nn.ModuleList()
        self.fc1_layers = nn.ModuleList()
        self.fc2_layers = nn.ModuleList()

        for _ in range(num_view):
            self.graph1_layers.append(nn.ModuleList([
                GATConv(num_expend, hidden_dim, heads=heads1),
                GATConv(hidden_dim * heads1, hidden_dim, heads=heads2),
                GATConv(hidden_dim * heads2, hidden_dim, heads=heads2),
            ]))
            self.graph2_layers.append(nn.ModuleList([
                GATConv(num_expend, hidden_dim, heads=heads1),
                GATConv(hidden_dim * heads1, hidden_dim, heads=heads2),
                GATConv(hidden_dim * heads2, hidden_dim, heads=heads2),
            ]))
            self.fc1_layers.append(nn.Linear(hidden_dim * heads2, hidden_dim))
            self.fc2_layers.append(nn.Linear(hidden_dim * heads2, hidden_dim))

        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    @staticmethod
    def graphReadout(x, node_num):
        x = rearrange(x, "(b n) d -> b n d", n=node_num)
        return x.sum(dim=1, keepdim=False)

    def process_view(self, x, edge_index, gat_layers, fc_layer, node_num):
        x = self.augmenter(x)
        for i, conv in enumerate(gat_layers):
            x = F.relu(conv(x, edge_index))
            if i < 2:
                x = F.dropout(x, p=0.3, training=self.training)
        x = fc_layer(x)
        return self.graphReadout(x, node_num)

    @staticmethod
    def apply_mask(x, mask_nodes, graph_num):
        """向量化节点 mask。"""
        x_masked = x.clone()
        idx = torch.arange(x.size(0), device=x.device)
        mask_nodes_t = torch.as_tensor(mask_nodes, dtype=torch.long, device=x.device)
        mask = torch.isin(idx % graph_num, mask_nodes_t)
        x_masked[mask] = 0
        return x_masked

    def forward(self, data1, data2):
        x1, edge_index1 = data1.x, data1.edge_index
        x2, edge_index2 = data2.x, data2.edge_index

        graph1_features = []
        for i in range(self.num_view):
            x_masked = self.apply_mask(x1, self.mask_nodes1[i], self.node_num1)
            graph1_features.append(
                self.process_view(x_masked, edge_index1, self.graph1_layers[i], self.fc1_layers[i], self.node_num1)
            )

        graph2_features = []
        for i in range(self.num_view):
            x_masked = self.apply_mask(x2, self.mask_nodes2[i], self.node_num2)
            graph2_features.append(
                self.process_view(x_masked, edge_index2, self.graph2_layers[i], self.fc2_layers[i], self.node_num2)
            )

        x1 = sum(graph1_features) / self.num_view
        x2 = sum(graph2_features) / self.num_view
        x = torch.cat((x1, x2), dim=1)
        return self.classifier(x)
