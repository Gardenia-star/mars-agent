import torch
from torch_geometric.nn import GATConv
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange

import modelClass


class CrossAttention(nn.Module):
    """交叉注意力模块"""

    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Layer norm
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, query, key_value, query_node_num, kv_node_num):
        """
        query: (batch_size * query_nodes, hidden_dim)
        key_value: (batch_size * kv_nodes, hidden_dim)
        """
        # Reshape to separate batch and nodes
        query = rearrange(query, "(b n) d -> b n d", n=query_node_num)
        key_value = rearrange(key_value, "(b n) d -> b n d", n=kv_node_num)

        # Project Q, K, V
        Q = self.q_proj(query)  # (b, query_nodes, hidden_dim)
        K = self.k_proj(key_value)  # (b, kv_nodes, hidden_dim)
        V = self.v_proj(key_value)  # (b, kv_nodes, hidden_dim)

        # Reshape for multi-head attention
        Q = rearrange(Q, "b n (h d) -> b h n d", h=self.num_heads)
        K = rearrange(K, "b n (h d) -> b h n d", h=self.num_heads)
        V = rearrange(V, "b n (h d) -> b h n d", h=self.num_heads)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim**0.5)
        attn_weights = F.softmax(scores, dim=-1)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)

        # Reshape back
        attn_output = rearrange(attn_output, "b h n d -> b n (h d)")

        # Output projection
        output = self.out_proj(attn_output)

        # Residual connection and layer norm
        output = self.norm(output + query)

        # Reshape back to original format
        output = rearrange(output, "b n d -> (b n) d")

        return output


class GAT_cross(nn.Module):
    def __init__(
        self,
        num_features=1,
        hidden_dim=64,
        heads1=8,
        heads2=4,
        num_classes=5,
        num_expend=32,
        cross_attn_heads=4,
    ):
        super().__init__()
        # 特征扩维
        self.augmenter = modelClass.FeatureAugmenter(num_features, num_expend)

        # 图1GAT函数
        self.conv11 = GATConv(num_expend, hidden_dim, heads=heads1)
        self.conv12 = GATConv(hidden_dim * heads1, hidden_dim, heads=heads2)
        self.conv13 = GATConv(hidden_dim * heads2, hidden_dim, heads=heads2)

        # 图2GAT函数
        self.conv21 = GATConv(num_expend, hidden_dim, heads=heads1)
        self.conv22 = GATConv(hidden_dim * heads1, hidden_dim, heads=heads2)
        self.conv23 = GATConv(hidden_dim * heads2, hidden_dim, heads=heads2)

        # 交叉注意力模块（在第二层GAT之后）
        self.cross_attn_1to2 = CrossAttention(hidden_dim * heads2, cross_attn_heads)
        self.cross_attn_2to1 = CrossAttention(hidden_dim * heads2, cross_attn_heads)

        # 行线性变换
        self.fc1 = nn.Linear(hidden_dim * heads2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim * heads2, hidden_dim)

        # 分类器 列线性变换
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def graphReadout(self, x, node_num):
        x = rearrange(x, "(b n) d -> b n d", n=node_num)
        return x.sum(dim=1, keepdim=False)

    def forward(self, data1, data2):
        """
        Advice:
        @ [graph 1, graph 2]:augmenter -> [hidden 1, hidden 2]
        @ [hidden 1, hidden 2]:gat1 -> [hidden 1, hidden 2]
        @ [hidden 1, hidden 2]: cross attention -> [hidden 1, hidden 2]
        @ [hidden 1, hidden 2]:gat2 -> [hidden 1, hidden 2]
        @ ...
        @ [hidden 1, hidden 2]:cat -> hidden
        @ [hidden]:classifier -> one-hot
        """
        # ====================图1第一层处理====================
        x1, edge_index1 = data1.x, data1.edge_index
        x1 = self.augmenter(x1)  # （32*52）*32
        x1 = F.relu(self.conv11(x1, edge_index1))
        x1 = F.dropout(x1, p=0.3, training=self.training)
        x1 = F.relu(self.conv12(x1, edge_index1))
        x1 = F.dropout(x1, p=0.3, training=self.training)

        # ====================图2第一层处理====================
        x2, edge_index2 = data2.x, data2.edge_index
        x2 = self.augmenter(x2)  # （32*52）*32
        x2 = F.relu(self.conv21(x2, edge_index2))
        x2 = F.dropout(x2, p=0.3, training=self.training)
        x2 = F.relu(self.conv22(x2, edge_index2))
        x2 = F.dropout(x2, p=0.3, training=self.training)

        # ====================第二层交叉注意力====================
        # 图1作为query，图2作为key-value
        x1_updated = self.cross_attn_1to2(x1, x2, query_node_num=11, kv_node_num=41)
        # 图2作为query，图1作为key-value
        x2_updated = self.cross_attn_2to1(x2, x1, query_node_num=41, kv_node_num=11)

        # 更新特征
        x1 = x1_updated
        x2 = x2_updated

        # ====================图1后续处理====================
        x1 = F.relu(self.conv13(x1, edge_index1))
        x1 = self.fc1(x1)  # [batchsize*num, 22] （32*52）* 64
        x1 = self.graphReadout(x1, 11)  # 32*64 # [B * D]

        # ====================图2后续处理====================
        x2 = F.relu(self.conv23(x2, edge_index2))
        x2 = self.fc2(x2)  # [batchsize*num, 22] （32*52）* 64
        x2 = self.graphReadout(x2, 41)  # 32*64 # [B * D]

        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(x)

        # 分类头
        return x
