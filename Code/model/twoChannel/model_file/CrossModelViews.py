"""四视图 GAT + 跨视图注意力模型。"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch_geometric.nn import GATConv


class CrossAttention(nn.Module):
    """交叉注意力模块。"""

    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, query, key_value, query_node_num, kv_node_num):
        query = rearrange(query, "(b n) d -> b n d", n=query_node_num)
        key_value = rearrange(key_value, "(b n) d -> b n d", n=kv_node_num)

        Q = self.q_proj(query)
        K = self.k_proj(key_value)
        V = self.v_proj(key_value)

        Q = rearrange(Q, "b n (h d) -> b h n d", h=self.num_heads)
        K = rearrange(K, "b n (h d) -> b h n d", h=self.num_heads)
        V = rearrange(V, "b n (h d) -> b h n d", h=self.num_heads)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        attn_output = rearrange(attn_output, "b h n d -> b n (h d)")
        output = self.out_proj(attn_output)
        output = self.norm(output + query)

        return rearrange(output, "b n d -> (b n) d")


class TemporalEncoder(nn.Module):
    """时序编码器，支持 CNN / GRU / TCN。

    输入: (N, L)，每个节点一个长度为 L 的原始窗口序列；
    输出: (N, out_dim)，将时序压缩为固定维度供 GAT 使用。
    """

    def __init__(self, in_channels=1, out_dim=16, encoder_type="cnn"):
        super().__init__()
        self.encoder_type = encoder_type

        if encoder_type == "cnn":
            self.conv1 = nn.Conv1d(in_channels, out_dim, kernel_size=5, padding=2)
            self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.act = nn.ReLU()
        elif encoder_type == "gru":
            self.gru = nn.GRU(
                input_size=in_channels,
                hidden_size=out_dim,
                num_layers=1,
                batch_first=True,
            )
        elif encoder_type == "tcn":
            self.convs = nn.ModuleList()
            for i, dilation in enumerate([1, 2, 4, 8]):
                in_ch = in_channels if i == 0 else out_dim
                self.convs.append(
                    nn.Conv1d(in_ch, out_dim, kernel_size=3, padding=dilation, dilation=dilation)
                )
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.act = nn.ReLU()
        else:
            raise ValueError(f"未知 encoder_type: {encoder_type}")

    def forward(self, x):
        if self.encoder_type == "cnn":
            x = x.unsqueeze(1)  # (N,1,L)
            x = self.act(self.conv1(x))
            x = self.act(self.conv2(x))
            x = self.pool(x).squeeze(-1)
        elif self.encoder_type == "gru":
            x = x.unsqueeze(-1)  # (N,L,1)
            _, h = self.gru(x)
            x = h[-1]  # (N,out_dim)
        elif self.encoder_type == "tcn":
            x = x.unsqueeze(1)
            for conv in self.convs:
                x = self.act(conv(x))
            x = self.pool(x).squeeze(-1)
        return x


class GAT_4Views_attention(nn.Module):
    """四视图 GAT，并在第一层 GAT 后加入跨视图注意力。"""

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
        cross_attn_heads=8,
        node_num1=11,
        node_num2=41,
        dropout=0.3,
        use_temporal_encoder=False,
        temporal_dim=16,
        encoder_type="cnn",
    ):
        super().__init__()
        self.mask_nodes1 = mask_nodes1 if mask_nodes1 is not None else [[0, 1]] * num_view
        self.mask_nodes2 = mask_nodes2 if mask_nodes2 is not None else [[0, 1]] * num_view
        self.num_view = num_view
        self.node_num1 = node_num1
        self.node_num2 = node_num2
        self.dropout = dropout
        self.use_temporal_encoder = use_temporal_encoder
        self.temporal_dim = temporal_dim

        # 若启用时序编码器，GAT 输入维度变为 temporal_dim
        gat_input_dim = temporal_dim if use_temporal_encoder else num_features
        if use_temporal_encoder:
            self.temporal_encoder = TemporalEncoder(
                in_channels=1, out_dim=temporal_dim, encoder_type=encoder_type
            )

        self.graph1_layers = nn.ModuleList()
        self.graph2_layers = nn.ModuleList()
        self.fc1_layers = nn.ModuleList()
        self.fc2_layers = nn.ModuleList()
        self.graph1_attention = nn.ModuleList()
        self.graph2_attention = nn.ModuleList()

        for _ in range(num_view):
            self.graph1_layers.append(nn.ModuleList([
                GATConv(gat_input_dim, hidden_dim, heads=heads1),
                GATConv(hidden_dim * heads1, hidden_dim, heads=heads2),
                GATConv(hidden_dim * heads2, hidden_dim, heads=heads2),
            ]))
            self.graph1_attention.append(nn.ModuleList([
                CrossAttention(hidden_dim * heads1, cross_attn_heads)
                for _ in range(num_view - 1)
            ]))

            self.graph2_layers.append(nn.ModuleList([
                GATConv(gat_input_dim, hidden_dim, heads=heads1),
                GATConv(hidden_dim * heads1, hidden_dim, heads=heads2),
                GATConv(hidden_dim * heads2, hidden_dim, heads=heads2),
            ]))
            self.graph2_attention.append(nn.ModuleList([
                CrossAttention(hidden_dim * heads1, cross_attn_heads)
                for _ in range(num_view - 1)
            ]))

            self.fc1_layers.append(nn.Linear(hidden_dim * heads2, hidden_dim))
            self.fc2_layers.append(nn.Linear(hidden_dim * heads2, hidden_dim))

        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    @staticmethod
    def graphReadout(x, node_num):
        x = rearrange(x, "(b n) d -> b n d", n=node_num)
        return x.sum(dim=1, keepdim=False)

    @staticmethod
    def apply_mask(x, mask_nodes, graph_num):
        """向量化节点 mask。"""
        x_masked = x.clone()
        idx = torch.arange(x.size(0), device=x.device)
        mask_nodes_t = torch.as_tensor(mask_nodes, dtype=torch.long, device=x.device)
        mask = torch.isin(idx % graph_num, mask_nodes_t)
        x_masked[mask] = 0
        return x_masked

    def process_view_early(self, x, edge_index, graph_layers):
        """第一层 GAT 处理。"""
        x = F.relu(graph_layers[0](x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def process_view_attention(self, graph_features, graph_attention, query_node_num, kv_node_num):
        """每个视图与其他视图做交叉注意力后取平均。"""
        attention_value = []
        for i in range(self.num_view):
            value_computer = [graph_features[j] for j in range(self.num_view) if j != i]
            value_att = [
                conv_attention(value_computer[k], graph_features[i], query_node_num, kv_node_num)
                for k, conv_attention in enumerate(graph_attention[i])
            ]
            attention_value.append(torch.mean(torch.stack(value_att), dim=0))
        return attention_value

    def process_view_later(self, x, edge_index, graph_layers, fc_layer, node_num):
        """后两层 GAT + FC + 图读出。"""
        for i, conv in enumerate(graph_layers[1:]):
            x = F.relu(conv(x, edge_index))
            if i < 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = fc_layer(x)
        return self.graphReadout(x, node_num)

    def process_view(
        self,
        x,
        edge_index,
        graph_masks,
        graph_layers,
        graph_attention,
        fc_layer,
        query_node_num,
        kv_node_num,
        node_num,
    ):
        graph_features = []
        for i in range(self.num_view):
            x_masked = self.apply_mask(x, graph_masks[i], node_num)
            graph_features.append(self.process_view_early(x_masked, edge_index, graph_layers[i]))

        graph_features = self.process_view_attention(
            graph_features, graph_attention, query_node_num, kv_node_num
        )

        return [
            self.process_view_later(graph_features[i], edge_index, graph_layers[i], fc_layer[i], node_num)
            for i in range(self.num_view)
        ]

    def forward_features(self, data1, data2):
        """返回分类器输入前的图特征向量，供 GRU 等序列模型使用。"""
        x1, edge_index1 = data1.x, data1.edge_index
        x2, edge_index2 = data2.x, data2.edge_index

        if self.use_temporal_encoder:
            x1 = self.temporal_encoder(x1)
            x2 = self.temporal_encoder(x2)

        graph1_features = self.process_view(
            x1, edge_index1, self.mask_nodes1, self.graph1_layers,
            self.graph1_attention, self.fc1_layers,
            self.node_num1, self.node_num1, self.node_num1,
        )
        graph2_features = self.process_view(
            x2, edge_index2, self.mask_nodes2, self.graph2_layers,
            self.graph2_attention, self.fc2_layers,
            self.node_num2, self.node_num2, self.node_num2,
        )

        x1 = sum(graph1_features) / self.num_view
        x2 = sum(graph2_features) / self.num_view
        return torch.cat((x1, x2), dim=1)

    def forward(self, data1, data2):
        features = self.forward_features(data1, data2)
        return self.classifier(features)
