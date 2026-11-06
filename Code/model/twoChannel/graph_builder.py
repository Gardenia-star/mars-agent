"""图构造策略与图质量评估工具。

所有策略只替换两个子图的邻接矩阵；图构造统一使用节点特征矩阵，不接触标签。
约定:
- LiNGAM/random/complete 生成有向边；
- Pearson / MI / partial 生成无向边，但为了与 PyG 的有向消息传递一致，
  每条选中的无向边会写成两条有向边；
- empty 只保留自环（GATConv 默认会加自环，因此这里不生成任何边）。
"""
import os

import numpy as np
import pandas as pd
import torch

# 当前数据中两个子图的默认 LiNGAM 边数（与 data/edges1.csv、edges2.csv 一致）
DEFAULT_EDGE_NUM1 = 29
DEFAULT_EDGE_NUM2 = 171

STRATEGIES = ["csv", "lingam", "pearson", "mi", "partial", "random", "complete", "empty"]


def load_node_features(root):
    """读取两个子图的节点特征矩阵。"""
    x1 = pd.read_csv(os.path.join(root, "graph1_node.csv"), header=None).values.astype(np.float32)
    x2 = pd.read_csv(os.path.join(root, "graph2_node.csv"), header=None).values.astype(np.float32)
    return x1, x2


def load_csv_edge_index(root):
    """读取原始 CSV 边文件。"""
    def _load(name):
        edge_df = pd.read_csv(os.path.join(root, name))
        return torch.tensor(edge_df[["source", "target"]].values.T, dtype=torch.long)

    return _load("edges1.csv"), _load("edges2.csv")


def adj_to_edge_index(adj):
    """邻接矩阵 -> PyG edge_index。"""
    src, dst = np.nonzero(adj)
    if len(src) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(np.vstack([src, dst]), dtype=torch.long)


def edge_index_to_adj(edge_index, n):
    """PyG edge_index -> 邻接矩阵。"""
    adj = np.zeros((n, n), dtype=np.int64)
    if edge_index is not None and edge_index.shape[1] > 0:
        e = edge_index.cpu().numpy()
        adj[e[0], e[1]] = 1
    return adj


def _top_undirected_pairs(sim_mat, num_directed_edges):
    """从对称相似矩阵中选 top 无向边，返回约 num_directed_edges 条有向边的边列表。"""
    n = sim_mat.shape[0]
    upper = np.triu(np.abs(sim_mat), 1)
    k = max(1, int(np.ceil(num_directed_edges / 2)))
    flat_order = np.argsort(upper, axis=None)[::-1]

    pairs = []
    for flat_idx in flat_order:
        i, j = np.unravel_index(flat_idx, upper.shape)
        if i < j and upper[i, j] > 0:
            pairs.append((i, j))
            if len(pairs) >= k:
                break

    # 无向边 -> 两条有向边，保证 PyG 中边数与原 LiNGAM 有向边数接近
    edges = []
    for i, j in pairs:
        edges.append((i, j))
        edges.append((j, i))
    return edges


def build_lingam_edges(x, num_edges, seed=0):
    """DirectLiNGAM + 按系数绝对值取 top-k 有向边，保证无环且密度可控。"""
    try:
        from lingam import DirectLiNGAM
    except ImportError as exc:
        raise ImportError("lingam 未安装，无法使用 lingam 图策略。") from exc

    model = DirectLiNGAM(random_state=seed)
    model.fit(x)
    adj = np.abs(model.adjacency_matrix_)
    np.fill_diagonal(adj, 0)

    n = x.shape[1]
    max_edges = n * (n - 1)
    num_edges = min(num_edges, max_edges)
    flat_order = np.argsort(adj, axis=None)[::-1]

    edges = []
    for flat_idx in flat_order:
        i, j = np.unravel_index(flat_idx, adj.shape)
        if i != j and adj[i, j] > 0:
            edges.append((i, j))
            if len(edges) >= num_edges:
                break

    return adj_to_edge_index(np.array([[1 if (i, j) in edges else 0 for j in range(n)] for i in range(n)]))


def build_pearson_edges(x, num_directed_edges, seed=0):
    """Pearson 相关绝对值 top 无向边。"""
    corr = np.corrcoef(x, rowvar=False)
    np.fill_diagonal(corr, 0)
    edges = _top_undirected_pairs(corr, num_directed_edges)
    return _edges_to_edge_index(edges, x.shape[1])


def build_mi_edges(x, num_directed_edges, seed=0):
    """互信息 top 无向边。"""
    from sklearn.feature_selection import mutual_info_regression

    n = x.shape[1]
    mi = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            score_ij = mutual_info_regression(x[:, [i]], x[:, j])[0]
            score_ji = mutual_info_regression(x[:, [j]], x[:, i])[0]
            mi[i, j] = mi[j, i] = (score_ij + score_ji) / 2.0
    edges = _top_undirected_pairs(mi, num_directed_edges)
    return _edges_to_edge_index(edges, n)


def build_partial_edges(x, num_directed_edges, seed=0):
    """偏相关（精度矩阵）绝对值 top 无向边。"""
    cov = np.cov(x, rowvar=False)
    prec = np.linalg.pinv(cov, rcond=1e-6)
    d = np.sqrt(np.diag(prec))
    d[d == 0] = 1e-12
    pcorr = -prec / np.outer(d, d)
    np.fill_diagonal(pcorr, 0)
    edges = _top_undirected_pairs(pcorr, num_directed_edges)
    return _edges_to_edge_index(edges, x.shape[1])


def build_random_edges(node_num, num_directed_edges, seed=0):
    """与 LiNGAM 边数相同的随机有向图。"""
    rng = np.random.default_rng(seed)
    all_pairs = [(i, j) for i in range(node_num) for j in range(node_num) if i != j]
    num_directed_edges = min(num_directed_edges, len(all_pairs))
    chosen = rng.choice(len(all_pairs), size=num_directed_edges, replace=False)
    edges = [all_pairs[k] for k in chosen]
    return _edges_to_edge_index(edges, node_num)


def build_complete_edges(node_num):
    """完全有向图（不含自环）。"""
    edges = [(i, j) for i in range(node_num) for j in range(node_num) if i != j]
    return _edges_to_edge_index(edges, node_num)


def build_empty_edges(node_num):
    """空图（GATConv 会加自环）。"""
    return torch.empty((2, 0), dtype=torch.long)


def _edges_to_edge_index(edges, node_num):
    if len(edges) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    e = np.array(edges, dtype=np.int64).T
    return torch.tensor(e, dtype=torch.long)


def build_graph(strategy, x1, x2, seed=0, root="."):
    """按指定策略构造两个子图的 edge_index。"""
    if strategy == "csv":
        return load_csv_edge_index(root)
    if strategy == "lingam":
        # 主实验的 LiNGAM 图已在 data/edges1.csv、edges2.csv 中，直接读取以保证与论文结果一致。
        # 如需从数据重新拟合 LiNGAM，可使用 build_lingam_edges()。
        return load_csv_edge_index(root)
    elif strategy == "pearson":
        e1 = build_pearson_edges(x1, DEFAULT_EDGE_NUM1)
        e2 = build_pearson_edges(x2, DEFAULT_EDGE_NUM2)
    elif strategy == "mi":
        e1 = build_mi_edges(x1, DEFAULT_EDGE_NUM1)
        e2 = build_mi_edges(x2, DEFAULT_EDGE_NUM2)
    elif strategy == "partial":
        e1 = build_partial_edges(x1, DEFAULT_EDGE_NUM1)
        e2 = build_partial_edges(x2, DEFAULT_EDGE_NUM2)
    elif strategy == "random":
        e1 = build_random_edges(x1.shape[1], DEFAULT_EDGE_NUM1, seed=seed)
        e2 = build_random_edges(x2.shape[1], DEFAULT_EDGE_NUM2, seed=seed)
    elif strategy == "complete":
        e1 = build_complete_edges(x1.shape[1])
        e2 = build_complete_edges(x2.shape[1])
    elif strategy == "empty":
        e1 = build_empty_edges(x1.shape[1])
        e2 = build_empty_edges(x2.shape[1])
    else:
        raise ValueError(f"未知图策略: {strategy}，可选值: {STRATEGIES}")
    return e1, e2


def graph_stats(edge_index, node_num):
    """统计一个有向图的基本性质。"""
    import networkx as nx

    g = nx.DiGraph()
    g.add_nodes_from(range(node_num))
    if edge_index is not None and edge_index.shape[1] > 0:
        e = edge_index.cpu().numpy()
        g.add_edges_from(zip(e[0].tolist(), e[1].tolist()))

    n = g.number_of_nodes()
    m = g.number_of_edges()
    density = nx.density(g) if n > 1 else 0.0
    in_deg = np.mean([d for _, d in g.in_degree()]) if n else 0.0
    out_deg = np.mean([d for _, d in g.out_degree()]) if n else 0.0
    is_dag = nx.is_directed_acyclic_graph(g)
    n_strong = nx.number_strongly_connected_components(g)
    return {
        "node_num": n,
        "edge_num": m,
        "density": density,
        "mean_in_degree": in_deg,
        "mean_out_degree": out_deg,
        "is_dag": is_dag,
        "strong_components": n_strong,
    }


def collect_graph_stats(edge_index1, edge_index2, node_num1, node_num2):
    """收集两个子图的图性质。"""
    s1 = graph_stats(edge_index1, node_num1)
    s2 = graph_stats(edge_index2, node_num2)
    rows = []
    for graph_name, stats, node_num in [("graph1", s1, node_num1), ("graph2", s2, node_num2)]:
        stats["graph"] = graph_name
        stats["node_num"] = node_num
        rows.append(stats)
    return rows


def save_graph_stats(edge_index1, edge_index2, node_num1, node_num2, save_path):
    """保存两个子图的图性质到 CSV。"""
    import pandas as pd

    rows = collect_graph_stats(edge_index1, edge_index2, node_num1, node_num2)
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    return df
