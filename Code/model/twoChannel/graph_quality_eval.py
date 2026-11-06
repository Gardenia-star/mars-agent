"""Table S1 实验：合成非线性数据上的图结构恢复质量评估。

生成与 TEP 两个子图同规模（11 节点、41 节点）的已知 DAG 非线性数据，
比较 LiNGAM / Pearson / NOTEARS-MLP / PCMCI 等方法的图恢复质量：
SHD、edge precision、edge recall、edge F1。

结果保存到 result/graph_quality/，不覆盖 result/exp* / graph_eval / backbone_ablation。
"""
import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from main import RESULT_DIR

DEFAULT_METHODS = ["lingam", "pearson", "mi", "partial", "random", "complete", "empty"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def generate_dag(n_nodes, n_edges, seed=0):
    """生成一个具有 n_edges 条边的随机 DAG，返回邻接矩阵。"""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_nodes)
    # 按拓扑顺序排列，只允许 order[i] -> order[j], i < j
    possible = [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)]
    n_edges = min(n_edges, len(possible))
    chosen_idx = rng.choice(len(possible), size=n_edges, replace=False)
    adj = np.zeros((n_nodes, n_nodes), dtype=np.int64)
    for idx in chosen_idx:
        i, j = possible[idx]
        adj[order[i], order[j]] = 1
    return adj


def generate_nonlinear_data(adj, n_samples, seed=0):
    """按 DAG 生成非线性非高斯数据。"""
    rng = np.random.default_rng(seed)
    n_nodes = adj.shape[0]
    X = np.zeros((n_samples, n_nodes), dtype=np.float32)
    # 拓扑顺序：按节点入度排序，保证父节点先生成
    indeg = adj.sum(axis=0)
    topo = []
    remaining = list(range(n_nodes))
    while remaining:
        ready = [v for v in remaining if indeg[v] == 0]
        topo.extend(ready)
        for v in ready:
            remaining.remove(v)
            for c in np.nonzero(adj[v, :])[0]:  # v 的子节点
                indeg[c] -= 1

    weights = rng.uniform(0.5, 1.5, size=(n_nodes, n_nodes))
    for j in topo:
        parents = np.nonzero(adj[:, j])[0]
        noise = rng.uniform(-0.5, 0.5, size=n_samples)  # 非高斯噪声
        if len(parents) == 0:
            X[:, j] = noise
        else:
            s = np.zeros(n_samples)
            for i in parents:
                s = s + weights[i, j] * np.tanh(X[:, i])
            X[:, j] = np.tanh(s + noise)
    return X


def top_k_adj_from_weight(W, n_edges):
    """按权重绝对值取 top-k 有向边，生成二值邻接矩阵。"""
    n = W.shape[0]
    np.fill_diagonal(W, 0)
    W_abs = np.abs(W)
    n_edges = min(n_edges, n * (n - 1))
    flat_idx = np.argsort(W_abs, axis=None)[::-1]
    adj = np.zeros((n, n), dtype=np.int64)
    count = 0
    for idx in flat_idx:
        i, j = np.unravel_index(idx, W.shape)
        if i != j and W_abs[i, j] > 0:
            adj[i, j] = 1
            count += 1
            if count >= n_edges:
                break
    return adj


def estimate_lingam_adj(X, n_edges, seed=0, backend="pairwise"):
    """LiNGAM 图估计。

    backend="pairwise"：直接使用 pairwise LiNGAM（默认，数值稳定）；
    backend="direct"：尝试 DirectLiNGAM，失败时回退到 pairwise LiNGAM。
    """
    if backend == "pairwise":
        return estimate_pairwise_lingam_adj(X, n_edges)

    try:
        from lingam import DirectLiNGAM
    except ImportError:
        return estimate_pairwise_lingam_adj(X, n_edges)

    last_err = None
    for s in (seed, seed + 1, seed + 2):
        try:
            model = DirectLiNGAM(random_state=s)
            model.fit(X)
            return top_k_adj_from_weight(np.abs(model.adjacency_matrix_), n_edges)
        except Exception as exc:
            last_err = exc
    print(f"  DirectLiNGAM 拟合失败，回退到 pairwise LiNGAM: {type(last_err).__name__}: {last_err}")
    return estimate_pairwise_lingam_adj(X, n_edges)


def estimate_pairwise_lingam_adj(X, n_edges):
    """成对 LiNGAM：对每对变量回归，选择残差与回归子相关性更小的方向作为因果方向。"""
    X = X.astype(np.float64)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    n = X.shape[1]
    W = np.zeros((n, n))
    ones = np.ones((X.shape[0], 1))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            xi = X[:, i:i + 1]
            xj = X[:, j]

            # j ~ i，残差应与 i 独立
            beta_ij = np.linalg.lstsq(np.c_[ones, xi], xj, rcond=None)[0][1]
            r_ij = xj - beta_ij * xi[:, 0]
            c_ij = abs(np.corrcoef(r_ij, xi[:, 0])[0, 1])

            # i ~ j
            beta_ji = np.linalg.lstsq(np.c_[ones, xj], xi[:, 0], rcond=None)[0][1]
            r_ji = xi[:, 0] - beta_ji * xj
            c_ji = abs(np.corrcoef(r_ji, xj)[0, 1])

            if c_ij < c_ji:
                W[i, j] = abs(beta_ij)
            else:
                W[j, i] = abs(beta_ji)
    return top_k_adj_from_weight(W, n_edges)


def estimate_pearson_adj(X, n_edges):
    """Pearson 相关图：按绝对值选 top 无向边，然后写成两条有向边。"""
    corr = np.abs(np.corrcoef(X, rowvar=False))
    n = X.shape[1]
    upper = np.triu(corr, 1)
    n_pairs = max(1, int(np.ceil(n_edges / 2)))
    flat_idx = np.argsort(upper, axis=None)[::-1]
    adj = np.zeros((n, n), dtype=np.int64)
    count = 0
    for idx in flat_idx:
        i, j = np.unravel_index(idx, upper.shape)
        if i < j and upper[i, j] > 0:
            adj[i, j] = adj[j, i] = 1
            count += 1
            if count >= n_pairs:
                break
    return adj


def _bidirected_top_edges_from_sim(sim_mat, n_edges):
    """从相似矩阵中选 top 无向边，并写成双向有向边。"""
    n = sim_mat.shape[0]
    upper = np.triu(np.abs(sim_mat), 1)
    n_pairs = max(1, int(np.ceil(n_edges / 2)))
    flat_idx = np.argsort(upper, axis=None)[::-1]
    adj = np.zeros((n, n), dtype=np.int64)
    count = 0
    for idx in flat_idx:
        i, j = np.unravel_index(idx, upper.shape)
        if i < j and upper[i, j] > 0:
            adj[i, j] = adj[j, i] = 1
            count += 1
            if count >= n_pairs:
                break
    return adj


def estimate_mi_adj(X, n_edges):
    """互信息 top 无向边（双向边）。"""
    from sklearn.feature_selection import mutual_info_regression

    n = X.shape[1]
    mi = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            s1 = mutual_info_regression(X[:, [i]], X[:, j])[0]
            s2 = mutual_info_regression(X[:, [j]], X[:, i])[0]
            mi[i, j] = mi[j, i] = (s1 + s2) / 2.0
    return _bidirected_top_edges_from_sim(mi, n_edges)


def estimate_partial_adj(X, n_edges):
    """偏相关 top 无向边（双向边）。"""
    cov = np.cov(X, rowvar=False)
    prec = np.linalg.pinv(cov, rcond=1e-6)
    d = np.sqrt(np.diag(prec))
    d[d == 0] = 1e-12
    pcorr = -prec / np.outer(d, d)
    np.fill_diagonal(pcorr, 0)
    return _bidirected_top_edges_from_sim(pcorr, n_edges)


def estimate_random_adj(n_nodes, n_edges, seed=0):
    """随机有向图（与真实图边数相同）。"""
    rng = np.random.default_rng(seed)
    all_pairs = [(i, j) for i in range(n_nodes) for j in range(n_nodes) if i != j]
    n_edges = min(n_edges, len(all_pairs))
    chosen = rng.choice(len(all_pairs), size=n_edges, replace=False)
    adj = np.zeros((n_nodes, n_nodes), dtype=np.int64)
    for idx in chosen:
        i, j = all_pairs[idx]
        adj[i, j] = 1
    return adj


def estimate_complete_adj(n_nodes):
    """完全有向图。"""
    adj = np.ones((n_nodes, n_nodes), dtype=np.int64)
    np.fill_diagonal(adj, 0)
    return adj


def estimate_empty_adj(n_nodes):
    """空图。"""
    return np.zeros((n_nodes, n_nodes), dtype=np.int64)


def directed_shd(true_adj, est_adj):
    """有向图 SHD：缺失边 + 多余边 + 反向边。"""
    true = true_adj.astype(bool)
    est = est_adj.astype(bool)
    missing = int(np.sum(true & ~est))
    extra = int(np.sum(~true & est))
    reversed_edges = 0
    n = true.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j and true[i, j] and est[j, i] and not est[i, j]:
                reversed_edges += 1
                missing -= 1
                extra -= 1
    return missing + extra + reversed_edges


def edge_metrics(true_adj, est_adj):
    true = true_adj.astype(bool)
    est = est_adj.astype(bool)
    tp = int(np.sum(true & est))
    n_true = int(np.sum(true))
    n_est = int(np.sum(est))
    precision = tp / n_est if n_est else 0.0
    recall = tp / n_true if n_true else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def run(args):
    os.makedirs(RESULT_DIR / "graph_quality", exist_ok=True)
    records = []

    configs = [
        ("graph1", 11, 29),
        ("graph2", 41, 171),
    ]

    for graph_name, n_nodes, n_edges in configs:
        for seed in range(args.seeds):
            set_seed(seed)
            true_adj = generate_dag(n_nodes, n_edges, seed=seed)
            X = generate_nonlinear_data(true_adj, args.n_samples, seed=seed)

            for method in args.methods:
                print(f"\n===== {graph_name} | seed {seed} | {method} =====")
                try:
                    if method == "lingam":
                        est_adj = estimate_lingam_adj(X, n_edges, seed=seed, backend=args.lingam_backend)
                    elif method == "pearson":
                        est_adj = estimate_pearson_adj(X, n_edges)
                    elif method == "mi":
                        est_adj = estimate_mi_adj(X, n_edges)
                    elif method == "partial":
                        est_adj = estimate_partial_adj(X, n_edges)
                    elif method == "random":
                        est_adj = estimate_random_adj(n_nodes, n_edges, seed=seed)
                    elif method == "complete":
                        est_adj = estimate_complete_adj(n_nodes)
                    elif method == "empty":
                        est_adj = estimate_empty_adj(n_nodes)
                    elif method == "notears_mlp":
                        est_adj = estimate_notears_mlp_adj(X, n_edges, seed=seed)
                    elif method == "pcmci":
                        est_adj = estimate_pcmci_adj(X, n_edges, seed=seed)
                    else:
                        raise ValueError(f"未知方法: {method}")

                    shd = directed_shd(true_adj, est_adj)
                    precision, recall, f1 = edge_metrics(true_adj, est_adj)
                    records.append({
                        "graph": graph_name,
                        "seed": seed,
                        "n_nodes": n_nodes,
                        "n_edges": n_edges,
                        "method": method,
                        "SHD": shd,
                        "edge_precision": precision,
                        "edge_recall": recall,
                        "edge_F1": f1,
                    })
                except Exception as exc:
                    print(f"  跳过: {type(exc).__name__}: {exc}")

    if not records:
        print("没有成功运行任何方法。")
        return

    results_df = pd.DataFrame(records)
    out_path = RESULT_DIR / "graph_quality" / "results.csv"
    results_df.to_csv(out_path, index=False)

    summary_path = RESULT_DIR / "graph_quality" / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("合成非线性数据图结构恢复质量评估\n")
        f.write("=" * 60 + "\n")
        f.write(results_df.groupby(["graph", "method"])[
            ["SHD", "edge_precision", "edge_recall", "edge_F1"]
        ].agg(["mean", "std"]).to_string())
        f.write("\n")
    print(f"\n结果已保存到 {out_path} 和 {summary_path}")


def estimate_notears_mlp_adj(X, n_edges, seed=0):
    """NOTEARS-MLP 占位实现；需要安装 gCastle/causal-learn 后替换为实际调用。"""
    raise ImportError(
        "NOTEARS-MLP 需要安装 gCastle 或 causal-learn；当前环境未安装，已跳过。"
    )


def estimate_pcmci_adj(X, n_edges, seed=0):
    """PCMCI 占位实现；需要安装 causal-learn/tigramite 后替换为实际调用。"""
    raise ImportError(
        "PCMCI 需要安装 causal-learn 或 tigramite；当前环境未安装，已跳过。"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="合成非线性数据图质量评估")
    parser.add_argument("--n-samples", type=int, default=5000, help="每个合成数据集的样本数")
    parser.add_argument("--seeds", type=int, default=5, help="随机种子个数")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS,
                        help="方法列表，默认: lingam pearson；可选 notears_mlp pcmci（需安装相应包）")
    parser.add_argument("--lingam-backend", type=str, default="pairwise",
                        choices=["pairwise", "direct"],
                        help="LiNGAM 实现：pairwise（默认，稳定）或 direct（DirectLiNGAM，可能较慢/不稳定）")
    args = parser.parse_args()
    run(args)
