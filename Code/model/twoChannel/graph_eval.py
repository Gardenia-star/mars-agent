"""图构造策略对比实验。

在相同数据划分、随机种子、模型结构和训练协议下，只替换两个子图的邻接矩阵，
比较不同图策略的最终诊断性能，并对 LiNGAM(CSV) vs 其他策略做配对 t 检验。

结果统一保存到 result/graph_eval/ 目录，不覆盖主实验结果 result/exp*。
"""
import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

import graph_builder
import loaderfile
import modelClass
from main import DATA_DIR, RESULT_DIR, main as train_model
from model_file import CrossModelViews

DEFAULT_STRATEGIES = ["csv", "pearson", "mi", "partial", "random", "complete", "empty"]
METRICS = ["test_acc", "test_f1", "test_recall"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_loaders(root, edge_index1, edge_index2, indices, batch_size):
    ds = modelClass.mGraphDataset(root, edge_index1=edge_index1, edge_index2=edge_index2)
    train_idx, val_idx, test_idx = indices
    train_loader = DataLoader(Subset(ds, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Subset(ds, test_idx), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def compute_metrics_from_csv(exp_dir, run):
    pred_path = exp_dir / "pred" / "test" / f"predLabels{run}.csv"
    true_path = exp_dir / "pred" / "test" / f"rawLabel{run}.csv"
    pred = pd.read_csv(pred_path).values.ravel()
    true = pd.read_csv(true_path).values.ravel()
    return {
        "test_acc": accuracy_score(true, pred),
        "test_f1": f1_score(true, pred, average="macro"),
        "test_recall": recall_score(true, pred, average="macro"),
    }


def run(args):
    os.makedirs(RESULT_DIR / "graph_eval", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x1_all, x2_all = graph_builder.load_node_features(DATA_DIR)
    base_ds = modelClass.mGraphDataset(DATA_DIR)  # 只用于生成与主实验一致的数据划分索引

    records = []
    graph_stats_records = []

    for seed in range(args.seeds):
        # 划分索引只由 seed 决定，不消耗全局 RNG（与 main.data_loading 保持一致）
        split_generator = torch.Generator().manual_seed(seed)
        subsets = loaderfile.random_split(
            base_ds, [0.6, 0.2, 0.2], generator=split_generator
        )
        indices = [s.indices for s in subsets]
        train_idx = indices[0]

        for strategy in args.strategies:
            print(f"\n===== Seed {seed} | Strategy: {strategy} =====")
            # 每个策略重置全局 RNG，使 mask 和模型初始化与 main.Cycle 中同 seed 的 csv 主实验完全一致
            set_seed(seed)

            if strategy == "csv":
                e1, e2 = graph_builder.load_csv_edge_index(DATA_DIR)
            else:
                e1, e2 = graph_builder.build_graph(
                    strategy,
                    x1_all[train_idx],
                    x2_all[train_idx],
                    seed=seed,
                    root=str(DATA_DIR),
                )

            train_loader, val_loader, test_loader = build_loaders(
                DATA_DIR, e1, e2, indices, args.batch_size
            )

            mask_nodes1, mask_nodes2 = loaderfile.mask_views(
                mask_ratio1=0.2,
                mask_ratio2=0.2,
                num_nodes1=11,
                num_nodes2=41,
                view=4,
            )
            model = CrossModelViews.GAT_4Views_attention(
                num_features=1,
                hidden_dim=64,
                heads1=8,
                heads2=4,
                num_classes=5,
                num_expend=32,
                mask_nodes1=mask_nodes1,
                mask_nodes2=mask_nodes2,
                num_view=4,
                cross_attn_heads=8,
                node_num1=11,
                node_num2=41,
            ).to(device)

            exp_dir = RESULT_DIR / "graph_eval" / strategy / f"seed{seed}"
            train_acc, train_loss, val_acc, val_loss, test_acc, test_loss = train_model(
                device,
                model,
                exp_dir,
                train_loader,
                val_loader,
                test_loader,
                run=seed,
                num_epochs=args.epochs,
                patience=args.patience,
                min_epochs=args.min_epochs,
                min_train_acc=args.min_train_acc,
            )

            csv_metrics = compute_metrics_from_csv(exp_dir, run=seed)
            record = {
                "seed": seed,
                "strategy": strategy,
                "test_acc": test_acc,
                "test_f1": csv_metrics["test_f1"],
                "test_recall": csv_metrics["test_recall"],
                "train_acc": train_acc,
                "val_acc": val_acc,
                "test_loss": test_loss,
            }
            records.append(record)

            # 保存图性质
            stats_rows = graph_builder.collect_graph_stats(e1, e2, 11, 41)
            for row in stats_rows:
                row["seed"] = seed
                row["strategy"] = strategy
            graph_stats_records.extend(stats_rows)

            stats_path = exp_dir / "graph_stats.csv"
            os.makedirs(exp_dir, exist_ok=True)
            pd.DataFrame(stats_rows).to_csv(stats_path, index=False)

    results_df = pd.DataFrame(records)
    results_path = RESULT_DIR / "graph_eval" / "results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n所有图策略对比结果已保存到 {results_path}")

    # 图性质汇总
    stats_df = pd.DataFrame(graph_stats_records)
    stats_df.to_csv(RESULT_DIR / "graph_eval" / "graph_stats.csv", index=False)

    # 配对 t 检验：LiNGAM(CSV) vs 其他策略
    if "csv" in results_df["strategy"].unique():
        baseline = "csv"
        ttest_rows = []
        for strategy in results_df["strategy"].unique():
            if strategy == baseline:
                continue
            for metric in METRICS:
                a = results_df.loc[results_df["strategy"] == baseline, metric].values
                b = results_df.loc[results_df["strategy"] == strategy, metric].values
                if len(a) >= 2 and len(a) == len(b):
                    t_stat, p_value = stats.ttest_rel(a, b)
                    ttest_rows.append({
                        "comparison": f"{baseline}_vs_{strategy}",
                        "metric": metric,
                        "t_stat": t_stat,
                        "p_value": p_value,
                        "baseline_mean": a.mean(),
                        "baseline_std": a.std(),
                        "other_mean": b.mean(),
                        "other_std": b.std(),
                    })
        if ttest_rows:
            ttest_df = pd.DataFrame(ttest_rows)
            ttest_path = RESULT_DIR / "graph_eval" / "paired_ttest.csv"
            ttest_df.to_csv(ttest_path, index=False)
            print(f"配对 t 检验结果已保存到 {ttest_path}")

    summary_path = RESULT_DIR / "graph_eval" / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("图构造策略对比实验结果汇总\n")
        f.write("=" * 60 + "\n")
        f.write(results_df.groupby("strategy")[METRICS].agg(["mean", "std"]).to_string())
        f.write("\n")
    print(f"汇总已保存到 {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="图构造策略对比实验")
    parser.add_argument("--epochs", type=int, default=10, help="每个策略的训练 epoch 数")
    parser.add_argument("--seeds", type=int, default=3, help="随机种子个数")
    parser.add_argument("--batch-size", type=int, default=128, help="批大小")
    parser.add_argument("--patience", type=int, default=5, help="早停耐心值")
    parser.add_argument("--min-epochs", type=int, default=5, help="允许早停前至少训练的 epoch 数")
    parser.add_argument("--min-train-acc", type=float, default=0.5, help="允许早停前训练集最低准确率")
    parser.add_argument("--strategies", nargs="+", default=DEFAULT_STRATEGIES,
                        help="图策略列表，默认: csv pearson mi partial random complete empty")
    args = parser.parse_args()
    run(args)
