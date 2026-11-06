"""骨干网络消融实验：固定 LiNGAM 图，比较不同下游骨干。

模型:
- linear:  单层 GCNConv（无激活） + readout + 线性分类器  （整体线性）
- mlp:     节点 MLP（1->64->64）+ readout + MLP 分类器     （非线性但无图消息传递）
- gat:     modelClass.GATClassifier(num_view=1)           （三层 GAT，无交叉注意力）
- full:    CrossModelViews.GAT_4Views_attention(num_view=4)（当前主模型：GAT + 交叉注意力）

所有模型均使用相同的 LiNGAM 图（data/edges1.csv、edges2.csv）和相同的 6:2:2 数据划分。
结果保存到 result/backbone_ablation/，不覆盖 result/exp* 和 result/graph_eval/。
"""
import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch_geometric.nn import GCNConv

import loaderfile
import modelClass
from main import DATA_DIR, RESULT_DIR, main as train_model
from model_file import CrossModelViews

DEFAULT_MODELS = ["linear", "mlp", "gat", "full"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class LinearGraphModel(nn.Module):
    """单层 GCNConv（无激活） + sum readout + 线性分类器。"""

    def __init__(self, in_dim=1, hidden_dim=64, num_classes=5, node_num1=11, node_num2=41):
        super().__init__()
        self.node_num1 = node_num1
        self.node_num2 = node_num2
        self.conv = GCNConv(in_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, data1, data2):
        x1 = self.conv(data1.x, data1.edge_index)
        x2 = self.conv(data2.x, data2.edge_index)
        x1 = x1.view(-1, self.node_num1, x1.size(-1)).sum(dim=1)
        x2 = x2.view(-1, self.node_num2, x2.size(-1)).sum(dim=1)
        return self.classifier(torch.cat([x1, x2], dim=1))


class MLPGraphModel(nn.Module):
    """节点 MLP（无图消息传递） + sum readout + MLP 分类器。"""

    def __init__(self, in_dim=1, hidden_dim=64, num_classes=5, node_num1=11, node_num2=41):
        super().__init__()
        self.node_num1 = node_num1
        self.node_num2 = node_num2
        self.node_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, data1, data2):
        x1 = self.node_mlp(data1.x)
        x2 = self.node_mlp(data2.x)
        x1 = x1.view(-1, self.node_num1, x1.size(-1)).sum(dim=1)
        x2 = x2.view(-1, self.node_num2, x2.size(-1)).sum(dim=1)
        return self.classifier(torch.cat([x1, x2], dim=1))


def build_model(model_name, mask_nodes1, mask_nodes2):
    if model_name == "linear":
        return LinearGraphModel()
    if model_name == "mlp":
        return MLPGraphModel()
    if model_name == "gat":
        return modelClass.GATClassifier(
            num_features=1,
            hidden_dim=64,
            heads1=8,
            heads2=4,
            num_classes=5,
            num_expend=32,
            mask_nodes1=mask_nodes1,
            mask_nodes2=mask_nodes2,
            num_view=1,
            node_num1=11,
            node_num2=41,
        )
    if model_name == "full":
        return CrossModelViews.GAT_4Views_attention(
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
        )
    raise ValueError(f"未知模型: {model_name}")


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
    os.makedirs(RESULT_DIR / "backbone_ablation", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = []

    for seed in range(args.seeds):
        # 与主实验相同的数据划分
        train_loader, val_loader, test_loader = loaderfile.data_loading(
            root=str(DATA_DIR), batch_size=args.batch_size, graph_strategy="csv", seed=seed
        )

        for model_name in args.models:
            print(f"\n===== Seed {seed} | Backbone: {model_name} =====")
            # 每个模型重置种子，确保 mask 和初始化可复现
            set_seed(seed)

            if model_name in ("linear", "mlp"):
                mask_nodes1, mask_nodes2 = loaderfile.mask_views(
                    0.2, 0.2, num_nodes1=11, num_nodes2=41, view=1
                )
            elif model_name == "gat":
                mask_nodes1, mask_nodes2 = loaderfile.mask_views(
                    0.2, 0.2, num_nodes1=11, num_nodes2=41, view=1
                )
            else:
                mask_nodes1, mask_nodes2 = loaderfile.mask_views(
                    0.2, 0.2, num_nodes1=11, num_nodes2=41, view=4
                )

            model = build_model(model_name, mask_nodes1, mask_nodes2).to(device)
            exp_dir = RESULT_DIR / "backbone_ablation" / model_name / f"seed{seed}"

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
            )

            csv_metrics = compute_metrics_from_csv(exp_dir, run=seed)
            records.append({
                "seed": seed,
                "model": model_name,
                "test_acc": test_acc,
                "test_f1": csv_metrics["test_f1"],
                "test_recall": csv_metrics["test_recall"],
                "train_acc": train_acc,
                "val_acc": val_acc,
                "test_loss": test_loss,
            })

    results_df = pd.DataFrame(records)
    out_path = RESULT_DIR / "backbone_ablation" / "results.csv"
    results_df.to_csv(out_path, index=False)

    summary_path = RESULT_DIR / "backbone_ablation" / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("骨干网络消融实验结果汇总（固定 LiNGAM 图）\n")
        f.write("=" * 60 + "\n")
        f.write(results_df.groupby("model")[["test_acc", "test_f1", "test_recall"]]
                .agg(["mean", "std"]).to_string())
        f.write("\n")
    print(f"\n结果已保存到 {out_path} 和 {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="骨干网络消融实验")
    parser.add_argument("--epochs", type=int, default=10, help="训练 epoch 数")
    parser.add_argument("--seeds", type=int, default=3, help="随机种子个数")
    parser.add_argument("--batch-size", type=int, default=128, help="批大小")
    parser.add_argument("--patience", type=int, default=5, help="早停耐心值")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help="模型列表，默认: linear mlp gat full")
    args = parser.parse_args()
    run(args)
