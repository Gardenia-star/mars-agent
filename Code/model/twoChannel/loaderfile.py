"""数据加载与测试辅助函数。"""
import math
import os
import warnings
from typing import List

import numpy as np
import pandas as pd
import torch
from torch import default_generator, randperm
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
import modelClass

# 当前数据集的图节点数
NODE1_NUM = 11
NODE2_NUM = 41


def random_split(dataset, lengths, generator=default_generator):
    """按比例随机划分数据集（与 PyG 的 random_split 逻辑一致）。"""
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: List[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            subset_lengths.append(int(math.floor(len(dataset) * frac)))
        remainder = len(dataset) - sum(subset_lengths)
        for i in range(remainder):
            subset_lengths[i % len(subset_lengths)] += 1
        lengths = subset_lengths

        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(f"Length of split at index {i} is 0. This might result in an empty dataset.")

    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths), generator=generator).tolist()
    return [
        Subset(dataset, indices[offset - length : offset])
        for offset, length in zip([sum(lengths[:i + 1]) for i in range(len(lengths))], lengths)
    ]


def data_loading(root, batch_size=128, edge_index1=None, edge_index2=None, graph_strategy="csv", seed=0):
    """读取 data 目录并按 60% / 20% / 20% 随机划分训练/验证/测试集。

    当 graph_strategy != "csv" 时，根据指定策略从节点特征中重新构造图：
    lingam / pearson / mi / partial / random / complete / empty。
    """
    if not os.path.exists(root):
        raise ValueError(f"数据根目录不存在: {root}")

    if graph_strategy != "csv":
        import graph_builder

        x1, x2 = graph_builder.load_node_features(root)
        edge_index1, edge_index2 = graph_builder.build_graph(
            graph_strategy, x1, x2, seed=seed, root=root
        )

    graphDataset = modelClass.mGraphDataset(
        root, edge_index1=edge_index1, edge_index2=edge_index2
    )

    # 使用独立 Generator，确保相同 seed 下 main 与 graph_eval 得到完全相同的划分索引
    split_generator = torch.Generator().manual_seed(seed)
    graph_train_set, graph_val_set, graph_test_set = random_split(
        graphDataset, [0.6, 0.2, 0.2], generator=split_generator
    )

    graph_train_loader = DataLoader(graph_train_set, batch_size=batch_size, shuffle=True)
    graph_val_loader = DataLoader(graph_val_set, batch_size=batch_size, shuffle=False)
    graph_test_loader = DataLoader(graph_test_set, batch_size=batch_size, shuffle=False)

    print(f"训练集样本数: {len(graph_train_set)}, 验证集样本数: {len(graph_val_set)}, 测试集样本数: {len(graph_test_set)}")
    return graph_train_loader, graph_val_loader, graph_test_loader


def mask_views(mask_ratio1=0.2, mask_ratio2=0.2, num_nodes1=NODE1_NUM, num_nodes2=NODE2_NUM, view=4):
    """为每个视图随机生成需要掩盖的节点索引。"""
    num_mask1 = max(int(mask_ratio1 * num_nodes1), 1)
    num_mask2 = max(int(mask_ratio2 * num_nodes2), 1)
    mask_nodes1 = [torch.randperm(num_nodes1)[:num_mask1] for _ in range(view)]
    mask_nodes2 = [torch.randperm(num_nodes2)[:num_mask2] for _ in range(view)]
    return mask_nodes1, mask_nodes2


def test(modelNew, graph_loader, criterion, device, save_path=None, run=None):
    """在给定数据加载器上评估模型；save_path 不为空时保存标签、预测和节点特征。"""
    modelNew.eval()
    all_predictions = []
    all_true_labels = []
    node1_list = []
    node2_list = []

    with torch.no_grad():
        correct_test = 0
        total_loss_test = 0
        total_test = 0

        for data in graph_loader:
            batch1_test, batch2_test = data
            batch1_test = batch1_test.to(device)
            batch2_test = batch2_test.to(device)

            out_test = modelNew(batch1_test, batch2_test)
            loss_test = criterion(out_test, batch1_test.y)

            total_loss_test += loss_test.item()
            pred = out_test.argmax(dim=1)
            correct_test += pred.eq(batch1_test.y).sum().item()
            total_test += batch1_test.y.size(0)

            if save_path is not None:
                all_predictions.extend(pred.cpu().tolist())
                all_true_labels.extend(batch1_test.y.cpu().tolist())
                node1_list.append(batch1_test.x.cpu().numpy())
                node2_list.append(batch2_test.x.cpu().numpy())

    Loss = total_loss_test / len(graph_loader)
    Acc = correct_test / total_test

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

        pd.DataFrame({"true_label": all_true_labels}).to_csv(
            os.path.join(save_path, f"rawLabel{run}.csv"), index=False
        )
        pd.DataFrame({"predicted_label": all_predictions}).to_csv(
            os.path.join(save_path, f"predLabels{run}.csv"), index=False
        )

        # 节点特征: (N, node_num, 1) -> (N, node_num)
        node1 = np.concatenate(node1_list, axis=0).squeeze(-1)
        node2 = np.concatenate(node2_list, axis=0).squeeze(-1)
        allnode = np.hstack((node1, node2))

        pd.DataFrame(node1).to_csv(os.path.join(save_path, "node1.csv"), index=False)
        pd.DataFrame(node2).to_csv(os.path.join(save_path, "node2.csv"), index=False)
        pd.DataFrame(allnode).to_csv(os.path.join(save_path, "allnode.csv"), index=False)

        print(f"评估结果已保存至: {save_path}（共 {len(all_true_labels)} 个样本）")

    return Acc, Loss
