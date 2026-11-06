"""主入口：读取 data 目录数据，训练双通道图注意力模型，结果按 run 写入 result/exp{run}/ 目录。"""

import argparse
import os
import random

# 减少 TensorBoard/TensorFlow 的无关注销信息
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from copy import deepcopy
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import loaderfile
from model_file import CrossModelViews

# 路径配置：所有输出统一放在项目根目录下的 result 目录
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "5class"
RESULT_DIR = BASE_DIR / "result"

# 数据配置（与 data 目录中的图节点数保持一致）
NUM_FEATURES = 1
NUM_CLASSES = 5
NODE1_NUM = 11
NODE2_NUM = 41

# 训练配置：为了缩短运行时间，默认使用较少 epoch 和 1 次独立实验
DEFAULT_EPOCHS = 1000
DEFAULT_RUNS = 1
DEFAULT_BATCH_SIZE = 128
DEFAULT_PATIENCE = 50
VAL_INTERVAL = 1


def main(
    device,
    model,
    exp_dir,
    graph_train_loader,
    graph_val_loader,
    graph_test_loader,
    run,
    num_epochs=DEFAULT_EPOCHS,
    patience=DEFAULT_PATIENCE,
    min_epochs=0,
    min_train_acc=0.0,
):
    """训练并评估一个模型，所有输出保存到当前 run 对应的 exp_dir 中。"""
    log_dir = exp_dir / "logs"
    pred_dir = exp_dir / "pred"
    model_path = exp_dir / f"gat4_{run}.pth"

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    best_val_acc = -1.0
    best_state = deepcopy(model.state_dict())
    patience_counter = 0

    writer = SummaryWriter(log_dir=str(log_dir))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(num_epochs, 1), eta_min=1e-6
    )
    criterion = nn.CrossEntropyLoss()

    pbar = tqdm(total=num_epochs)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        pbar.set_description(f"Epoch:{epoch}")

        for data in graph_train_loader:
            batch1, batch2 = data
            optimizer.zero_grad()

            batch1 = batch1.to(device)
            batch2 = batch2.to(device)
            out = model(batch1, batch2)
            loss = criterion(out, batch1.y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += out.argmax(dim=1).eq(batch1.y).sum().item()
            total += batch1.y.size(0)

        # 验证
        if (epoch + 1) % VAL_INTERVAL == 0:
            train_Loss = total_loss / len(graph_train_loader)
            train_Acc = correct / total
            val_Acc, val_Loss = loaderfile.test(
                model, graph_val_loader, criterion, device
            )

            if val_Acc > best_val_acc:
                best_val_acc = val_Acc
                best_state = deepcopy(model.state_dict())
                patience_counter = 0
                torch.save(model.state_dict(), model_path)
            else:
                patience_counter += 1

            if patience_counter >= patience:
                # 只有达到最低 epoch 且训练精度足够时，才真正早停，避免图策略对比实验中欠拟合
                if epoch + 1 >= min_epochs and train_Acc >= min_train_acc:
                    print(
                        f"Early stopping at epoch {epoch}, best val acc: {best_val_acc:.4f}"
                    )
                    pbar.update(1)
                    break
                print(
                    f"Early stopping postponed at epoch {epoch}: "
                    f"train_acc={train_Acc:.4f} < min_train_acc={min_train_acc:.4f} "
                    f"or epoch {epoch + 1} < min_epochs={min_epochs}"
                )
                patience_counter = 0

            pbar.set_postfix(
                train_Loss=train_Loss,
                train_Acc=train_Acc,
                val_Loss=val_Loss,
                val_Acc=val_Acc,
            )
            writer.add_scalars("loss", {"train": train_Loss, "val": val_Loss}, epoch)
            writer.add_scalars("Acc", {"train": train_Acc, "val": val_Acc}, epoch)

        scheduler.step()
        pbar.update(1)

    writer.flush()
    writer.close()
    pbar.close()

    # 载入验证集上最优的模型状态
    model.load_state_dict(best_state)
    modelNew = model

    print(f"\nTesting with best model (val_acc={best_val_acc:.4f})...")
    train_acc, train_loss = loaderfile.test(
        modelNew,
        graph_train_loader,
        criterion,
        device,
        save_path=str(pred_dir / "train"),
        run=run,
    )
    val_acc, val_loss = loaderfile.test(
        modelNew,
        graph_val_loader,
        criterion,
        device,
        save_path=str(pred_dir / "val"),
        run=run,
    )
    test_acc, test_loss = loaderfile.test(
        modelNew,
        graph_test_loader,
        criterion,
        device,
        save_path=str(pred_dir / "test"),
        run=run,
    )

    print(f"Train Accuracy: {train_acc:.4f}, Train Loss: {train_loss:.4f}")
    print(f"Val Accuracy: {val_acc:.4f}, Val Loss: {val_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")
    return train_acc, train_loss, val_acc, val_loss, test_acc, test_loss


def setting(device, run):
    """创建当前 run 的 exp 目录并构建模型。"""
    exp_dir = RESULT_DIR / f"exp{run}"
    os.makedirs(exp_dir, exist_ok=True)

    mask_nodes1, mask_nodes2 = loaderfile.mask_views(
        mask_ratio1=0.2,
        mask_ratio2=0.2,
        num_nodes1=NODE1_NUM,
        num_nodes2=NODE2_NUM,
        view=4,
    )
    print(f"图1掩码节点:{mask_nodes1}")
    print(f"图2掩码节点:{mask_nodes2}")

    model = CrossModelViews.GAT_4Views_attention(
        num_features=NUM_FEATURES,
        hidden_dim=64,
        heads1=8,
        heads2=4,
        num_classes=NUM_CLASSES,
        num_expend=32,
        mask_nodes1=mask_nodes1,
        mask_nodes2=mask_nodes2,
        num_view=4,
        cross_attn_heads=8,
        node_num1=NODE1_NUM,
        node_num2=NODE2_NUM,
    ).to(device)

    return device, model, exp_dir, mask_nodes1, mask_nodes2


def Cycle(
    num_epochs=DEFAULT_EPOCHS,
    num_runs=DEFAULT_RUNS,
    batch_size=DEFAULT_BATCH_SIZE,
    patience=DEFAULT_PATIENCE,
    graph_strategy="csv",
    seed=0,
):
    """多次独立实验：每次运行的输出保存在 result/exp{run}/ 中，最终汇总保存到 result/result_TFF.txt。"""
    os.makedirs(RESULT_DIR, exist_ok=True)

    # 固定随机种子，保证结果可复现
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_accs, train_losses = [], []
    val_accs, val_losses = [], []
    test_accs, test_losses = [], []
    mask_nodes1_list, mask_nodes2_list = [], []

    # 自动递增 exp 目录编号，避免覆盖之前的运行结果
    first_exp = 0
    while (RESULT_DIR / f"exp{first_exp}").exists():
        first_exp += 1
    exp_indices = list(range(first_exp, first_exp + num_runs))

    for run in range(num_runs):
        exp_idx = exp_indices[run]
        print(f"\n第 {run + 1}/{num_runs} 次实验（结果目录: exp{exp_idx}）：")
        device, model, exp_dir, mask_nodes1, mask_nodes2 = setting(device, exp_idx)

        graph_train_loader, graph_val_loader, graph_test_loader = (
            loaderfile.data_loading(
                root=str(DATA_DIR),
                batch_size=batch_size,
                graph_strategy=graph_strategy,
                seed=seed,
            )
        )

        train_acc, train_loss, val_acc, val_loss, test_acc, test_loss = main(
            device,
            model,
            exp_dir,
            graph_train_loader,
            graph_val_loader,
            graph_test_loader,
            exp_idx,
            num_epochs=num_epochs,
            patience=patience,
        )

        train_accs.append(train_acc)
        train_losses.append(train_loss)
        val_accs.append(val_acc)
        val_losses.append(val_loss)
        test_accs.append(test_acc)
        test_losses.append(test_loss)
        mask_nodes1_list.append([m.tolist() for m in mask_nodes1])
        mask_nodes2_list.append([m.tolist() for m in mask_nodes2])

        # 当前 run 的独立结果文件
        with open(exp_dir / "result.txt", "w", encoding="utf-8") as f:
            f.write(f"run : {exp_idx}\n")
            f.write(f"Train_acc : {train_acc}\n")
            f.write(f"Train_loss : {train_loss}\n")
            f.write(f"Val_acc : {val_acc}\n")
            f.write(f"Val_loss : {val_loss}\n")
            f.write(f"Test_acc : {test_acc}\n")
            f.write(f"Test_loss : {test_loss}\n")
            f.write(f"mask_nodes1 : {[m.tolist() for m in mask_nodes1]}\n")
            f.write(f"mask_nodes2 : {[m.tolist() for m in mask_nodes2]}\n")
        print(f"本次结果已保存到 {exp_dir}")

    # 汇总文件，便于快速查看所有 run 的结果
    with open(RESULT_DIR / "result_TFF.txt", "w", encoding="utf-8") as f:
        f.write(f"exp : {exp_indices}\n")
        f.write(f"Train_acc : {train_accs}\n")
        f.write(f"Train_loss : {train_losses}\n")
        f.write(f"Val_acc : {val_accs}\n")
        f.write(f"Val_loss : {val_losses}\n")
        f.write(f"Test_acc : {test_accs}\n")
        f.write(f"Test_loss : {test_losses}\n")
        f.write(f"mask_nodes1 : {mask_nodes1_list}\n")
        f.write(f"mask_nodes2 : {mask_nodes2_list}\n")
    print(f"所有 run 的汇总结果已保存到 {RESULT_DIR / 'result_TFF.txt'}")

    n = max(num_runs, 1)
    print("\n最终平均实验结果:")
    print(
        f"Train Accuracy: {sum(train_accs)/n:.4f}, Train Loss: {sum(train_losses)/n:.4f}"
    )
    print(f"Val Accuracy: {sum(val_accs)/n:.4f}, Val Loss: {sum(val_losses)/n:.4f}")
    print(f"Test Accuracy: {sum(test_accs)/n:.4f}, Test Loss: {sum(test_losses)/n:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="双通道图注意力模型训练")
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS, help="训练 epoch 数"
    )
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="独立实验次数")
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="批大小"
    )
    parser.add_argument(
        "--patience", type=int, default=DEFAULT_PATIENCE, help="早停耐心值"
    )
    parser.add_argument(
        "--graph",
        type=str,
        default="csv",
        choices=[
            "csv",
            "lingam",
            "pearson",
            "mi",
            "partial",
            "random",
            "complete",
            "empty",
        ],
        help="图构造策略，默认 csv 使用 data/edges1.csv 和 edges2.csv",
    )
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    args = parser.parse_args()

    Cycle(
        num_epochs=args.epochs,
        num_runs=args.runs,
        batch_size=args.batch_size,
        patience=args.patience,
        graph_strategy=args.graph,
        seed=args.seed,
    )
