"""
SchNet 分子性质预测训练脚本
目标：预测 QM9 数据集中的 HOMO-LUMO gap (gap) 或 内能 (U0)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import QM9
import torch_geometric.nn as geom_nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
import argparse

from schnet_model import SchNet


# ============================ 配置 ============================
class Config:
    # 数据配置
    target_idx = 4  # QM9 targets: 4='gap' (HOMO-LUMO), 7='U0' (Internal energy)
    dataset_root = './data/qm9'

    # 模型超参数
    hidden_channels = 128
    num_filters = 128
    num_interactions = 3
    num_gaussians = 50
    cutoff = 5.0
    atom_types = 100  # QM9 原子类型覆盖

    # 训练超参数
    batch_size = 32
    num_epochs = 50
    learning_rate = 1e-4
    weight_decay = 1e-5

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 随机种子
    seed = 42


# ============================ 数据处理 ============================
def get_qm9_dataset(target_idx, root='./data/qm9', transform=None):
    """加载 QM9 数据集

    Args:
        target_idx: 目标属性索引 (4='gap', 7='U0')
        root: 数据存储路径
        transform: 数据转换函数
    """
    print(f"正在加载 QM9 数据集，目标索引: {target_idx}...")

    # QM9 targets: [0:mu, 1:alpha, 2:homo, 3:lumo, 4:gap, 5:r2, 6:zpve, 7:U0, 8:U, 9:H, 10:G, 11:Cv]
    dataset = QM9(root=root, transform=transform)

    # 提取目标属性
    dataset.data.y = dataset.data.y[:, target_idx].unsqueeze(1)

    print(f"数据集大小: {len(dataset)} 个分子")
    print(f"目标属性范围: [{dataset.data.y.min().item():.4f}, {dataset.data.y.max().item():.4f}]")

    return dataset


class Transform(object):
    """数据预处理：基于距离阈值创建边（减少显存占用）"""
    def __init__(self, cutoff=5.0):
        self.cutoff = cutoff

    def __call__(self, data):
        # 基于欧几里得距离创建边（更稀疏的图）
        pos = data.pos
        num_nodes = pos.size(0)

        # 计算所有原子对之间的距离
        edge_index = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                # 计算距离
                dist = torch.norm(pos[i] - pos[j])
                if dist.item() <= self.cutoff:
                    # 添加双向边
                    edge_index.append([i, j])
                    edge_index.append([j, i])

        if edge_index:
            data.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        else:
            # 如果没有边，创建自环
            data.edge_index = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)])

        return data


# ============================ 训练与评估 ============================
def train_epoch(model, loader, optimizer, criterion, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    count = 0

    pbar = tqdm(loader, desc="训练")
    for batch in pbar:
        batch = batch.to(device)

        # 前向传播
        optimizer.zero_grad()
        pred = model(
            z=batch.z,
            pos=batch.pos,
            edge_index=batch.edge_index,
            batch=batch.batch
        )

        # 计算损失
        loss = criterion(pred, batch.y)

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        count += batch.num_graphs

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / count


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    embeddings = []

    for batch in tqdm(loader, desc="评估"):
        batch = batch.to(device)

        # 前向传播（获取预测和嵌入）
        pred = model(
            z=batch.z,
            pos=batch.pos,
            edge_index=batch.edge_index,
            batch=batch.batch
        )

        loss = criterion(pred, batch.y)
        total_loss += loss.item() * batch.num_graphs

        predictions.append(pred.cpu().numpy())
        targets.append(batch.y.cpu().numpy())

        # 获取分子嵌入（用于可视化）
        # 重新运行前向传播并获取中间层输出
        x = model.embedding(batch.z)
        for interaction in model.interactions:
            edge_attr = torch.norm(
                batch.pos[batch.edge_index[0]] - batch.pos[batch.edge_index[1]],
                dim=-1, keepdim=True
            )
            x = interaction(x, batch.edge_index, edge_attr)

        mol_embedding = geom_nn.global_add_pool(x, batch.batch)
        embeddings.append(mol_embedding.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0).flatten()
    targets = np.concatenate(targets, axis=0).flatten()
    embeddings = np.concatenate(embeddings, axis=0)

    # 计算 MAE 和 RMSE
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))

    return total_loss / len(loader.dataset), mae, rmse, predictions, targets, embeddings


# ============================ 可视化 ============================
def plot_predictions(predictions, targets, save_path='./results/predictions.png'):
    """绘制预测值 vs 真实值散点图"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 8))
    plt.scatter(targets, predictions, alpha=0.3, s=10)

    # 对角线（完美预测线）
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

    # 计算并显示 R²
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    mae = np.mean(np.abs(predictions - targets))

    plt.xlabel('True Values', fontsize=14)
    plt.ylabel('Predicted Values', fontsize=14)
    plt.title(f'Prediction Results\nR² = {r2:.4f}, MAE = {mae:.4f}', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ 预测散点图已保存至: {save_path}")


def plot_loss_curve(train_losses, val_losses, save_path='./results/loss_curve.png'):
    """绘制训练损失曲线"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, 'b-o', label='Train Loss', markersize=4)
    plt.plot(epochs, val_losses, 'r-s', label='Val Loss', markersize=4)

    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss (MSE)', fontsize=14)
    plt.title('Training Progress', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ 损失曲线已保存至: {save_path}")


def plot_tsne(embeddings, targets, save_path='./results/tsne.png'):
    """t-SNE 可视化分子嵌入"""
    print("正在生成 t-SNE 可视化...")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 降维到 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1],
        c=targets, cmap='viridis', alpha=0.6, s=20
    )
    plt.colorbar(scatter, label='Target Value')
    plt.xlabel('t-SNE Dimension 1', fontsize=14)
    plt.ylabel('t-SNE Dimension 2', fontsize=14)
    plt.title('t-SNE Visualization of Molecular Embeddings', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ t-SNE 可视化已保存至: {save_path}")


# ============================ 主函数 ============================
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='SchNet Molecular Property Prediction')
    parser.add_argument('--target', type=str, default='gap', choices=['gap', 'U0'],
                        help='Target property (gap or U0)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    args = parser.parse_args()

    # 设置目标索引
    target_idx = 4 if args.target == 'gap' else 7

    # 设置随机种子
    torch.manual_seed(Config.seed)
    np.random.seed(Config.seed)

    # 打印配置
    print("=" * 60)
    print("SchNet 分子性质预测")
    print("=" * 60)
    print(f"目标属性: {args.target} (索引: {target_idx})")
    print(f"设备: {Config.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print("=" * 60)

    # 加载数据
    print("\n[1/5] 加载数据集...")
    transform = Transform()
    dataset = get_qm9_dataset(target_idx, Config.dataset_root, transform)

    # 划分数据集
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    print(f"训练集: {train_size}, 验证集: {val_size}, 测试集: {test_size}")

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 初始化模型
    print("\n[2/5] 初始化 SchNet 模型...")
    model = SchNet(
        hidden_channels=Config.hidden_channels,
        num_filters=Config.num_filters,
        num_interactions=Config.num_interactions,
        num_gaussians=Config.num_gaussians,
        cutoff=Config.cutoff,
        atom_types=Config.atom_types
    ).to(Config.device)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=Config.weight_decay)
    criterion = nn.MSELoss()

    # 训练循环
    print("\n[3/5] 开始训练...")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 40)

        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, criterion, Config.device)
        train_losses.append(train_loss)

        # 验证
        val_loss, val_mae, val_rmse, _, _, _ = evaluate(
            model, val_loader, criterion, Config.device
        )
        val_losses.append(val_loss)

        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss: {val_loss:.6f} | MAE: {val_mae:.6f} | RMSE: {val_rmse:.6f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            os.makedirs('./checkpoints', exist_ok=True)
            torch.save(model.state_dict(), './checkpoints/best_model.pth')
            print(f"✓ 最佳模型已保存 (Epoch {epoch})")

    print("\n训练完成!")
    print(f"最佳验证损失: {best_val_loss:.6f} (Epoch {best_epoch})")

    # 测试集评估
    print("\n[4/5] 测试集评估...")
    model.load_state_dict(torch.load('./checkpoints/best_model.pth'))
    test_loss, test_mae, test_rmse, predictions, targets, embeddings = evaluate(
        model, test_loader, criterion, Config.device
    )

    print(f"Test Loss: {test_loss:.6f}")
    print(f"Test MAE: {test_mae:.6f}")
    print(f"Test RMSE: {test_rmse:.6f}")

    # 可视化结果
    print("\n[5/5] 生成可视化...")
    plot_predictions(predictions, targets)
    plot_loss_curve(train_losses, val_losses)
    plot_tsne(embeddings, targets)

    print("\n" + "=" * 60)
    print("✓ 所有任务完成!")
    print(f"✓ 模型检查点: ./checkpoints/best_model.pth")
    print(f"✓ 可视化结果: ./results/")
    print("=" * 60)


if __name__ == '__main__':
    main()
