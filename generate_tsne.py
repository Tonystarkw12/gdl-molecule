"""
生成 t-SNE 可视化（独立脚本）
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import QM9

from schnet_model import SchNet


def get_qm9_dataset(target_idx, root='./data/qm9', subset_size=None, transform=None):
    """加载 QM9 数据集"""
    dataset = QM9(root=root, transform=transform)
    dataset.data.y = dataset.data.y[:, target_idx].unsqueeze(1)

    if subset_size is not None and subset_size < len(dataset):
        indices = torch.randperm(len(dataset))[:subset_size]
        dataset = dataset[indices]

    return dataset


class TransformOptimized(object):
    """数据预处理：基于距离阈值创建边"""
    def __init__(self, cutoff=5.0):
        self.cutoff = cutoff

    def __call__(self, data):
        pos = data.pos
        num_nodes = pos.size(0)
        dist_matrix = torch.cdist(pos, pos, p=2)
        mask = (dist_matrix <= self.cutoff) & (dist_matrix > 0)
        edge_index = torch.nonzero(mask, as_tuple=False).t()

        if edge_index.size(1) == 0:
            edge_index = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)])

        data.edge_index = edge_index
        return data


@torch.no_grad()
def get_embeddings(model, loader, device):
    """获取分子嵌入"""
    model.eval()
    embeddings = []
    targets = []

    import torch_geometric.nn as geom_nn

    for batch in loader:
        batch = batch.to(device)

        # 获取嵌入
        x = model.embedding(batch.z)
        for interaction in model.interactions:
            edge_attr = torch.norm(
                batch.pos[batch.edge_index[0]] - batch.pos[batch.edge_index[1]],
                dim=-1, keepdim=True
            )
            x = interaction(x, batch.edge_index, edge_attr)

        mol_embedding = geom_nn.global_add_pool(x, batch.batch)
        embeddings.append(mol_embedding.cpu().numpy())
        targets.append(batch.y.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    targets = np.concatenate(targets, axis=0).flatten()

    return embeddings, targets


def plot_tsne(embeddings, targets, save_path='./results/tsne_lite.png'):
    """t-SNE 可视化分子嵌入"""
    print("正在生成 t-SNE 可视化...")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 降维到 2D（兼容新旧版本scikit-learn）
    try:
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    except TypeError:
        # 新版本使用 max_iter 而不是 n_iter
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1],
        c=targets, cmap='viridis', alpha=0.6, s=20
    )
    plt.colorbar(scatter, label='Target Value')
    plt.xlabel('t-SNE Dimension 1', fontsize=14)
    plt.ylabel('t-SNE Dimension 2', fontsize=14)
    plt.title('t-SNE Visualization of Molecular Embeddings (Lite Model)', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ t-SNE 可视化已保存至: {save_path}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("加载测试数据...")
    transform = TransformOptimized()
    dataset = get_qm9_dataset(target_idx=4, root='./data/qm9', subset_size=10000, transform=transform)

    # 使用测试集
    _, _, test_dataset = torch.utils.data.random_split(
        dataset, [8000, 1000, 1000]
    )

    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    print("加载模型...")
    model = SchNet(
        hidden_channels=64,
        num_filters=64,
        num_interactions=2,
        num_gaussians=25,
        cutoff=5.0,
        atom_types=100
    ).to(device)

    model.load_state_dict(torch.load('./checkpoints/best_model_lite.pth'))
    print("✓ 模型加载成功")

    print("生成嵌入...")
    embeddings, targets = get_embeddings(model, test_loader, device)

    print("生成 t-SNE 可视化...")
    plot_tsne(embeddings, targets)

    print("\n✓ t-SNE 可视化完成!")


if __name__ == '__main__':
    main()
