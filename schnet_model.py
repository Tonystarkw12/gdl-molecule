"""
SchNet: 深度连续滤波卷积神经网络用于量子化学
SchNet: Deep Continuous Filter Convolutional Neural Networks for Quantum Chemistry

基于论文: SchNet et al., 2018 (https://arxiv.org/abs/1706.08566)

核心组件：
1. Interaction Block: 连续滤波卷积
2. Filter Network: 高斯径向基函数(RBF)扩展的距离
3. Atom-wise Layers: 原子特征更新
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_add_pool
from math import pi as PI


class ShiftedSoftplus(nn.Module):
    """ShiftedSoftplus 激活函数

    softplus(x) = log(1 + exp(x))
    shifted_softplus(x) = softplus(x) - log(2)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.softplus(x) - torch.log(torch.tensor(2.0))


class GaussianSmearing(nn.Module):
    """将原子间距离扩展为高斯径向基函数

    Args:
        start: 距离下限
        stop: 距离上限
        n_gaussians: 高斯函数数量
    """
    def __init__(self, start=0.0, stop=5.0, n_gaussians=50):
        super().__init__()
        self.start = start
        self.stop = stop
        self.n_gaussians = n_gaussians

        # 计算高斯中心点和宽度 - 注册为 buffer 以自动移到正确设备
        self.register_buffer('offset', torch.linspace(start, stop, n_gaussians))
        self.register_buffer('coeff', torch.tensor(-0.5 / ((stop - start) / (n_gaussians - 1)) ** 2))

    def forward(self, dist):
        """计算高斯径向基函数

        Args:
            dist: 原子间距离 [N_edges, 1]

        Returns:
            高斯RBF特征 [N_edges, n_gaussians]
        """
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * dist ** 2)


class CFConv(MessagePassing):
    """连续滤波卷积层 (Continuous Filter Convolution)

    核心思想：可学习滤波器 W(d) 是原子间距离 d 的连续函数

    操作流程：
    1. 计算原子对之间的距离 d_ij
    2. 通过高斯RBF将距离扩展为高维特征 φ(d_ij)
    3. 通过滤波网络生成距离依赖的滤波器 W(d_ij)
    4. 应用滤波器：x_i = Σ_j W(d_ij) * x_j

    Args:
        in_channels: 输入特征维度
        out_channels: 输出特征维度
        num_gaussians: 高斯RBF数量
        hidden_channels: 隐藏层维度
    """
    def __init__(self, in_channels, out_channels, num_gaussians, hidden_channels):
        super().__init__(aggr='add')  # 加法聚合
        self.gaussians = GaussianSmearing(n_gaussians=num_gaussians)

        # 滤波网络：将距离映射到滤波器权重
        # W(d) = MLP(φ(d))  其中 φ 是高斯RBF
        self.filter_network = nn.Sequential(
            nn.Linear(num_gaussians, hidden_channels),
            ShiftedSoftplus(),
            nn.Linear(hidden_channels, hidden_channels),
            ShiftedSoftplus(),
            nn.Linear(hidden_channels, in_channels * out_channels)
        )

        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x: 节点特征 [N_nodes, in_channels]
            edge_index: 边索引 [2, N_edges]
            edge_attr: 边属性（距离）[N_edges, 1]

        Returns:
            更新后的节点特征 [N_nodes, out_channels]
        """
        # 计算距离依赖的滤波器 W(d)
        # [N_edges, n_gaussians] -> [N_edges, in_channels * out_channels]
        filter = self.filter_network(self.gaussians(edge_attr))

        # reshape 为 [N_edges, in_channels, out_channels]
        filter = filter.view(-1, self.in_channels, self.out_channels)

        # 应用连续滤波卷积
        # x_j: [N_edges, in_channels] (邻居节点特征)
        # filter * x_j: [N_edges, out_channels]
        # message: [N_edges, out_channels]
        out = self.propagate(edge_index, x=x, filter=filter)

        return out

    def message(self, x_j, filter):
        """消息函数：应用滤波器到邻居特征

        Args:
            x_j: 邻居节点特征 [N_edges, in_channels]
            filter: 距离依赖的滤波器 [N_edges, in_channels, out_channels]

        Returns:
            加权后的消息 [N_edges, out_channels]
        """
        # 滤波器加权：W(d_ij) * x_j
        # [N_edges, in_channels, out_channels] @ [N_edges, in_channels, 1]
        # 使用 einsum 进行批量的矩阵-向量乘法
        return torch.einsum('eij,ej->ei', filter, x_j)


class InteractionBlock(nn.Module):
    """SchNet 交互块

    结构：
    1. 连续滤波卷积 (CFConv)
    2. 原子级前馈网络 (Atom-wise FFN)

    Args:
        hidden_channels: 隐藏层维度
        num_gaussians: 高斯RBF数量
        num_filters: 滤波网络隐藏层维度
    """
    def __init__(self, hidden_channels, num_gaussians, num_filters):
        super().__init__()

        # 连续滤波卷积
        self.cfconv = CFConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            num_gaussians=num_gaussians,
            hidden_channels=num_filters
        )

        # 原子级前馈网络
        self.atom_wise1 = nn.Linear(hidden_channels, hidden_channels)
        self.atom_wise2 = nn.Linear(hidden_channels, hidden_channels)

    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x: 节点特征 [N_nodes, hidden_channels]
            edge_index: 边索引
            edge_attr: 边属性（距离）

        Returns:
            更新后的节点特征 [N_nodes, hidden_channels]
        """
        # 残差连接的初始值
        residual = x

        # 连续滤波卷积
        x = self.cfconv(x, edge_index, edge_attr)

        # 原子级前馈 + 残差连接
        x = F.silu(self.atom_wise1(x))
        x = self.atom_wise2(x)
        x = x + residual

        return x


class SchNet(nn.Module):
    """SchNet 分子性质预测模型

    架构流程：
    1. Embedding: 原子序数 z -> 初始特征 x
    2. N x InteractionBlock: 迭代更新原子特征（考虑3D几何）
    3. Global Pooling: 聚合分子级表示
    4. Prediction: MLP -> 目标性质

    Args:
        hidden_channels: 隐藏层维度 (默认: 128)
        num_filters: 滤波网络隐藏层维度 (默认: 128)
        num_interactions: 交互块数量 (默认: 3)
        num_gaussians: 高斯RBF数量 (默认: 50)
        cutoff: 距离截断值
        atom_types: 原子类型数量 (QM9: 1-5,6-9号元素)
    """
    def __init__(
        self,
        hidden_channels=128,
        num_filters=128,
        num_interactions=3,
        num_gaussians=50,
        cutoff=5.0,
        atom_types=100
    ):
        super().__init__()

        # 原子嵌入层
        self.embedding = nn.Embedding(atom_types, hidden_channels)

        # 交互块堆叠
        self.interactions = nn.ModuleList([
            InteractionBlock(hidden_channels, num_gaussians, num_filters)
            for _ in range(num_interactions)
        ])

        # 输出层（分子级）
        self.output1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.output2 = nn.Linear(hidden_channels // 2, 1)

        self.num_gaussians = num_gaussians
        self.cutoff = cutoff

    def forward(self, z, pos, edge_index, batch=None):
        """
        Args:
            z: 原子序数 [N_nodes]
            pos: 3D坐标 [N_nodes, 3]
            edge_index: 边索引 [2, N_edges]
            batch: 节点到图索引 [N_nodes]

        Returns:
            预测值 [N_graphs, 1]
        """
        # 原子嵌入
        x = self.embedding(z)

        # 计算边属性（欧几里得距离）
        # [N_edges, 3] -> [N_edges, 1]
        edge_attr = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=-1, keepdim=True)

        # 通过交互块堆叠
        for interaction in self.interactions:
            x = interaction(x, edge_index, edge_attr)

        # 全局池化（求和聚合）
        if batch is None:
            batch = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
        x = global_add_pool(x, batch)

        # 预测层
        x = F.silu(self.output1(x))
        x = self.output2(x)

        return x

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'hidden_channels={self.hidden_channels}, '
                f'num_filters={self.num_filters}, '
                f'num_interactions={self.num_interactions})')
