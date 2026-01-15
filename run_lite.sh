#!/bin/bash

# 轻量级训练脚本
# 适合 8GB 显存 GPU 的快速训练

echo "=========================================="
echo "SchNet 分子性质预测 - 轻量级训练"
echo "=========================================="
echo ""
echo "配置说明："
echo "  - 数据集: 10,000 分子 (7.5%)"
echo "  - 模型参数: 约 200k (轻量级)"
echo "  - 显存占用: 约 5-6 GB"
echo "  - 预计时间: 30-45 分钟"
echo ""
echo "=========================================="
echo ""

# 激活 conda 环境
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "✓ 当前 conda 环境: $CONDA_DEFAULT_ENV"
else
    echo "⚠ 未检测到 conda 环境，请确保已激活 gdl_molecule 环境"
    echo "  运行: conda activate gdl_molecule"
fi

echo ""
echo "开始训练..."
echo ""

# 运行轻量级训练
python train_lite.py \
    --target gap \
    --epochs 20 \
    --batch-size 16 \
    --lr 1e-4 \
    --subset-size 10000

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
echo ""
echo "结果文件："
echo "  - 模型检查点: ./checkpoints/best_model_lite.pth"
echo "  - 预测散点图: ./results/predictions_lite.png"
echo "  - 损失曲线: ./results/loss_curve_lite.png"
echo "  - t-SNE可视化: ./results/tsne_lite.png"
echo ""
