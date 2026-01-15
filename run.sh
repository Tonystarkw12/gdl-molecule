#!/bin/bash

# 快速启动脚本 - SchNet 分子性质预测

echo "========================================"
echo "  SchNet 分子性质预测 - 快速启动"
echo "========================================"
echo ""

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到 Python3"
    exit 1
fi

# 检查 CUDA
echo "🔍 检查 CUDA 可用性..."
python3 -c "import torch; print(f'CUDA 可用: {torch.cuda.is_available()}')"

echo ""
echo "请选择运行模式:"
echo "  1) 安装依赖环境"
echo "  2) 训练模型 (gap, 50 epochs)"
echo "  3) 训练模型 (U0, 50 epochs)"
echo "  4) 训练模型 (gap, 100 epochs, 慢但效果更好)"
echo ""
read -p "输入选项 [1-4]: " choice

case $choice in
    1)
        echo ""
        echo "📦 安装依赖..."
        echo "方法 A: Conda (推荐)"
        echo "  conda env create -f environment.yml"
        echo "  conda activate gdl_molecule"
        echo ""
        echo "方法 B: pip"
        echo "  pip install -r requirements.txt"
        echo ""
        ;;
    2)
        echo ""
        echo "🚀 开始训练 (gap, 50 epochs)..."
        python3 train.py --target gap --epochs 50 --batch-size 32
        ;;
    3)
        echo ""
        echo "🚀 开始训练 (U0, 50 epochs)..."
        python3 train.py --target U0 --epochs 50 --batch-size 32
        ;;
    4)
        echo ""
        echo "🚀 开始训练 (gap, 100 epochs)..."
        python3 train.py --target gap --epochs 100 --batch-size 32 --lr 1e-4
        ;;
    *)
        echo "❌ 无效选项"
        exit 1
        ;;
esac

echo ""
echo "✓ 完成!"
