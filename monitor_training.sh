#!/bin/bash

echo "========== 训练进度监控 =========="
echo "开始时间: $(date)"
echo ""

# 每 30 秒检查一次，最多检查 60 次（30分钟）
for i in {1..60}; do
    clear
    echo "========== 训练进度监控 [$i/60] =========="
    echo "当前时间: $(date)"
    echo ""

    # 检查进程
    if ps aux | grep -v grep | grep "python train.py" > /dev/null; then
        echo "✓ 训练进程正在运行"

        # 显示进程信息
        ps aux | grep "python train.py" | grep -v grep | grep -v conda | awk '{printf "  CPU: %s%%, 内存: %s%%, 运行时间: %s\n", $3, $4, $10}'

        # 显示 GPU 使用情况
        if command -v nvidia-smi &> /dev/null; then
            echo ""
            nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader | head -1 | awk '{printf "  GPU: %s 使用率, %s 显存使用, 温度: %s°C\n", $1, $2, $4}'
        fi

        # 检查结果文件
        echo ""
        echo "结果文件:"
        if [ -f "./checkpoints/best_model.pth" ]; then
            echo "  ✓ 模型检查点已创建"
            ls -lh ./checkpoints/best_model.pth | awk '{print "    大小: "$5", 修改时间: "$6" "$7" "$8}'
        else
            echo "  ✗ 模型检查点未生成"
        fi

        if [ -f "./results/predictions.png" ]; then
            echo "  ✓ 预测散点图已生成"
        fi

        if [ -f "./results/loss_curve.png" ]; then
            echo "  ✓ 损失曲线已生成"
        fi

        if [ -f "./results/tsne.png" ]; then
            echo "  ✓ t-SNE 可视化已生成"
        fi

    else
        echo "✗ 训练进程已结束"
        echo ""
        echo "最终结果:"
        ls -lh ./results/ ./checkpoints/ 2>/dev/null
        break
    fi

    echo ""
    echo "下次检查: 30 秒后..."
    sleep 30
done

echo ""
echo "========== 监控结束 =========="
