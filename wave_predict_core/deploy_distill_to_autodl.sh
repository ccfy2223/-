#!/bin/bash
# deploy_distill_to_autodl.sh
# 将知识蒸馏脚本部署到 AutoDL 并运行

echo "=========================================="
echo "部署知识蒸馏脚本到 AutoDL"
echo "=========================================="

# 1. 上传脚本到 AutoDL
echo ""
echo "[1/3] 上传脚本文件..."
echo "请手动执行以下命令上传文件："
echo ""
echo "方法1：使用 AutoDL 文件管理器"
echo "  - 打开 AutoDL 控制台 -> JupyterLab"
echo "  - 上传 train_distill_batch.py 到 /root/"
echo ""
echo "方法2：使用 scp（如果有 SSH 访问）"
echo "  scp train_distill_batch.py root@your-autodl-ip:/root/"
echo ""
echo "方法3：使用 git（推荐）"
echo "  # 在本地"
echo "  git add train_distill_batch.py"
echo "  git commit -m 'Add distillation script'"
echo "  git push"
echo ""
echo "  # 在 AutoDL"
echo "  cd /root/autodl-tmp"
echo "  git pull"
echo ""

# 2. 在 AutoDL 上运行的命令
cat > /tmp/autodl_run_distill.sh << 'EOF'
#!/bin/bash
# 在 AutoDL 上执行此脚本

cd /root

# 检查软标签是否存在
echo "检查软标签文件..."
if [ ! -f "/root/sota_runs/chronos_fullseq_labels/labels_024h.csv" ]; then
    echo "错误：软标签文件不存在！"
    echo "请先运行 generate_chronos_fullseq_labels.py"
    exit 1
fi

echo "软标签文件检查通过"
ls -lh /root/sota_runs/chronos_fullseq_labels/

# 运行蒸馏实验
echo ""
echo "=========================================="
echo "开始知识蒸馏实验 - 24h Horizon"
echo "=========================================="

python /root/train_distill_batch.py \
    --input-dir /root/autodl-tmp/processed_csv/aligned_stations \
    --soft-labels-dir /root/sota_runs/chronos_fullseq_labels \
    --metadata-path /root/autodl-tmp/processed_csv/shared_timeline_metadata.json \
    --output-dir /root/sota_runs/distill_runs \
    --horizon-hours 24 \
    --alphas 0.3,0.5,0.7 \
    --time-limit 3600 \
    --gpu-id 0

echo ""
echo "=========================================="
echo "实验完成！"
echo "=========================================="
echo "结果保存在："
echo "  /root/sota_runs/distill_runs/horizon_024h/"
ls -lh /root/sota_runs/distill_runs/horizon_024h/
EOF

echo ""
echo "[2/3] 生成 AutoDL 运行脚本"
echo "已生成：/tmp/autodl_run_distill.sh"
echo ""

# 3. 显示完整步骤
cat << 'EOF'

========================================
完整部署步骤
========================================

步骤1：上传脚本到 AutoDL
------------------------
在本地执行：

# 方法A：使用 AutoDL JupyterLab 文件上传
1. 打开 AutoDL 控制台
2. 点击 "JupyterLab"
3. 在左侧文件浏览器中，上传以下文件到 /root/：
   - train_distill_batch.py

# 方法B：使用命令行（如果有 SSH）
scp train_distill_batch.py root@your-autodl-ip:/root/


步骤2：在 AutoDL 上运行
------------------------
在 AutoDL 终端执行：

cd /root

# 检查文件是否上传成功
ls -lh train_distill_batch.py

# 检查软标签是否存在
ls -lh /root/sota_runs/chronos_fullseq_labels/labels_024h.csv

# 运行蒸馏实验
python train_distill_batch.py \
    --input-dir /root/autodl-tmp/processed_csv/aligned_stations \
    --soft-labels-dir /root/sota_runs/chronos_fullseq_labels \
    --metadata-path /root/autodl-tmp/processed_csv/shared_timeline_metadata.json \
    --output-dir /root/sota_runs/distill_runs \
    --horizon-hours 24 \
    --alphas 0.3,0.5,0.7 \
    --time-limit 3600 \
    --gpu-id 0


步骤3：监控进度
------------------------
# 查看输出
tail -f nohup.out

# 或者在后台运行
nohup python train_distill_batch.py \
    --input-dir /root/autodl-tmp/processed_csv/aligned_stations \
    --soft-labels-dir /root/sota_runs/chronos_fullseq_labels \
    --metadata-path /root/autodl-tmp/processed_csv/shared_timeline_metadata.json \
    --output-dir /root/sota_runs/distill_runs \
    --horizon-hours 24 \
    --alphas 0.3,0.5,0.7 \
    --time-limit 3600 \
    --gpu-id 0 > distill_24h.log 2>&1 &

# 查看日志
tail -f distill_24h.log


步骤4：检查结果
------------------------
# 查看生成的模型
ls -lh /root/sota_runs/distill_runs/horizon_024h/

# 查看汇总结果
cat /root/sota_runs/distill_runs/horizon_024h/summary.json

# 查看各个 α 的结果
cat /root/sota_runs/distill_runs/horizon_024h/alpha_0.5/result.json


========================================
预计时间
========================================
- 单个 α 值：约 1 小时
- 3 个 α 值：约 3 小时
- 8 个 horizon × 3 个 α：约 24 小时

========================================
故障排查
========================================

问题1：软标签文件不存在
解决：等待 generate_chronos_fullseq_labels.py 完成

问题2：内存不足
解决：减少 batch_size 或 limit_stations

问题3：GPU 内存不足
解决：减少 context_length 或 d_model

EOF

echo ""
echo "[3/3] 部署指南已生成"
echo ""
