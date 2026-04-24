# Chronos Teacher 卡顿问题排查指南

## 问题诊断
脚本在 `Beginning AutoGluon training...` 后卡住，可能原因：
1. **日志缓冲** - AutoGluon的输出被缓冲，看不到实时进度
2. **模型微调缓慢** - 3000个微调步骤需要较长时间
3. **网络问题** - 首次下载Chronos模型较慢

## 改进措施
已对 `generate_chronos_teacher.py` 进行以下改进：

### 1. **实时日志输出**
- 配置Python的logging模块，确保日志实时输出到stdout
- 在关键步骤添加进度日志，包括：
  - fit()开始前的详细参数信息
  - fit()完成时的耗时统计
  - 每个batch的预测进度
- 使用 `sys.stdout.flush()` 确保日志立即显示

### 2. **更详细的进度信息**
现在会输出：
```
========================================
Starting AutoGluon fit() with Chronos teacher model
  Fine-tune: True
  Fine-tune steps: 3000
  Batch size: 8
  Train data shape: (5391856, 8)
  Val data shape: (6932384, 8)
========================================
[实时微调进度]
AutoGluon fit completed in XXX.X seconds
```

### 3. **错误处理**
- 捕获fit()过程中的异常
- 完整输出错误堆栈信息用于调试

## 使用建议

### 建议 1: 使用合理的time-limit
```bash
# 原始命令（无限期等待，可能卡住看起来）
--teacher-time-limit 0

# 建议改为（1小时超时，自动保存中间结果）
--teacher-time-limit 3600
```

### 建议 2: 在AutoDL上使用daemon模式
根据 https://www.autodl.com/docs/daemon/，添加断点续传支持：

```bash
# 在脚本开始时添加daemon检测
# 这样中断后可以自动恢复
```

### 建议 3: 监控资源使用
```bash
# 在另一个终端监控GPU/内存
nvidia-smi -l 1
top -p $(pgrep -f generate_chronos_teacher)
```

### 建议 4: 添加检查点（可选）
如果需要支持中断续传，可以：
1. 在输出目录检查现有的 `teacher_model` 目录
2. 如果存在，直接使用 `--pretrained-teacher-path` 加载
3. 从该检查点继续生成预测

示例：
```bash
python generate_chronos_teacher.py \
  [... 其他参数 ...] \
  --pretrained-teacher-path /root/autodl-tmp/chronos_teacher_6h/teacher_model \
  --short-horizon-hours 6
```

## 调试步骤

### Step 1: 运行改进后的脚本
```bash
python /root/autodl-tmp/generate_chronos_teacher.py \
  --gpu-id 0 \
  --input-dir /root/autodl-tmp/processed_csv/aligned_stations \
  --metadata-path /root/autodl-tmp/processed_csv/shared_timeline_metadata.json \
  --output-root /root/autodl-tmp/chronos_teacher_6h \
  --short-horizon-hours 6 \
  --teacher-time-limit 3600 \
  --verbosity 3
```

**注意**：移除 `--teacher-time-limit 0`，改为 `--teacher-time-limit 3600`

### Step 2: 观察日志输出
现在应该看到实时的进度信息。如果仍然卡住，查看日志中的最后一条消息指示在哪一步卡住。

### Step 3: 根据日志调整参数
- 如果在fit()阶段超时，增加 `--teacher-time-limit`
- 如果在预测阶段缓慢，增加 `--teacher-batch-size` 以加速处理

## 性能优化建议

### 减少训练时间
```bash
# 减少微调步骤数
--teacher-fine-tune-steps 1000  # 默认3000

# 增加eval_steps间隔（减少验证频率）
--teacher-eval-steps 200  # 默认100

# 增加gradient accumulation（有助于稳定性）
--teacher-gradient-accumulation 8  # 默认4
```

### 加速预测生成
```bash
# 增加batch size（如果GPU内存充足）
--teacher-batch-size 16  # 默认8

# 减少最大窗口数
--train-max-windows-per-station 128  # 默认256
--val-max-windows-per-station 32     # 默认64
--test-max-windows-per-station 32    # 默认64
```

## 故障排除

| 现象 | 原因 | 解决方案 |
|------|------|--------|
| 卡在"fit()"阶段 | AutoGluon微调很慢 | 增加time-limit或检查GPU使用率 |
| CUDA Out of Memory | batch size太大 | 减少 `--teacher-batch-size` |
| 网络下载缓慢 | 首次下载模型 | 使用 HF_ENDPOINT 加速 |
| 中断后需重新训练 | 没有保存检查点 | 使用 `--pretrained-teacher-path` 加载已训练模型 |

## 预期耗时

- **模型下载**: 5-10分钟（首次）
- **模型微调**: 30-60分钟（取决于fine-tune-steps）
- **预测生成**: 10-30分钟（取决于窗口数）
- **总耗时**: 45m-90m

## 文件输出

完成后会在 `--output-root` 目录生成：
```
chronos_teacher_6h/
├── teacher_model/          # AutoGluon模型（可用于恢复）
├── teacher_log.json        # 完整执行日志
├── teacher_forecasts_train.csv
├── teacher_forecasts_val.csv
├── teacher_forecasts_test.csv
└── window_index_*.csv      # 窗口索引
```

其中 `teacher_model/` 可用作下一次运行的 `--pretrained-teacher-path`。
