---
title: 评测协议 Evaluation Protocol
date: 2026-03-02
author: 示例作者
---

# 公平对比的三条硬性要求
1. 输入分辨率（resolution）与训练数据（train_data）必须一致，否则指标不可比。
2. 固定随机种子 seed=42，并固定评测切分 DemoSet-1k/val。
3. 推理延迟必须在同一硬件、同一 batch_size 下测量。

# 常见坑
- 用 160 分辨率 / 子集训练的结果去和 224 分辨率的 baseline 比准确率，是最容易犯的错误。
- 延迟看起来更优，往往只是因为输入更小，而不是方法更高效。