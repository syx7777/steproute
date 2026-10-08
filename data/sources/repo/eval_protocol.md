---
title: 评测协议 Evaluation Protocol
date: 2025-09-02
author: 苏甬新
---

# 公平对比的三条硬性要求
1. 抽帧数（frames）与输入分辨率（resolution）必须一致，否则指标不可比。
2. 固定随机种子 seed=42，并固定评测切分 VideoVQA-mini/val。
3. 推理延迟必须在同一硬件（A100 单卡）、同一 batch_size 下测量。

# 常见坑
- 用 16 帧 / 224 的结果去和 32 帧 / 336 的 baseline 比准确率，是最容易犯的错误。
- 延迟看起来更优，往往只是因为输入更小，而不是方法更高效。