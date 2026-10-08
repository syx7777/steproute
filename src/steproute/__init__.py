"""科研工作流 Agent Harness —— 面向多步 Agent 的动态模型路由与调度。

核心思想
--------
1. **Step-level Routing**：路由粒度是 Agent 的单个步骤，而不是用户最初的 Query。
2. **两级决策**：
   * ``Jev``：判断当前步骤**最低需要**哪种能力（fast / balanced / strong）；
   * ``RuntimeScheduler``：在满足能力下限的候选模型里，结合实时延迟、GPU 队列、
     前缀缓存与成本，挑选**实际调用**的模型。
3. **Cache-aware / Failure-aware**：优先复用已有前缀缓存的模型以降低 TTFT；
   当小模型反复失败或长时间无进展时自动升级能力档。

对外入口见 :mod:`steproute.cli`。
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__"]