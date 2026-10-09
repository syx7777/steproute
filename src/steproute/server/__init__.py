"""可视化网站子包（FastAPI 后端 + 静态前端）。

需要可选依赖：``uv sync --extra server``。
"""

from .app import create_app

__all__ = ["create_app"]