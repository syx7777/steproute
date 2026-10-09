"""安全的密钥管理。

设计原则
--------
1. **绝不明文落盘**：`.env` 只承载非敏感配置；密钥解析优先级为
   ``环境变量 → 操作系统凭据管理器(keyring) → 本地加密密钥库(Fernet) → 交互式输入``。
2. **绝不硬编码**：源码中不出现任何真实密钥，测试与文档统一使用占位符。
3. **读取即脱敏**：任何日志 / CLI 输出经过 :func:`redact` 处理。
4. **不读取明文密钥文件**：程序不会去解析任何以明文保存密钥的说明文档。

Windows 上 ``keyring`` 后端即 Windows Credential Manager（DPAPI 加密存储）。
本地加密密钥库使用 PBKDF2-HMAC-SHA256 派生密钥 + AES-128-CBC/HMAC（Fernet）。
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# 常量与异常
# --------------------------------------------------------------------------- #

SERVICE_NAME = "steproute"
SECRET_NAME = "llm_api_key"
ENV_VAR = "STEPROUTE_LLM_API_KEY"
VAULT_PASSPHRASE_ENV = "STEPROUTE_VAULT_PASSPHRASE"

#: 已知密钥名 → 对应环境变量（供 :func:`manager_for` 与 CLI 复用）。
KNOWN_SECRET_ENV: dict[str, str] = {
    SECRET_NAME: ENV_VAR,
    "feishu_app_secret": "STEPROUTE_FEISHU_APP_SECRET",
}

_KDF_ITERATIONS = 480_000
_VAULT_VERSION = 1

# 通用 API Key 形态（sk-xxx / 长十六进制串等），用于日志兜底脱敏
_GENERIC_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-]{12,}"),
    re.compile(r"(?i)\b(api[_-]?key|token|authorization)\b\s*[:=]\s*[\"']?([A-Za-z0-9_\-\.]{16,})"),
)


class SecretError(RuntimeError):
    """密钥相关错误基类。"""


class SecretNotFoundError(SecretError):
    """未能在任何后端找到密钥。"""


class BackendUnavailableError(SecretError):
    """请求的密钥后端当前环境不可用。"""


# --------------------------------------------------------------------------- #
# 脱敏
# --------------------------------------------------------------------------- #

def mask_secret(value: str | None, keep: int = 4) -> str:
    """把密钥变成可用于展示的掩码，如 ``sk-7***c4c6``。"""
    if not value:
        return "<empty>"
    if len(value) <= keep * 2:
        return "*" * len(value)
    return f"{value[:keep]}{'*' * (len(value) - keep * 2)}{value[-keep:]}"


_KNOWN_SECRETS: set[str] = set()


def register_secret(value: str | None) -> None:
    """登记已知密钥，供 :func:`redact` 全量擦除。"""
    if value and len(value) >= 8:
        _KNOWN_SECRETS.add(value)


def redact(text: str) -> str:
    """擦除文本中出现的所有已知密钥以及常见密钥形态。"""
    if not text:
        return text
    scrubbed = text
    for secret in sorted(_KNOWN_SECRETS, key=len, reverse=True):
        if secret and secret in scrubbed:
            scrubbed = scrubbed.replace(secret, mask_secret(secret))
    scrubbed = _GENERIC_SECRET_PATTERNS[0].sub(lambda m: mask_secret(m.group(0)), scrubbed)

    def _sub_kv(match: re.Match[str]) -> str:
        return f"{match.group(1)}={mask_secret(match.group(2))}"

    return _GENERIC_SECRET_PATTERNS[1].sub(_sub_kv, scrubbed)


# --------------------------------------------------------------------------- #
# 后端 1：操作系统凭据管理器
# --------------------------------------------------------------------------- #

def _keyring_module() -> Any | None:
    try:  # pragma: no cover - 取决于运行环境
        import keyring  # type: ignore import-not-found
    except Exception:  # pragma: no cover
        return None
    return keyring


def keyring_available() -> bool:
    """判断当前环境是否有可用的 keyring 后端（Windows Credential Manager 等）。"""
    keyring = _keyring_module()
    if keyring is None:
        return False
    try:  # 某些后端（fail.Keyring）会在调用时抛错
        from keyring.backends.fail import Keyring as FailKeyring  # type: ignore

        if isinstance(keyring.get_keyring(), FailKeyring):
            return False
    except Exception:  # pragma: no cover
        pass
    return True


# --------------------------------------------------------------------------- #
# 后端 2：本地加密密钥库（Fernet）
# --------------------------------------------------------------------------- #

class EncryptedVault:
    """基于口令的本地加密密钥库。

    文件结构（JSON）::

        {"v": 1, "kdf": "pbkdf2-sha256", "iterations": 480000,
         "salt": "<b64>", "token": "<fernet-token>"}
    """

    def __init__(self, path: Path, passphrase: str | None = None) -> None:
        self.path = Path(path)
        self._passphrase = passphrase

    # -- 能力探测 ---------------------------------------------------------- #
    @staticmethod
    def _fernet_module() -> Any | None:
        try:  # pragma: no cover
            from cryptography.fernet import Fernet  # type: ignore import-not-found

            return Fernet
        except Exception:  # pragma: no cover
            return None

    @classmethod
    def available(cls) -> bool:
        return cls._fernet_module() is not None

    @property
    def exists(self) -> bool:
        return self.path.is_file()

    # -- 口令 -------------------------------------------------------------- #
    def _resolve_passphrase(self, *, prompt: bool = False) -> str:
        if self._passphrase:
            return self._passphrase
        env_value = os.environ.get(VAULT_PASSPHRASE_ENV)
        if env_value:
            self._passphrase = env_value
            return env_value
        if prompt:  # pragma: no cover - 交互路径
            import getpass

            self._passphrase = getpass.getpass("请输入密钥库主口令: ")
            return self._passphrase
        raise BackendUnavailableError(
            f"需要密钥库主口令：请设置环境变量 {VAULT_PASSPHRASE_ENV} 或使用 --passphrase 传入。"
        )

    # -- 加解密 ------------------------------------------------------------ #
    @staticmethod
    def _derive_key(passphrase: str, salt: bytes, iterations: int) -> bytes:
        raw = hashlib.pbkdf2_hmac("sha256", passphrase.encode("utf-8"), salt, iterations)
        return base64.urlsafe_b64encode(raw)

    def load(self) -> dict[str, str]:
        Fernet = self._fernet_module()
        if Fernet is None:  # pragma: no cover
            raise BackendUnavailableError("缺少 cryptography 依赖，无法使用本地加密密钥库。")
        if not self.exists:
            return {}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("v") != _VAULT_VERSION:
            raise SecretError(f"不支持的密钥库版本: {payload.get('v')}")
        salt = base64.b64decode(payload["salt"])
        iterations = int(payload.get("iterations", _KDF_ITERATIONS))
        key = self._derive_key(self._resolve_passphrase(), salt, iterations)
        try:
            data = Fernet(key).decrypt(payload["token"].encode("utf-8"))
        except Exception as exc:  # 口令错误 / 文件损坏
            raise SecretError("密钥库解密失败（主口令错误或文件损坏）。") from exc
        return json.loads(data.decode("utf-8"))

    def save(self, data: Mapping[str, str]) -> None:
        Fernet = self._fernet_module()
        if Fernet is None:  # pragma: no cover
            raise BackendUnavailableError("缺少 cryptography 依赖，无法使用本地加密密钥库。")
        salt = os.urandom(16)
        iterations = _KDF_ITERATIONS
        key = self._derive_key(self._resolve_passphrase(prompt=True), salt, iterations)
        token = Fernet(key).encrypt(json.dumps(dict(data)).encode("utf-8"))
        payload = {
            "v": _VAULT_VERSION,
            "kdf": "pbkdf2-sha256",
            "iterations": iterations,
            "salt": base64.b64encode(salt).decode("ascii"),
            "token": token.decode("ascii"),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        try:  # 尽力收紧权限（Windows 上是 no-op）
            os.chmod(self.path, 0o600)
        except OSError:  # pragma: no cover
            pass


# --------------------------------------------------------------------------- #
# 统一入口
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ResolvedSecret:
    """解析结果：密钥值 + 来源后端。"""

    value: str
    source: str

    def __repr__(self) -> str:  # 避免密钥进入 repr / traceback
        return f"ResolvedSecret(value={mask_secret(self.value)}, source={self.source!r})"


class SecretsManager:
    """密钥解析与写入的统一入口。"""

    def __init__(
        self,
        *,
        name: str = SECRET_NAME,
        service: str = SERVICE_NAME,
        env_var: str = ENV_VAR,
        vault: EncryptedVault | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self.name = name
        self.service = service
        self.env_var = env_var
        self.environ = environ if environ is not None else os.environ
        self.vault = vault if vault is not None else EncryptedVault(_default_vault_path())

    # -- 读 ---------------------------------------------------------------- #
    def resolve(self) -> ResolvedSecret | None:
        """按优先级解析密钥：环境变量 → keyring → 加密密钥库。"""
        env_value = self.environ.get(self.env_var)
        if env_value and env_value.strip():
            register_secret(env_value.strip())
            return ResolvedSecret(env_value.strip(), "env")

        if keyring_available():
            try:
                stored = _keyring_module().get_password(self.service, self.name)
            except Exception:  # pragma: no cover - 后端异常时降级
                stored = None
            if stored and stored.strip():
                register_secret(stored.strip())
                return ResolvedSecret(stored.strip(), "keyring")

        if self.vault.exists and EncryptedVault.available():
            try:
                data = self.vault.load()
            except SecretError:
                data = {}
            stored = (data or {}).get(self.name)
            if stored and stored.strip():
                register_secret(stored.strip())
                return ResolvedSecret(stored.strip(), "vault")

        return None

    def get_llm_api_key(self) -> str:
        """获取 LLM API Key，找不到时抛出带指引的异常。"""
        resolved = self.resolve()
        if resolved is None:
            raise SecretNotFoundError(
                "未找到 LLM API Key。请任选一种方式配置：\n"
                f"  1) 推荐：uv run steproute secrets set        # 存入系统凭据管理器\n"
                f"  2) 写进 .env：STEPROUTE_LLM_API_KEY=<your-key>（明文落盘，.env 已 gitignore）\n"
                f"  3) 临时：$env:{self.env_var} = '<your-key>'      # 仅当前会话有效\n"
                f"  4) 加密库：uv run steproute secrets set --backend vault"
            )
        return resolved.value

    def status(self) -> dict[str, Any]:
        """返回各后端可用性与当前解析来源（不含密钥明文）。"""
        resolved = self.resolve()
        return {
            "env_var": self.env_var,
            "env_present": bool(self.environ.get(self.env_var)),
            "keyring_available": keyring_available(),
            "vault_path": str(self.vault.path),
            "vault_exists": self.vault.exists,
            "vault_available": EncryptedVault.available(),
            "active_source": resolved.source if resolved else None,
            "active_masked": mask_secret(resolved.value) if resolved else None,
        }

    # -- 写 ---------------------------------------------------------------- #
    @staticmethod
    def validate(value: str) -> str:
        value = (value or "").strip()
        if len(value) < 16:
            raise SecretError("密钥长度过短（至少 16 个字符），请确认是否复制完整。")
        if any(ch.isspace() for ch in value):
            raise SecretError("密钥中不应包含空白字符。")
        return value

    def set_secret(self, value: str, *, backend: str = "keyring") -> str:
        """写入密钥（键名即 ``self.name``），返回实际使用的后端名。"""
        value = self.validate(value)
        if backend == "keyring":
            if not keyring_available():
                raise BackendUnavailableError("当前环境不可用 keyring，请改用 --backend vault。")
            _keyring_module().set_password(self.service, self.name, value)
        elif backend == "vault":
            if not EncryptedVault.available():  # pragma: no cover
                raise BackendUnavailableError("缺少 cryptography 依赖，无法使用加密密钥库。")
            data = self.vault.load() if self.vault.exists else {}
            data[self.name] = value
            self.vault.save(data)
        else:
            raise SecretError(f"不支持的后端: {backend}（可选 keyring / vault）")
        register_secret(value)
        return backend

    def delete_secret(self) -> list[str]:
        """从所有可用后端删除密钥（``self.name``），返回被清理的后端列表。"""
        removed: list[str] = []
        if keyring_available():
            try:
                _keyring_module().delete_password(self.service, self.name)
                removed.append("keyring")
            except Exception:  # pragma: no cover - 本来就不存在
                pass
        if self.vault.exists and EncryptedVault.available():
            try:
                data = self.vault.load()
            except SecretError:
                data = {}
            if self.name in data:
                data.pop(self.name)
                self.vault.save(data)
                removed.append("vault")
        return removed


def _default_vault_path() -> Path:
    override = os.environ.get("STEPROUTE_VAULT_PATH")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".steproute" / "secrets.enc"


def default_manager(**kwargs: Any) -> SecretsManager:
    """构造默认 :class:`SecretsManager`（键名为 ``llm_api_key``）。"""
    return SecretsManager(**kwargs)


def manager_for(name: str = SECRET_NAME, **kwargs: Any) -> SecretsManager:
    """按密钥名构造管理器，自动挑选对应的环境变量。

    ``llm_api_key`` → ``STEPROUTE_LLM_API_KEY``；``feishu_app_secret`` →
    ``STEPROUTE_FEISHU_APP_SECRET``；其它名称回退为 ``STEPROUTE_<NAME>``。
    """
    env_var = KNOWN_SECRET_ENV.get(name, f"STEPROUTE_{name.upper()}")
    return SecretsManager(name=name, env_var=env_var, **kwargs)