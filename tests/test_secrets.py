"""密钥管理测试：不落明文、读取即脱敏、多后端优先级。"""

from __future__ import annotations

from pathlib import Path

import pytest

from steproute.secrets import (
    EncryptedVault,
    SecretError,
    SecretNotFoundError,
    SecretsManager,
    mask_secret,
    redact,
    register_secret,
)

FAKE_KEY = "sk-test-1234567890abcdefghij"


def test_mask_secret_hides_middle() -> None:
    masked = mask_secret(FAKE_KEY)
    assert masked.startswith("sk-t")
    assert masked.endswith("ghij")
    assert FAKE_KEY not in masked


def test_redact_removes_registered_secret() -> None:
    register_secret(FAKE_KEY)
    text = f"Authorization: Bearer {FAKE_KEY}"
    scrubbed = redact(text)
    assert FAKE_KEY not in scrubbed
    assert "Bearer" in scrubbed


def test_redact_matches_generic_pattern_without_registration() -> None:
    scrubbed = redact("key=sk-abcdefghijklmnopqrstuvwx")
    assert "sk-abcdefghijklmnopqrstuvwx" not in scrubbed


def test_manager_prefers_env_over_vault(tmp_path: Path) -> None:
    vault = EncryptedVault(tmp_path / "secrets.enc", passphrase="pw")
    vault.save({"llm_api_key": "sk-from-vault-0000000000"})
    manager = SecretsManager(vault=vault, environ={"STEPROUTE_LLM_API_KEY": FAKE_KEY})
    resolved = manager.resolve()
    assert resolved is not None
    assert resolved.source == "env"
    assert resolved.value == FAKE_KEY


def test_manager_reads_from_encrypted_vault(tmp_path: Path) -> None:
    vault_path = tmp_path / "secrets.enc"
    vault = EncryptedVault(vault_path, passphrase="pw")
    vault.save({"llm_api_key": FAKE_KEY})
    # 落盘内容必须是密文，不能包含明文
    assert FAKE_KEY not in vault_path.read_text(encoding="utf-8")

    manager = SecretsManager(vault=EncryptedVault(vault_path, passphrase="pw"), environ={})
    resolved = manager.resolve()
    assert resolved is not None
    assert resolved.source == "vault"
    assert resolved.value == FAKE_KEY


def test_resolve_missing_raises_with_guidance(tmp_path: Path) -> None:
    manager = SecretsManager(vault=EncryptedVault(tmp_path / "none.enc", passphrase="pw"), environ={})
    with pytest.raises(SecretNotFoundError) as excinfo:
        manager.get_llm_api_key()
    assert "secrets set" in str(excinfo.value)


def test_validate_rejects_short_and_whitespace() -> None:
    manager = SecretsManager(environ={})
    with pytest.raises(SecretError):
        manager.validate("short")
    with pytest.raises(SecretError):
        manager.validate("sk-abcdefghij klmnop")


def test_resolved_secret_repr_is_masked(tmp_path: Path) -> None:
    manager = SecretsManager(environ={"STEPROUTE_LLM_API_KEY": FAKE_KEY})
    resolved = manager.resolve()
    assert resolved is not None
    assert FAKE_KEY not in repr(resolved)


def test_status_never_exposes_plaintext() -> None:
    manager = SecretsManager(environ={"STEPROUTE_LLM_API_KEY": FAKE_KEY})
    status = manager.status()
    dumped = str(status)
    assert FAKE_KEY not in dumped
    assert status["active_source"] == "env"