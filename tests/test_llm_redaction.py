import pytest

from pexams.llm import LLMClient, redact_secrets

FAKE_KEY = "sk-ant-api03-" + "x" * 40


def test_redact_secrets_hides_api_keys_but_keeps_the_text():
    assert redact_secrets(f"Illegal header value b'{FAKE_KEY}\\r'") == "Illegal header value b'sk-ant-[redacted]\\r'"
    assert redact_secrets("no key here, sk- alone") == "no key here, sk- alone"


def test_sdk_errors_never_carry_the_key(monkeypatch):
    monkeypatch.setenv("TEST_REDACTION_KEY", FAKE_KEY + "\r")  # a .env with Windows line endings
    client = LLMClient("claude-opus-5-5", api_key_env="TEST_REDACTION_KEY")

    def failing_call(*args, **kwargs):
        try:
            raise ValueError(f"Illegal header value b'{FAKE_KEY}'")
        except ValueError as inner:
            raise ConnectionError("Connection error.") from inner

    monkeypatch.setattr(client, "_anthropic", failing_call)
    with pytest.raises(RuntimeError) as excinfo:
        client.complete("hi")
    assert FAKE_KEY not in str(excinfo.value)
    assert excinfo.value.__cause__ is None and excinfo.value.__suppress_context__
