import json

import inference


def test_resolve_config_supports_api_key_alias(monkeypatch):
    monkeypatch.setenv("API_BASE_URL", "https://router.huggingface.co/v1")
    monkeypatch.setenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "alias-key")

    cfg = inference.resolve_config()
    assert cfg["api_key"] == "alias-key"


def test_resolve_config_uses_provider_default_base(monkeypatch):
    monkeypatch.delenv("API_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_BASE_URL", raising=False)
    monkeypatch.setenv("MODEL_NAME", "gemini-2.5-flash-lite")
    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")

    cfg = inference.resolve_config()
    assert cfg["api_base"] == "https://generativelanguage.googleapis.com/v1beta/openai"


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeCompletion:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return _FakeCompletion("{not-json")
        payload = {
            "category": "billing_dispute",
            "priority": "high",
            "queue": "billing_ops",
            "next_action": "refund_review",
            "response_text": "Thanks, we will review your refund request next.",
        }
        return _FakeCompletion(json.dumps(payload))


class _FakeChat:
    def __init__(self):
        self.completions = _FakeCompletions()


class _FakeClient:
    def __init__(self):
        self.chat = _FakeChat()


def test_call_model_retries_on_malformed_json(monkeypatch):
    monkeypatch.setattr(inference.time, "sleep", lambda _: None)
    client = _FakeClient()
    ticket = {
        "ticket_id": "E-001",
        "subject": "Refund issue",
        "body": "I was charged twice",
        "sla_hours_remaining": 8,
    }
    config = {
        "retry_max": 1,
        "retry_seconds": 0.0,
        "temperature": 0.0,
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "verbose": False,
    }

    plan = inference.call_model_for_ticket(client, ticket, config)

    assert client.chat.completions.calls == 2
    assert plan["category"] == "billing_dispute"
    assert plan["priority"] == "high"


def test_log_baseline_does_not_round_to_boundaries(capsys):
    inference.log_baseline(
        {
            "easy": inference.STRICT_SCORE_EPSILON,
            "medium": 1.0 - inference.STRICT_SCORE_EPSILON,
        },
        seed=42,
        temperature=0.0,
    )
    out = capsys.readouterr().out.strip()
    assert "easy=0.000001" in out
    assert "medium=0.999999" in out
    assert "overall=0.500000" in out
