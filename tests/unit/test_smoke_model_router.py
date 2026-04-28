from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path

from ticket_agent.domain.model import ModelResponse

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "smoke_model_router.py"
SPEC = importlib.util.spec_from_file_location("smoke_model_router", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
smoke_model_router = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke_model_router)


def test_smoke_model_router_skips_real_call_without_remote_keys(
    monkeypatch,
    capsys,
):
    router = FakeRouter()
    monkeypatch.setattr(
        smoke_model_router,
        "create_model_router",
        lambda timeout_s: router,
    )

    exit_code = smoke_model_router.main(env={})

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "router created: yes" in output
    assert "DEEPSEEK_API_KEY present: no" in output
    assert "GEMINI_API_KEY present: no" in output
    assert "Router construction smoke test passed; skipping real LLM call." in output
    assert router.calls == []


def test_smoke_model_router_uses_gemini_when_key_is_present(monkeypatch, capsys):
    router = FakeRouter(
        response=ModelResponse(
            content="router smoke ok",
            model="gemini-2.5-flash",
            provider="gemini",
            capability="trivial.respond",
            input_tokens=8,
            output_tokens=4,
        )
    )
    monkeypatch.setattr(
        smoke_model_router,
        "create_model_router",
        lambda timeout_s: router,
    )

    exit_code = smoke_model_router.main(env={"GEMINI_API_KEY": "secret-value"})

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "selected remote provider for smoke call: gemini" in output
    assert "selected model: gemini-2.5-flash" in output
    assert "provider: gemini" in output
    assert "response text: router smoke ok" in output
    assert "token usage: input=8 output=4" in output
    assert router.calls == [
        (
            "trivial.respond",
            [
                {"role": "system", "content": "You are a concise test responder."},
                {"role": "user", "content": "Reply with exactly: router smoke ok"},
            ],
        )
    ]


class FakeRouter:
    def __init__(self, response: ModelResponse | None = None) -> None:
        self._providers = {
            "deepseek": object(),
            "gemini": object(),
            "ollama": FakeOllamaProvider(),
        }
        self._selector = object()
        self._response = response or ModelResponse(
            content="unused",
            model="unused",
            provider="unused",
            capability="unused",
        )
        self.calls: list[tuple[str, list[dict]]] = []

    async def invoke(self, capability: str, messages: list[dict]) -> ModelResponse:
        await asyncio.sleep(0)
        self.calls.append((capability, messages))
        return self._response


class FakeOllamaProvider:
    base_url = "http://localhost:11434"
