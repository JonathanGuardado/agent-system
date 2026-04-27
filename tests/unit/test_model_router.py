from __future__ import annotations

import json
from urllib.error import URLError

import pytest

from ticket_agent.domain.errors import AllModelsFailedError
from ticket_agent.domain.model_selection import ModelEndpoint, ModelSelection
from ticket_agent.router import model_router
from ticket_agent.router.model_router import ModelRouter


def test_invoke_calls_the_selected_primary_model(monkeypatch):
    requests = _patch_selection_and_urlopen(monkeypatch)

    ModelRouter().invoke(
        "implement OAuth login",
        [{"role": "user", "content": "implement OAuth login"}],
    )

    assert requests[0].payload["model"] == "primary-deployment"


def test_invoke_returns_content_from_openai_compatible_response(monkeypatch):
    _patch_selection_and_urlopen(monkeypatch, content="done")

    response = ModelRouter().invoke(
        "implement OAuth login",
        [{"role": "user", "content": "implement OAuth login"}],
    )

    assert response.content == "done"
    assert response.model_used == "primary-tier"
    assert response.provider_used == "primary-provider"
    assert response.deployment_used == "primary-deployment"
    assert response.fallback_used is False


def test_invoke_retries_first_fallback_when_primary_fails(monkeypatch):
    requests = _patch_selection_and_urlopen(
        monkeypatch,
        failures={"primary-deployment"},
        content="fallback answer",
    )

    response = ModelRouter().invoke(
        "implement OAuth login",
        [{"role": "user", "content": "implement OAuth login"}],
    )

    assert [request.payload["model"] for request in requests] == [
        "primary-deployment",
        "fallback-one-deployment",
    ]
    assert response.content == "fallback answer"
    assert response.model_used == "fallback-one-tier"
    assert response.deployment_used == "fallback-one-deployment"


def test_fallback_used_is_true_when_fallback_succeeds(monkeypatch):
    _patch_selection_and_urlopen(monkeypatch, failures={"primary-deployment"})

    response = ModelRouter().invoke(
        "implement OAuth login",
        [{"role": "user", "content": "implement OAuth login"}],
    )

    assert response.fallback_used is True


def test_if_all_models_fail_raises_all_models_failed(monkeypatch):
    _patch_selection_and_urlopen(
        monkeypatch,
        failures={
            "primary-deployment",
            "fallback-one-deployment",
            "fallback-two-deployment",
        },
    )

    with pytest.raises(AllModelsFailedError) as exc_info:
        ModelRouter().invoke(
            "implement OAuth login",
            [{"role": "user", "content": "implement OAuth login"}],
        )

    assert [failure.model for failure in exc_info.value.failures] == [
        "primary-tier",
        "fallback-one-tier",
        "fallback-two-tier",
    ]
    assert [failure.deployment for failure in exc_info.value.failures] == [
        "primary-deployment",
        "fallback-one-deployment",
        "fallback-two-deployment",
    ]


def test_authorization_header_is_only_sent_when_api_key_is_provided(monkeypatch):
    requests = _patch_selection_and_urlopen(monkeypatch)

    ModelRouter(api_key=None).invoke("hello", [{"role": "user", "content": "hello"}])
    ModelRouter(api_key="secret").invoke("hello", [{"role": "user", "content": "hello"}])

    assert requests[0].authorization is None
    assert requests[1].authorization == "Bearer secret"


def test_selection_metadata_headers_are_sent(monkeypatch):
    requests = _patch_selection_and_urlopen(monkeypatch)

    ModelRouter().invoke("hello", [{"role": "user", "content": "hello"}])

    assert requests[0].selection_tier == "primary-tier"
    assert requests[0].provider == "primary-provider"
    assert requests[0].invocation == "openai_chat"


class CapturedRequest:
    def __init__(self, request):
        self.payload = json.loads(request.data.decode("utf-8"))
        self.authorization = request.get_header("Authorization")
        self.selection_tier = request.get_header("X-model-selection-tier")
        self.provider = request.get_header("X-model-provider")
        self.invocation = request.get_header("X-model-invocation")


class FakeResponse:
    status = 200

    def __init__(self, content: str) -> None:
        self._content = content

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self) -> bytes:
        return json.dumps(
            {"choices": [{"message": {"content": self._content}}]}
        ).encode("utf-8")


def _patch_selection_and_urlopen(
    monkeypatch,
    *,
    failures: set[str] | None = None,
    content: str = "ok",
) -> list[CapturedRequest]:
    failures = failures or set()
    requests: list[CapturedRequest] = []

    def fake_select(_capability_or_text: str) -> ModelSelection:
        return ModelSelection(
            capability="code.implement",
            primary=ModelEndpoint(
                selection_tier="primary-tier",
                provider="primary-provider",
                model_name="primary-model",
                deployment_name="primary-deployment",
            ),
            fallbacks=(
                ModelEndpoint(
                    selection_tier="fallback-one-tier",
                    provider="fallback-one-provider",
                    model_name="fallback-one-model",
                    deployment_name="fallback-one-deployment",
                ),
                ModelEndpoint(
                    selection_tier="fallback-two-tier",
                    provider="fallback-two-provider",
                    model_name="fallback-two-model",
                    deployment_name="fallback-two-deployment",
                ),
            ),
        )

    def fake_urlopen(request, timeout: int):
        del timeout
        captured = CapturedRequest(request)
        requests.append(captured)
        if captured.payload["model"] in failures:
            raise URLError("boom")
        return FakeResponse(content)

    monkeypatch.setattr(model_router, "select_model_for_capability", fake_select)
    monkeypatch.setattr(model_router, "urlopen", fake_urlopen)
    return requests
