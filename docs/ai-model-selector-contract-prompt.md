# Prompt: Implement Selector Execution Contract

You are working in `/home/jguardado/repos/ai-model-selector`.

Goal: make `ai-model-selector` return a complete, stable execution plan that downstream routers can execute without duplicating model-selection policy.

Current desired boundary:

- `ai-model-selector` owns intent resolution, request context construction, model ranking, primary/fallback choice, and selector debug metadata.
- The consuming app/router owns provider credentials, HTTP/client execution, retries, cost logging, and response shaping.
- The consuming app should not maintain a separate `selection_tier -> model` mapping, because that splits model policy across two codebases.

Required contract:

```python
resolution = resolver.resolve(prompt)
context = build_request_context(resolution)
plan = selector.select(context)
```

The returned plan must include full endpoint metadata for the primary and fallbacks:

```python
@dataclass(frozen=True, slots=True)
class ModelSelection:
    selection_tier: str       # stable policy/interface name
    provider: str             # runtime provider key: deepseek, gemini, ollama, etc.
    model_name: str           # human/config model identity
    deployment_name: str      # exact model/deployment string the router sends
    invocation: str = "openai_chat"  # optional if you add this in config


@dataclass(frozen=True, slots=True)
class SelectionDecision:
    capability: str
    profile: str
    primary: ModelSelection
    fallbacks: tuple[ModelSelection, ...]
    ranked_candidates: tuple[RankedCandidate, ...]
    filtered_candidates: tuple[FilteredCandidate, ...]
    debug_reasons: tuple[str, ...]
```

Field semantics:

- `selection_tier` is the stable app-facing interface, such as `dev`, `coding_primary`, `reasoning_primary`, `web_agent`, or `local_fast`.
- `provider` is a lookup key used by the internal router to find credentials/client config.
- `deployment_name` is the exact value the internal router sends to the provider client.
- `model_name` is descriptive model identity and may equal `deployment_name`.
- `invocation` or `api_style`, if added, tells the app/router how to call the endpoint. Start with `openai_chat` as the default. Do not put API keys in selector config.

Implementation tasks:

1. Preserve and document the existing explicit flow in `examples/usage_example.py`:

   ```python
   resolution = resolver.resolve(prompt)
   context = build_request_context(resolution)
   decision = selector.select(context)
   ```

2. Add or strengthen a convenience helper for app integrations:

   ```python
   decision = selector.select_prompt(prompt)
   ```

   This should require `capabilities_path` to be passed to `DeterministicSelector.from_yaml(...)` or otherwise raise a clear error.

3. Ensure `decision.primary` and every item in `decision.fallbacks` are full `ModelSelection` objects with `selection_tier`, `provider`, `model_name`, and `deployment_name`.

4. Keep compatibility properties:

   ```python
   decision.primary_model              # returns primary.model_name
   decision.primary_selection_tier     # returns primary.selection_tier
   decision.fallback_models            # returns fallback.model_name values
   decision.fallback_selection_tiers   # returns fallback.selection_tier values
   ```

5. Consider adding `invocation` to model config and model dataclasses with a default of `openai_chat`, but keep this backwards-compatible with existing YAML.

6. Add tests that prove:

   - Free-form prompt -> capability -> context -> decision works end to end.
   - `decision.primary.deployment_name` is preserved from `models.yaml`.
   - Fallbacks are returned as full endpoint objects, not just tier strings.
   - Compatibility properties still return the expected string tuples.
   - `select_prompt(...)` works when capabilities are configured and fails clearly when not configured.

7. Update README docs with the app/router contract:

   - Selector answers: "which model endpoint should handle this request?"
   - Router answers: "how do I call this provider/deployment?"
   - Selector must not perform network calls or know provider secrets.

Important downstream expectation:

The consumer in `/home/jguardado/repos/agent-system` now expects a selection plan shaped like:

```python
selection.primary.selection_tier
selection.primary.provider
selection.primary.model_name
selection.primary.deployment_name
selection.fallbacks
```

The app sends `endpoint.deployment_name` to the internal provider client while preserving `endpoint.selection_tier` in metadata/logs.
