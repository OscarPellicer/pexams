"""Provider-neutral LLM calls (text + images) with provenance logging.

This module is duplicated, byte for byte, in pexams (pexams/llm.py) and pevaluate
(autocorrect/llm.py) so that neither package depends on the other; a test in each repository
fails if the two copies diverge.

Models named with a bare Anthropic id (``claude-…``) are called through the official
``anthropic`` SDK; any other name (``google/gemini-…``, ``anthropic/claude-…``) goes to
OpenRouter through the ``openai`` SDK. Every call can be appended to a JSONL provenance log
(model actually served, effort, request id, token usage, hashes of the prompt and images), so
a grading run can be traced and reported later.

``complete(..., json_schema=schema)`` asks for structured output: Anthropic's native
``output_config.format`` when the installed SDK supports it (otherwise a strict-JSON
instruction), and a strict ``json_schema`` ``response_format`` on OpenRouter. The answer is
checked to be a JSON object either way.
"""
import base64
import copy
import datetime
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

# Models that accept the server-side refusal fallback (``fallbacks: "default"``).
_FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"}
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


def resolve_provider(model: str, provider: str = "auto") -> str:
    if provider and provider != "auto":
        return provider
    return "anthropic" if str(model).startswith("claude-") else "openrouter"


def _sha256(data) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


_SECRET_PATTERN = re.compile(r"(sk-[A-Za-z0-9]{2,8}-|sk-|AIza|r8_)[A-Za-z0-9_\-]{8,}")


def redact_secrets(text: str) -> str:
    """Replace anything that looks like an API key (Anthropic/OpenAI/OpenRouter ``sk-…``,
    Google ``AIza…``, Replicate ``r8_…``) by its prefix and ``[redacted]``."""
    return _SECRET_PATTERN.sub(lambda m: m.group(1) + "[redacted]", str(text))


def parse_json_response(text: str) -> dict:
    """Extract the first JSON object from a model response (tolerates code fences)."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        text = match.group(0)
    return json.loads(text)


def strict_json_schema(schema: dict) -> dict:
    """Copy of ``schema`` in the strict form OpenAI-style structured outputs require: every
    object forbids extra properties and lists all its properties as required."""
    schema = copy.deepcopy(schema)

    def visit(node):
        if isinstance(node, dict):
            if node.get("type") == "object" or "properties" in node:
                node["additionalProperties"] = False
                if node.get("properties"):
                    node["required"] = list(node["properties"].keys())
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(schema)
    return schema


def _anthropic_structured_output_supported() -> bool:
    """True when the installed anthropic SDK accepts ``output_config={"format": ...}``."""
    try:
        from anthropic.types import OutputConfigParam
    except ImportError:
        return False
    return "format" in getattr(OutputConfigParam, "__annotations__", {})


def _anthropic_schema(schema: dict) -> dict:
    """Schema in the subset Anthropic accepts; the SDK's ``transform_schema`` moves unsupported
    constraints (e.g. ``minItems: 3``) into the descriptions."""
    try:
        from anthropic import transform_schema
    except ImportError:
        transform_schema = None
    if transform_schema is not None:
        try:
            return transform_schema(copy.deepcopy(schema))
        except Exception:
            pass
    return strict_json_schema(schema)


def _schema_instruction(schema: dict) -> str:
    return (
        "\n\nRespond with a single JSON object, and nothing else, that matches this JSON Schema:\n"
        + json.dumps(schema, ensure_ascii=False)
    )


class LLMClient:
    """One configured model. ``complete`` sends a prompt plus optional images and returns
    ``(text, meta)``; ``meta`` is also appended to ``log_path`` when given."""

    def __init__(
        self,
        model: str,
        provider: str = "auto",
        effort: Optional[str] = None,
        api_key_env: Optional[str] = None,
        log_path: Optional[str] = None,
        tool: str = "ai4exams",
        max_retries: int = 3,
        timeout: float = 180.0,
    ):
        self.model = model
        self.provider = resolve_provider(model, provider)
        self.effort = effort
        self.log_path = log_path
        self.tool = tool
        default_env = "ANTHROPIC_API_KEY" if self.provider == "anthropic" else "OPENROUTER_API_KEY"
        self.api_key_env = api_key_env or default_env
        # Strip whitespace and quotes: a .env with Windows line endings leaves a trailing "\r",
        # which httpx rejects as an illegal header value (reported only as "Connection error").
        api_key = (os.getenv(self.api_key_env) or "").strip().strip("\"'")
        if not api_key:
            raise RuntimeError(f"{self.api_key_env} is not set (needed for {self.provider} model {model}).")
        if self.provider == "anthropic":
            import anthropic

            self._client = anthropic.Anthropic(api_key=api_key, max_retries=max_retries, timeout=timeout)
        else:
            import openai

            self._client = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key, max_retries=max_retries, timeout=timeout)

    def describe(self) -> dict:
        return {"provider": self.provider, "model": self.model, "effort": self.effort, "api_key_env": self.api_key_env}

    def complete(
        self,
        prompt: str,
        images: Sequence[Tuple[bytes, str]] = (),
        system: Optional[str] = None,
        max_tokens: int = 16000,
        json_output: bool = False,
        purpose: str = "",
        context: Optional[dict] = None,
        json_schema: Optional[dict] = None,
        schema_name: str = "",
    ) -> Tuple[str, dict]:
        """``json_schema`` (e.g. ``Model.model_json_schema()``) requests structured output and
        implies ``json_output``; the text returned is then a JSON object (``ValueError`` if not)."""
        start = time.time()
        output_format = "json_schema" if json_schema else ("json_object" if json_output else "text")
        try:
            if self.provider == "anthropic":
                if json_schema and not _anthropic_structured_output_supported():
                    output_format = "json_instruction"
                    prompt = prompt + _schema_instruction(json_schema)
                    json_schema = None
                text, served, request_id, stop_reason, usage, fallback_used = self._anthropic(prompt, images, system, max_tokens, json_schema)
            else:
                text, served, request_id, stop_reason, usage, fallback_used = self._openrouter(
                    prompt, images, system, max_tokens, json_output, json_schema, schema_name
                )
        except Exception as e:
            # SDK errors can carry the request headers, API key included, in their message or in
            # the exception chain (e.g. httpx "Illegal header value b'sk-…'"). Re-raise a redacted
            # message and drop the chain, so a key never reaches a log, a report or the console.
            raise RuntimeError(f"{type(e).__name__}: {redact_secrets(str(e))}") from None
        meta = {
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "tool": self.tool,
            "tool_version": package_version(self.tool),
            "purpose": purpose,
            "context": context or {},
            "provider": self.provider,
            "model_requested": self.model,
            "model_served": served,
            "fallback_used": fallback_used,
            "effort": self.effort,
            "max_tokens": max_tokens,
            "output_format": output_format,
            "request_id": request_id,
            "stop_reason": stop_reason,
            "usage": usage,
            "latency_s": round(time.time() - start, 2),
            "system_sha256": _sha256(system) if system else "",
            "prompt_sha256": _sha256(prompt),
            "image_sha256": [_sha256(data) for data, _ in images],
            "response_sha256": _sha256(text or ""),
        }
        if self.log_path:
            Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        if stop_reason == "refusal":
            raise RuntimeError(f"The model declined the request ({purpose}).")
        if output_format in ("json_schema", "json_instruction"):
            try:
                parsed = parse_json_response(text)
            except (json.JSONDecodeError, TypeError):
                parsed = None
            if not isinstance(parsed, dict):
                raise ValueError(f"The model did not return a JSON object ({purpose}, stop reason {stop_reason}).")
            if output_format == "json_instruction":
                text = json.dumps(parsed, ensure_ascii=False)
        return text, meta

    def _anthropic(self, prompt, images, system, max_tokens, json_schema=None):
        content = [
            {"type": "image", "source": {"type": "base64", "media_type": mime, "data": base64.standard_b64encode(data).decode("ascii")}}
            for data, mime in images
        ]
        content.append({"type": "text", "text": prompt})
        params = {"model": self.model, "max_tokens": max_tokens, "messages": [{"role": "user", "content": content}]}
        if system:
            params["system"] = system
        output_config = {}
        if self.effort:
            output_config["effort"] = self.effort
        if json_schema:
            output_config["format"] = {"type": "json_schema", "schema": _anthropic_schema(json_schema)}
        if output_config:
            params["output_config"] = output_config
        # Streaming keeps long requests alive and avoids HTTP timeouts on large outputs.
        if self.model in _FALLBACK_MODELS:
            stream = self._client.beta.messages.stream(betas=[_FALLBACK_BETA], fallbacks="default", **params)
        else:
            stream = self._client.messages.stream(**params)
        with stream as events:
            response = events.get_final_message()
        text = "".join(block.text for block in response.content if block.type == "text")
        iterations = getattr(response.usage, "iterations", None) or []
        fallback_used = any(getattr(entry, "type", "") == "fallback_message" for entry in iterations)
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", None),
        }
        return text, response.model, getattr(response, "_request_id", None), response.stop_reason, usage, fallback_used

    def _openrouter(self, prompt, images, system, max_tokens, json_output, json_schema=None, schema_name=""):
        user_content = [{"type": "text", "text": prompt}] + [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64," + base64.b64encode(data).decode("ascii")}}
            for data, mime in images
        ]
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user_content}]
        extra_body = {}
        if self.effort:
            extra_body["reasoning"] = {"effort": self.effort}
        elif "flash" in self.model.lower():
            # Gemini Flash reasons by default; minimal effort avoids paying for reasoning tokens.
            extra_body["reasoning"] = {"effort": "minimal"}
        params = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "extra_body": extra_body}
        if json_schema:
            name = re.sub(r"[^A-Za-z0-9_-]", "_", schema_name or str(json_schema.get("title") or "structured_output"))[:64]
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": name, "strict": True, "schema": strict_json_schema(json_schema)},
            }
        elif json_output:
            params["response_format"] = {"type": "json_object"}
        response = self._client.chat.completions.create(**params)
        # OpenRouter reports upstream failures (rate limits, provider errors) inside a 200 response.
        error = getattr(response, "error", None)
        if error or not getattr(response, "choices", None):
            message = error.get("message", error) if isinstance(error, dict) else (error or "no choices in the response")
            raise RuntimeError(f"OpenRouter returned an error for {self.model}: {message}")
        choice = response.choices[0]
        usage = {
            "input_tokens": getattr(response.usage, "prompt_tokens", None),
            "output_tokens": getattr(response.usage, "completion_tokens", None),
        }
        return choice.message.content or "", getattr(response, "model", self.model), getattr(response, "id", None), choice.finish_reason, usage, False


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return ""


def _git_commit(module_name: str) -> str:
    """Commit of the package's source checkout (editable installs), without importing it."""
    from importlib.util import find_spec

    try:
        spec = find_spec(module_name)
    except (ImportError, ValueError):
        return ""
    if spec is None or not spec.origin:
        return ""
    folder = Path(spec.origin).resolve().parent
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=folder, capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=folder, capture_output=True, text=True, timeout=5)
        commit = out.stdout.strip()
        return f"{commit}{'-dirty' if dirty.stdout.strip() else ''}" if commit else ""
    except Exception:
        return ""


def write_run_manifest(output_dir: str, tool: str, llm: Optional[LLMClient] = None, inputs: Iterable[str] = (), extra: Optional[dict] = None) -> str:
    """Write ``run_manifest.json`` (command, versions, model, input file hashes) to ``output_dir``."""
    os.makedirs(output_dir, exist_ok=True)
    packages = {}
    for name, module in (("pexams", "pexams"), ("ptestgen", "ptestgen"), ("pevaluate", "autocorrect"), ("anthropic", "anthropic"), ("openai", "openai")):
        version = package_version(name)
        if version:
            packages[name] = {"version": version, "git_commit": _git_commit(module)}
    input_hashes = {}
    for path in inputs:
        if path and os.path.isfile(path):
            input_hashes[os.path.abspath(path)] = _sha256(Path(path).read_bytes())
    manifest = {
        "tool": tool,
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "command": sys.argv,
        "cwd": os.getcwd(),  # relative paths in the command resolve from here
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
        "llm": llm.describe() if llm else None,
        "inputs_sha256": input_hashes,
        **(extra or {}),
    }
    path = os.path.join(output_dir, "run_manifest.json")
    Path(path).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
