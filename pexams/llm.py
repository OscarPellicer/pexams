"""Provider-neutral LLM calls (text + images) with provenance logging.

This module is duplicated, byte for byte, in pexams (pexams/llm.py) and pevaluate
(autocorrect/llm.py) so that neither package depends on the other; a test in each repository
fails if the two copies diverge.

Models named with a bare Anthropic id (``claude-…``) are called through the official
``anthropic`` SDK; any other name (``google/gemini-…``, ``anthropic/claude-…``) goes to
OpenRouter through the ``openai`` SDK. Every call can be appended to a JSONL provenance log
(model actually served, effort, request id, token usage, hashes of the prompt and images), so
a grading run can be traced and reported later.
"""
import base64
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
        api_key = os.getenv(self.api_key_env)
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
    ) -> Tuple[str, dict]:
        start = time.time()
        if self.provider == "anthropic":
            text, served, request_id, stop_reason, usage, fallback_used = self._anthropic(prompt, images, system, max_tokens)
        else:
            text, served, request_id, stop_reason, usage, fallback_used = self._openrouter(prompt, images, system, max_tokens, json_output)
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
        return text, meta

    def _anthropic(self, prompt, images, system, max_tokens):
        content = [
            {"type": "image", "source": {"type": "base64", "media_type": mime, "data": base64.standard_b64encode(data).decode("ascii")}}
            for data, mime in images
        ]
        content.append({"type": "text", "text": prompt})
        params = {"model": self.model, "max_tokens": max_tokens, "messages": [{"role": "user", "content": content}]}
        if system:
            params["system"] = system
        if self.effort:
            params["output_config"] = {"effort": self.effort}
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

    def _openrouter(self, prompt, images, system, max_tokens, json_output):
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
        if json_output:
            params["response_format"] = {"type": "json_object"}
        response = self._client.chat.completions.create(**params)
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
