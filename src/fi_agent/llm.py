"""Ollama client wrapper enforcing schema-valid output.

Everything the agents send goes through `structured()`, which:

* pins `num_ctx` explicitly - Ollama defaults to 4096, which would silently truncate
  article text and yield confident nonsense with no error anywhere,
* constrains generation with the Pydantic model's JSON schema via Ollama's `format`,
* validates the result and, on failure, retries with the validation error fed back,
* gives up gracefully rather than raising, so one malformed response degrades a single
  ticker instead of aborting the report,
* records every call to a JSONL trace for latency and token accounting.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, TypeVar

import ollama
from pydantic import BaseModel, ValidationError

from fi_agent.config import LLMSettings

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMUnavailable(RuntimeError):
    """The Ollama server could not be reached or the model is missing."""


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve $ref/$defs into an inlined schema.

    Pydantic emits nested models as `$defs` plus `$ref` pointers. Ollama compiles the
    schema into a grammar and handles references inconsistently across versions, so
    inlining removes the ambiguity entirely.
    """
    defs = schema.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                target = defs.get(ref.split("/")[-1], {})
                merged = {**resolve(target), **{k: v for k, v in node.items() if k != "$ref"}}
                return merged
            return {key: resolve(value) for key, value in node.items()}
        if isinstance(node, list):
            return [resolve(item) for item in node]
        return node

    return resolve(schema)


class LLMClient:
    def __init__(self, settings: LLMSettings, trace_path: Path | None = None) -> None:
        self.settings = settings
        self.trace_path = trace_path
        self.calls = 0
        self.total_seconds = 0.0
        self._client = ollama.Client(host=settings.base_url, timeout=settings.request_timeout_s)

    # -- health ------------------------------------------------------------------------

    def check(self) -> tuple[bool, str]:
        """Is the server up and does it have the configured model?"""
        try:
            listing = self._client.list()
        except Exception as exc:
            return False, f"cannot reach {self.settings.base_url}: {exc}"

        names = {m.model for m in listing.models if m.model}
        if self.settings.model not in names:
            return False, (
                f"model {self.settings.model!r} not on server; available: {sorted(names)}"
            )
        return True, f"{self.settings.model} available at {self.settings.base_url}"

    # -- generation --------------------------------------------------------------------

    def _trace(self, record: dict[str, Any]) -> None:
        if self.trace_path is None:
            return
        try:
            with self.trace_path.open("a") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
        except OSError as exc:
            log.debug("trace write failed: %s", exc)

    def structured(
        self,
        schema: type[T],
        system: str,
        user: str,
        label: str = "",
    ) -> T | None:
        """Return a schema-valid instance, or None if the model could not produce one."""
        json_schema = _inline_refs(schema.model_json_schema())
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        last_error = ""
        for attempt in range(self.settings.max_retries + 1):
            started = time.monotonic()
            try:
                response = self._client.chat(
                    model=self.settings.model,
                    messages=messages,
                    format=json_schema,
                    options={
                        "num_ctx": self.settings.num_ctx,
                        "temperature": self.settings.temperature,
                    },
                )
            except Exception as exc:
                elapsed = time.monotonic() - started
                self.calls += 1
                self.total_seconds += elapsed
                self._trace(
                    {"label": label, "attempt": attempt, "error": str(exc), "seconds": elapsed}
                )
                log.warning("%s: ollama call failed (%s)", label or schema.__name__, exc)
                last_error = str(exc)
                continue

            elapsed = time.monotonic() - started
            self.calls += 1
            self.total_seconds += elapsed
            content = response.message.content or ""

            self._trace(
                {
                    "label": label,
                    "attempt": attempt,
                    "seconds": round(elapsed, 2),
                    "prompt_tokens": response.prompt_eval_count,
                    "output_tokens": response.eval_count,
                    "chars": len(content),
                }
            )

            try:
                return schema.model_validate_json(content)
            except ValidationError as exc:
                last_error = str(exc)
                log.warning(
                    "%s: schema validation failed on attempt %d", label or schema.__name__,
                    attempt + 1,
                )
                # Feed the failure back so the retry has something to correct against.
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": (
                            "That response did not match the required schema:\n"
                            f"{last_error}\n\nReturn corrected JSON only."
                        ),
                    },
                ]

        log.error("%s: giving up after %d attempts", label or schema.__name__,
                  self.settings.max_retries + 1)
        return None
