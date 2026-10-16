"""Use AnyJev to obtain typed probability decisions from a local LLM."""

from __future__ import annotations

import hashlib
import importlib
import os
from typing import Any, Literal

from .models import json_compatible
from .strategies import _resolve_device

QuestionKind = Literal["choice", "noul", "score"]
ShowcaseBackend = Literal["hf", "vllm", "fixture"]


def _fixture_content(state: str, option: str) -> float:
    digest = hashlib.sha256(f"{state}|{option}".encode()).digest()
    return int.from_bytes(digest[:4], "big") / 2**32


def _build_question(
    question_type: Any,
    kind: QuestionKind,
    text: str,
    options: list[str],
    bins: int,
) -> Any:
    if kind == "choice":
        if len(options) < 2:
            raise ValueError("choice questions require at least two --options")
        return question_type.choice(text, options, name="decision")
    if kind == "noul":
        return question_type.noul(text, name="decision")
    return question_type.score(text, bins=bins, name="decision")


def run_showcase(
    *,
    state: str,
    question: str,
    kind: QuestionKind,
    options: list[str],
    bins: int,
    levels: list[str],
    backend: ShowcaseBackend,
    model: str,
    revision: str | None,
    device: str,
    base_url: str | None = None,
    tokenizer: str | None = None,
    api_key_env: str = "VLLM_API_KEY",
) -> dict[str, Any]:
    """Run one typed question through Raw and/or L0 on a selected backend."""
    anyjev = importlib.import_module("anyjev")
    typed_question = _build_question(anyjev.Question, kind, question, options, bins)
    if any(level not in {"raw", "L0"} for level in levels):
        raise ValueError("showcase levels must be raw or L0")

    if backend == "fixture":
        fake_module = importlib.import_module("anyjev.backends.fake")
        model_backend = fake_module.FakeBackend(
            _fixture_content,
            position_bias=[0.8 - index * 0.2 for index in range(len(typed_question.options))],
        )
        resolved_device = "fixture"
    elif backend == "hf":
        torch = importlib.import_module("torch")
        resolved_device = _resolve_device(device, torch)
        hf_module = importlib.import_module("anyjev.backends.hf")
        model_backend = hf_module.HFBackend(
            model,
            device=resolved_device,
            revision=revision,
        )
    else:
        if not base_url:
            raise ValueError("vllm showcase requires --base-url")
        vllm_module = importlib.import_module("anyjev.backends.vllm")
        model_backend = vllm_module.VLLMBackend(
            base_url=base_url,
            model=model,
            tokenizer_name=tokenizer,
            api_key=os.environ.get(api_key_env, "EMPTY"),
        )
        resolved_device = "served"

    decisions: dict[str, Any] = {}
    for level in levels:
        decider = anyjev.Decider(
            model_backend,
            level=level,
            prior="content_free",
            adaptive_shifts=False,
        )
        answer = decider.decide(state, [typed_question], require=level)["decision"]
        decisions[level] = json_compatible(answer.to_dict())

    return {
        "schema_version": 1,
        "backend": backend,
        "model": model,
        "model_revision": revision,
        "device": resolved_device,
        "state": state,
        "question": {
            "kind": kind,
            "text": question,
            "options": list(typed_question.options),
        },
        "decisions": decisions,
        "notice": (
            "Deterministic fixture output demonstrates the API only; it is not a model result."
            if backend == "fixture"
            else "Real typed decisions from the selected compatible model backend."
        ),
    }
