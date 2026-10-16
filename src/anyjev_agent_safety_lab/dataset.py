"""Dataset loading, validation, and stable hashing."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .models import SafetyCase

EXPECTED_DOMAINS = {"filesystem", "database", "communications", "deployment"}


class DatasetError(ValueError):
    """Raised when the public synthetic dataset violates its contract."""


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_dataset(
    path: Path, *, canonical: bool | None = None
) -> tuple[list[SafetyCase], dict[str, Any]]:
    """Load and validate a dataset file."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise DatasetError("dataset must be an object with schema_version=1")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list):
        raise DatasetError("dataset cases must be a list")
    cases = [SafetyCase.from_dict(item) for item in raw_cases]
    canonical_profile = (
        payload.get("name") == "agent-action-risk-matched-pairs" if canonical is None else canonical
    )
    validate_dataset(cases, canonical=canonical_profile)
    return cases, payload


def validate_dataset(cases: list[SafetyCase], *, canonical: bool = False) -> None:
    """Enforce the reusable schema and optional canonical benchmark profile."""
    errors: list[str] = []
    if len(cases) < 2:
        errors.append("a dataset must contain at least two cases")
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        errors.append("case IDs must be unique")
    domain_counts = Counter(case.domain for case in cases)
    unknown_domains = set(domain_counts) - EXPECTED_DOMAINS
    if unknown_domains:
        errors.append(f"unsupported domains: {sorted(unknown_domains)}")
    label_counts = Counter(case.risky for case in cases)
    pairs: dict[str, list[SafetyCase]] = defaultdict(list)
    for case in cases:
        pairs[case.pair_id].append(case)
        if not case.id.strip() or not case.pair_id.strip():
            errors.append("case IDs and pair IDs must not be empty")
        if not case.rationale.strip() or not case.risk_factors:
            errors.append(f"{case.id}: rationale and risk_factors are required")
    for pair_id, members in pairs.items():
        if len(members) != 2 or {member.risky for member in members} != {False, True}:
            errors.append(f"{pair_id}: each pair must contain one safe and one risky case")
        if len({member.domain for member in members}) != 1:
            errors.append(f"{pair_id}: paired cases must share a domain")
    if canonical:
        if len(cases) != 80:
            errors.append(f"canonical dataset requires 80 cases, found {len(cases)}")
        if set(domain_counts) != EXPECTED_DOMAINS or any(
            count != 20 for count in domain_counts.values()
        ):
            errors.append(f"each canonical domain must contain 20 cases; got {dict(domain_counts)}")
        if label_counts != {False: 40, True: 40}:
            errors.append(f"canonical dataset must be balanced 40/40; got {dict(label_counts)}")
        if len(pairs) != 40:
            errors.append(f"canonical dataset requires 40 pairs, found {len(pairs)}")
        for case in cases:
            if not case.id.startswith(f"{case.domain[:3]}-"):
                errors.append(f"{case.id}: canonical ID must begin with the domain prefix")
    if errors:
        raise DatasetError("; ".join(errors))


def dataset_hash(payload: dict[str, Any]) -> str:
    """Hash canonical dataset content."""
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()
