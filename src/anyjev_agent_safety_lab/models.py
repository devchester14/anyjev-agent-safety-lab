"""Typed domain models and JSON conversion helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, cast

Domain = Literal["filesystem", "database", "communications", "deployment"]
StrategyName = Literal["json", "raw", "l0"]
PolicyAction = Literal["allow", "review", "block"]
BackendName = Literal["fake", "hf"]


def json_compatible(value: Any) -> Any:
    """Convert third-party scalar and array diagnostics to JSON-native values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]
    if hasattr(value, "tolist"):
        return json_compatible(value.tolist())
    if hasattr(value, "item"):
        return json_compatible(value.item())
    return str(value)


@dataclass(frozen=True)
class ToolCall:
    """A synthetic tool call proposed by an agent."""

    name: str
    arguments: dict[str, Any]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ToolCall:
        return cls(name=str(value["name"]), arguments=dict(value.get("arguments", {})))


@dataclass(frozen=True)
class SafetyCase:
    """One labelled, synthetic agent-action example."""

    id: str
    pair_id: str
    domain: Domain
    request: str
    proposed_action: ToolCall
    context: str
    risky: bool
    rationale: str
    risk_factors: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SafetyCase:
        return cls(
            id=str(value["id"]),
            pair_id=str(value["pair_id"]),
            domain=cast(Domain, value["domain"]),
            request=str(value["request"]),
            proposed_action=ToolCall.from_dict(value["proposed_action"]),
            context=str(value["context"]),
            risky=bool(value["risky"]),
            rationale=str(value["rationale"]),
            risk_factors=tuple(str(item) for item in value.get("risk_factors", [])),
        )

    def model_state(self) -> dict[str, Any]:
        """Return only fields visible to a decision model."""
        return {
            "domain": self.domain,
            "request": self.request,
            "proposed_action": asdict(self.proposed_action),
            "context": self.context,
        }


@dataclass(frozen=True)
class DecisionResult:
    """Normalized output from every benchmark strategy."""

    case_id: str
    strategy: StrategyName
    backend: BackendName
    model: str
    model_revision: str
    option_order: tuple[str, str]
    decision: Literal["safe", "risky"] | None
    probabilities: dict[str, float]
    risk_probability: float | None
    action: PolicyAction
    parsed: bool
    latency_ms: float
    diagnostics: dict[str, Any] = field(default_factory=dict)
    raw_output: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["option_order"] = list(self.option_order)
        return cast(dict[str, Any], json_compatible(value))


@dataclass(frozen=True)
class SimulationOutcome:
    """Result of applying the gate to an isolated simulated tool."""

    case_id: str
    action: PolicyAction
    executed: bool
    human_reviewed: bool
    harmful_execution: bool
    success_without_review: bool
    success_with_oracle_review: bool
    effect: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
