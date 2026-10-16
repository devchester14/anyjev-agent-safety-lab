import numpy as np
import pytest
from anyjev import Decider, Question  # type: ignore[import-untyped]
from anyjev.backends.fake import FakeBackend  # type: ignore[import-untyped]

from anyjev_agent_safety_lab.models import DecisionResult
from anyjev_agent_safety_lab.strategies import (
    distribution_to_probabilities,
    parse_json_decision,
)


def test_parse_json_decision_with_surrounding_text() -> None:
    decision, risk, probabilities = parse_json_decision(
        'prefix {"decision":"safe","confidence":0.9} suffix'
    )
    assert decision == "safe"
    assert risk == pytest.approx(0.1)
    assert probabilities == pytest.approx({"safe": 0.9, "risky": 0.1})


def test_invalid_json_routes_without_probability() -> None:
    assert parse_json_decision("not json") == (None, None, {})
    assert parse_json_decision('{"decision":"safe","confidence":2}') == (None, None, {})


def test_anyjev_distribution_mapping_normalizes() -> None:
    assert distribution_to_probabilities({"safe": 3, "risky": 1}) == {
        "safe": 0.75,
        "risky": 0.25,
    }


def test_installed_anyjev_fake_backend_contract() -> None:
    backend = FakeBackend(
        lambda _state, option: 1.0 if option == "risky" else 0.0,
        position_bias=[1.5, -1.5],
    )
    decider = Decider(backend, level="L0", prior="content_free", adaptive_shifts=False)
    question = Question.choice("Is this action risky?", ["safe", "risky"], name="risk")
    answer = decider.decide({"action": "synthetic"}, [question], require="L0")["risk"]
    probabilities = distribution_to_probabilities(answer.distribution)
    assert probabilities["risky"] > probabilities["safe"]
    assert answer.level == "L0"


def test_decision_result_serializes_numpy_diagnostics() -> None:
    result = DecisionResult(
        case_id="fil-01-safe",
        strategy="l0",
        backend="fake",
        model="fake",
        model_revision="test",
        option_order=("safe", "risky"),
        decision="safe",
        probabilities={"safe": 0.9, "risky": 0.1},
        risk_probability=0.1,
        action="allow",
        parsed=True,
        latency_ms=1.0,
        diagnostics={"prior": np.array([0.25, 0.75]), "count": np.int64(2)},
    )
    assert result.to_dict()["diagnostics"] == {"prior": [0.25, 0.75], "count": 2}
