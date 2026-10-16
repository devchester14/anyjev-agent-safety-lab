import pytest

from anyjev_agent_safety_lab.policy import route_probability


@pytest.mark.parametrize(
    ("probability", "expected"),
    [
        (None, "review"),
        (0.0, "allow"),
        (0.20, "allow"),
        (0.200001, "review"),
        (0.799999, "review"),
        (0.80, "block"),
        (1.0, "block"),
    ],
)
def test_policy_boundaries(probability: float | None, expected: str) -> None:
    assert route_probability(probability) == expected


def test_policy_rejects_invalid_probability() -> None:
    with pytest.raises(ValueError):
        route_probability(1.1)


def test_custom_thresholds() -> None:
    assert route_probability(0.35, allow_max=0.40, block_min=0.90) == "allow"
    assert route_probability(0.90, allow_max=0.40, block_min=0.90) == "block"
    with pytest.raises(ValueError, match="thresholds"):
        route_probability(0.5, allow_max=0.8, block_min=0.2)
