from pathlib import Path

from anyjev_agent_safety_lab.dataset import load_dataset
from anyjev_agent_safety_lab.simulator import simulate


def test_review_oracle_executes_safe_and_denies_risky() -> None:
    cases, _ = load_dataset(Path("data/agent_actions.json"))
    safe = next(case for case in cases if not case.risky)
    risky = next(case for case in cases if case.risky)
    safe_outcome = simulate(safe, "review")
    risky_outcome = simulate(risky, "review")
    assert safe_outcome.executed
    assert not risky_outcome.executed
    assert safe_outcome.success_with_oracle_review
    assert risky_outcome.success_with_oracle_review


def test_allowing_risky_action_is_harmful() -> None:
    cases, _ = load_dataset(Path("data/agent_actions.json"))
    risky = next(case for case in cases if case.risky)
    assert simulate(risky, "allow").harmful_execution
