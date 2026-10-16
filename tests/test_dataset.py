from pathlib import Path

import pytest

from anyjev_agent_safety_lab.dataset import DatasetError, load_dataset, validate_dataset

DATASET = Path("data/agent_actions.json")


def test_public_dataset_contract() -> None:
    cases, _ = load_dataset(DATASET)
    assert len(cases) == 80
    assert sum(case.risky for case in cases) == 40
    assert len({case.pair_id for case in cases}) == 40


def test_labels_do_not_leak_into_model_state() -> None:
    cases, _ = load_dataset(DATASET)
    state = cases[0].model_state()
    assert "risky" not in state
    assert "rationale" not in state
    assert "risk_factors" not in state


def test_validation_rejects_missing_case() -> None:
    cases, _ = load_dataset(DATASET)
    with pytest.raises(DatasetError, match="requires 80 cases"):
        validate_dataset(cases[:-1], canonical=True)


def test_custom_matched_dataset_can_be_smaller() -> None:
    cases, _ = load_dataset(DATASET)
    validate_dataset(cases[:2])
