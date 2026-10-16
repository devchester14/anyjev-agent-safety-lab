from anyjev_agent_safety_lab.metrics import compute_metrics, expected_calibration_error


def test_ece_empty_is_none() -> None:
    assert expected_calibration_error([]) is None


def test_metrics_handle_zero_automation() -> None:
    record = {
        "case_id": "fil-01-safe",
        "ground_truth_risky": False,
        "decision": {
            "strategy": "json",
            "option_order": ["safe", "risky"],
            "risk_probability": None,
            "decision": None,
            "action": "review",
            "latency_ms": 1.0,
        },
        "outcome": {
            "success_without_review": False,
            "success_with_oracle_review": True,
            "harmful_execution": False,
        },
    }
    metrics = compute_metrics({"results": [record]})["json"]
    assert metrics["accuracy"] is None
    assert metrics["policy"]["automated_error_rate"] is None
    assert metrics["format_failure_rate"] == 1.0
