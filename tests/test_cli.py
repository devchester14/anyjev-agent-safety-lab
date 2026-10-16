from __future__ import annotations

import json
from pathlib import Path

import pytest

from anyjev_agent_safety_lab.cli import main


def test_showcase_exposes_all_typed_question_kinds(capsys: pytest.CaptureFixture[str]) -> None:
    examples = [
        ("choice", ["billing", "technical", "security"]),
        ("noul", []),
        ("score", []),
    ]
    for kind, options in examples:
        arguments = [
            "showcase",
            "--backend",
            "fixture",
            "--model",
            "local/example-causal-lm",
            "--revision",
            "example-revision",
            "--state",
            "A customer reports a duplicate card charge.",
            "--question",
            "What decision should be made?",
            "--kind",
            kind,
        ]
        if options:
            arguments.extend(["--options", *options])
        assert main(arguments) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["question"]["kind"] == kind
        assert set(payload["decisions"]) == {"raw", "L0"}
        assert payload["backend"] == "fixture"
        assert payload["model"] == "local/example-causal-lm"
        assert payload["model_revision"] == "example-revision"
        assert "not a model result" in payload["notice"]


def test_cli_describes_the_local_llm_story(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert "local LLM" in output
    assert "anyjev-decision-lab" in output


def test_vllm_showcase_requires_a_server_url(capsys: pytest.CaptureFixture[str]) -> None:
    status = main(
        [
            "showcase",
            "--backend",
            "vllm",
            "--model",
            "local/served-model",
            "--state",
            "A proposed action",
            "--question",
            "Does this need review?",
            "--kind",
            "noul",
        ]
    )
    assert status == 2
    assert "requires --base-url" in capsys.readouterr().err


def test_fixture_end_to_end(tmp_path: Path) -> None:
    output = tmp_path / "run"
    site = tmp_path / "site" / "index.html"
    assert main(["validate-data", "--input", "data/agent_actions.json"]) == 0
    assert (
        main(
            [
                "benchmark",
                "--input",
                "data/agent_actions.json",
                "--output",
                str(output),
                "--backend",
                "fixture",
            ]
        )
        == 0
    )
    decisions = output / "decisions.json"
    assert (
        main(
            [
                "report",
                "--decisions",
                str(decisions),
                "--output",
                str(output),
                "--site",
                str(site),
            ]
        )
        == 0
    )
    assert (
        main(["verify", "--decisions", str(decisions), "--report", str(output / "report.json")])
        == 0
    )
    report = json.loads((output / "report.json").read_text())
    assert not report["verified_real_model_run"]
    site_text = site.read_text()
    assert "Results pending" in site_text
    assert "Turn your local LLM into a typed decision model" in site_text
