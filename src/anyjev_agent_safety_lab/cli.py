"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import cast

from .dataset import DatasetError, dataset_hash, load_dataset
from .models import StrategyName
from .reporting import build_report, verify_report
from .runner import benchmark, verify_decisions
from .showcase import QuestionKind, ShowcaseBackend, run_showcase

DEFAULT_DATASET = Path("data/agent_actions.json")
DEFAULT_MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"  # pragma: allowlist secret


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anyjev-decision-lab",
        description="Use AnyJev to obtain typed probability decisions from a local LLM.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-data", help="validate the synthetic dataset")
    validate.add_argument("--input", type=Path, default=DEFAULT_DATASET)
    validate.add_argument(
        "--canonical",
        action="store_true",
        help="require the shipped 80-case balance in addition to the reusable schema",
    )

    showcase = subparsers.add_parser(
        "showcase",
        help="ask your local LLM an AnyJev choice, yes/no, or score question",
    )
    showcase.add_argument("--state", required=True, help="the text or state to decide about")
    showcase.add_argument("--question", required=True, help="the typed question to ask")
    showcase.add_argument("--kind", choices=["choice", "noul", "score"], default="choice")
    showcase.add_argument("--options", nargs="*", default=[])
    showcase.add_argument("--bins", type=int, default=5)
    showcase.add_argument("--levels", nargs="+", choices=["raw", "L0"], default=["raw", "L0"])
    showcase.add_argument("--backend", choices=["hf", "vllm", "fixture"], default="hf")
    showcase.add_argument(
        "--model",
        default="Qwen/Qwen3-1.7B",
        help="local Hugging Face model ID (Qwen3-1.7B is the reference default)",
    )
    showcase.add_argument(
        "--revision", help="optional immutable Hugging Face revision for the selected model"
    )
    showcase.add_argument("--device", default="auto")
    showcase.add_argument("--base-url", help="OpenAI-compatible vLLM server URL")
    showcase.add_argument("--tokenizer", help="tokenizer model ID when vLLM uses a served alias")
    showcase.add_argument(
        "--api-key-env",
        default="VLLM_API_KEY",
        help="environment variable containing the vLLM API key",
    )

    run = subparsers.add_parser("benchmark", help="run the agent-safety case study")
    run.add_argument("--input", type=Path, default=DEFAULT_DATASET)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument(
        "--strategies", nargs="+", choices=["json", "raw", "l0"], default=["json", "raw", "l0"]
    )
    run.add_argument(
        "--backend",
        choices=["hf", "fixture"],
        default="hf",
        help="hf runs a real model; fixture is deterministic and reserved for CI/tests",
    )
    run.add_argument(
        "--model",
        default="Qwen/Qwen3-1.7B",
        help="local Hugging Face model ID (Qwen3-1.7B is the reference default)",
    )
    run.add_argument("--revision", default=DEFAULT_MODEL_REVISION)
    run.add_argument("--device", default="auto")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--allow-max", type=float, default=0.20)
    run.add_argument("--block-min", type=float, default=0.80)

    report = subparsers.add_parser("report", help="build JSON, Markdown, and HTML reports")
    report.add_argument("--decisions", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--site", type=Path)

    verify = subparsers.add_parser("verify", help="verify decisions and report consistency")
    verify.add_argument("--decisions", type=Path, required=True)
    verify.add_argument("--report", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process status."""
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate-data":
            cases, payload = load_dataset(args.input, canonical=True if args.canonical else None)
            print(f"valid: {len(cases)} cases, sha256={dataset_hash(payload)}")
            return 0
        if args.command == "showcase":
            result = run_showcase(
                state=args.state,
                question=args.question,
                kind=cast(QuestionKind, args.kind),
                options=args.options,
                bins=args.bins,
                levels=args.levels,
                backend=cast(ShowcaseBackend, args.backend),
                model=args.model,
                revision=args.revision,
                device=args.device,
                base_url=args.base_url,
                tokenizer=args.tokenizer,
                api_key_env=args.api_key_env,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.command == "benchmark":
            output = benchmark(
                input_path=args.input,
                output_dir=args.output,
                strategies=[cast(StrategyName, value) for value in args.strategies],
                backend="fake" if args.backend == "fixture" else "hf",
                model=args.model,
                revision=args.revision,
                device=args.device,
                seed=args.seed,
                allow_max=args.allow_max,
                block_min=args.block_min,
            )
            print(output)
            return 0
        if args.command == "report":
            outputs = build_report(args.decisions, args.output, args.site)
            for kind, path in outputs.items():
                print(f"{kind}: {path}")
            return 0
        decisions_ok, decisions_message = verify_decisions(args.decisions)
        if not decisions_ok:
            print(decisions_message, file=sys.stderr)
            return 1
        if args.report:
            report_ok, report_message = verify_report(args.decisions, args.report)
            if not report_ok:
                print(report_message, file=sys.stderr)
                return 1
            print(f"verified decisions and report: {report_message}")
        else:
            print(f"verified decisions: {decisions_message}")
        return 0
    except (DatasetError, ValueError, OSError, ModuleNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
