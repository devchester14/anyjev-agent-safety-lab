# Synthetic agent-action dataset

`data/agent_actions.json` contains 80 authored, synthetic cases: 10 safe/risky matched
pairs in each of filesystem, database, communications, and deployment.

Each pair changes a material property such as scope, destination, environment, or
reversibility. Labels and rationales are retained for evaluation but removed by
`SafetyCase.model_state()` before inference.

This is an exploratory synthetic dataset, not a sample of production traffic. It contains no
private prompts, logs, tickets, credentials, or customer data. Rebuild it with:

```bash
python scripts/build_dataset.py
uv run anyjev-decision-lab validate-data
```

For reuse, create any number of matched pairs using the same schema and a different
dataset `name`. Each pair must contain one safe and one risky action in the same domain.
The ordinary validator accepts that reusable profile; pass `--canonical` only when
validating the exact 80-case benchmark shipped here.

Licensed under Apache-2.0 with the rest of this repository.
