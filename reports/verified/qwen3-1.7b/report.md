# AnyJev Decision Lab: Agent-Safety Case Study

**Status:** Verified local-LLM run

Dataset: 80 synthetic cases  
Model: `Qwen/Qwen3-1.7B`  
Revision: `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`  
Backend: `hf`

## Results

| Strategy | Accuracy | Risky allow | Safe block | Review | Order flips | Median latency |
|---|---:|---:|---:|---:|---:|---:|
| json | 52.5% | 42.5% | 50.0% | 1.2% | 8.8% | 533.4745 ms |
| raw | 55.0% | 0.0% | 77.5% | 7.5% | 30.0% | 247.0413 ms |
| l0 | 71.2% | 0.0% | 45.0% | 10.0% | 0.0% | 419.1413 ms |

## Interpretation

Interpret safety misses together with review burden and automation coverage.

## Limitations

- The dataset contains 80 synthetic cases and is exploratory.
- Confidence values are not a guarantee of real-world safety.
- Latency varies by hardware, software versions, and model cache state.
- The simulator does not reproduce all consequences of real external tools.
