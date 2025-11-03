# Evaluation Results Summary (Mock Data)

## Top-K Accuracy Comparison

| Condition | Top-1 | Top-3 | Top-5 |
|-----------|-------|-------|-------|
| Baseline Mixed Skill | 68.5% | 85.2% | 91.7% |
| Expert Only 2500 | 74.2% | 89.1% | 94.3% |
| Game Theoretic Reg | 76.8% | 90.5% | 95.1% |

## Stockfish Alignment (KL-Divergence)

| Condition | Mean KL | Std KL |
|-----------|---------|--------|
| Baseline Mixed Skill | 1.85 | 0.52 |
| Expert Only 2500 | 1.52 | 0.41 |
| Game Theoretic Reg | 1.28 | 0.35 |

## Sample Complexity (for 70% Accuracy)

| Condition | Samples Needed | Reduction vs Baseline |
|-----------|----------------|----------------------|
| Baseline | 14,411 | 0.0% |
| Expert Only | 5,738 | 60.2% |
| Game Theoretic | 4,417 | 69.3% |

## Key Findings (Mock Data)

1. **Expert-only approach** achieves 60.2% reduction in sample complexity
2. **Game-theoretic regularization** achieves 69.3% total reduction
3. **Top-1 accuracy** improves from 68.5% (baseline) to 76.8% (game-theoretic)

---
*Note: These are mock results for demonstration purposes*