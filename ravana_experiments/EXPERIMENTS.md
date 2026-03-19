# RAVANA Classroom Experiments

## Overview

This directory contains a comprehensive benchmark environment for evaluating RAVANA's cognitive architecture against baseline agents in a simulated classroom setting.

## Quick Start

```bash
# Run comprehensive test suite
python test_env_and_metrics.py

# Run experiments (quick test, 100 episodes)
python run_ravana_experiments.py --quick

# Run full experiments (1000 episodes)
python run_ravana_experiments.py --episodes 1000
```

## Directory Structure

```
ravana_experiments/
├── classroom_env/          # Simulated classroom environment
│   ├── __init__.py
│   ├── environment.py      # Core environment with student profiles
│   ├── tasks.py           # MCQ, open-ended, ethical dilemma tasks
│   └── rewards.py         # Reward signal definitions
├── metrics/               # Paper-style metrics
│   ├── __init__.py
│   ├── paper_metrics.py   # Dissonance, identity, wisdom scores
│   └── comparison_metrics.py  # Baseline implementations
├── run_ravana_experiments.py  # Main experiment runner
├── test_env_and_metrics.py    # Comprehensive tests
└── EXPERIMENTS.md        # This file
```

## Environment Components

### Task Types

1. **MCQ (Multiple Choice)**: Factual knowledge questions
   - 4 options, single correct answer
   - Evaluated with student ability noise
   
2. **Open-Ended**: Explanation and reasoning questions
   - Rubric-based keyword evaluation
   - Explanation quality scoring
   
3. **Ethical Dilemmas**: Value-based decision scenarios
   - Options with value alignment scores
   - Student value preferences affect evaluation

### Student Profiles

Each student has:
- `ability`: Base competence [0, 1]
- `demographic_group`: Group identifier ("A", "B", etc.)
- `value_preferences`: {value: weight} mapping
- `learning_rate`: Improvement speed

### Reward Signals

Three-component reward:
- **Primary**: Correctness + explanation quality
- **Social Norm**: Alignment with values and norms
- **Fairness**: Inverse of demographic disparity

## Agent Configurations

### 1. RAVANA-0 (Baseline)
- Mock perception (no heavy dependencies)
- Simple MCTS (depth=3, sims=20)
- Mock constraint solver
- All default pressures enabled

### 2. RAVANA-Pro (Production)
- ResNet50 + Wav2Vec2 extractors (if available)
- Deep MCTS (depth=8, sims=100)
- Z3 solver (if available)
- All pressures at full strength

### 3. RAVANA-Ablated
- Same as RAVANA-0 but with:
  - Dissonance reappraisal disabled
  - Or: Constraint solver disabled
  - Or: Dream sabotage disabled

### 4. Baselines
- **Naive RL**: Simple Q-learning, no cognitive architecture
- **LLM Stub**: Pattern matching, no structured reasoning
- **Rule-Based**: Fixed heuristics, no adaptation

## Metrics

### Paper-Style Metrics (from RAVANA paper)

1. **Dissonance Reduction**: Target ~0.8 → ~0.2
   - Measures cognitive coherence improvement
   
2. **Identity Strength**: Target ~0.3 → ~0.85
   - Measures value commitment consolidation
   
3. **Generalization Accuracy**: Target ~0.9
   - Accuracy on held-out tasks
   
4. **Transfer Efficiency**: Target ~0.8
   - Speed of knowledge transfer
   
5. **Demographic Parity Gap**: Target ~20 → ~5 points
   - Fairness across demographic groups
   
6. **Wisdom Score**: Composite metric
   - Epistemic humility + integrity + empathy + meaning

### Per-Episode Logging

Each episode logs:
- Cognitive dissonance D
- Identity strength index
- Mean/volatility confidence
- Emotional VAD state
- Reasoning system used (System 1/2)
- MCTS depth and rollouts
- Pressure triggers (falsification, dissonance, dream, meaning)
- Constraint violations
- Demographic parity gap
- All reward components

## Running Experiments

### Basic Usage

```bash
# Quick test (100 episodes)
python run_ravana_experiments.py --quick

# Full run (1000 episodes)
python run_ravana_experiments.py

# Custom episodes
python run_ravana_experiments.py --episodes 500

# Custom output directory
python run_ravana_experiments.py --output my_results

# Custom seed
python run_ravana_experiments.py --seed 123
```

### Output Files

Results are saved in the specified output directory (default: `results/`):

```
results/
├── logs/
│   ├── ravana_0.csv           # RAVANA-0 episode logs
│   ├── ravana_pro.csv         # RAVANA-Pro episode logs
│   ├── ravana_ablated.csv     # Ablated variant logs
│   ├── baseline_rl.csv        # Naive RL logs
│   ├── baseline_llm.csv       # LLM stub logs
│   └── baseline_rule.csv      # Rule-based logs
├── plots/
│   └── metrics_trajectories.png  # 6-panel metric plots
├── comparison.txt             # RAVANA vs baselines comparison
└── summary.json               # Final metrics summary
```

## Interpreting Results

### Metric Trajectories

The plots show:
1. **Dissonance over time**: Should decrease from ~0.8 to ~0.2
2. **Identity strength**: Should increase from ~0.3 to ~0.85
3. **Learning curve**: Accuracy should improve over episodes
4. **Demographic parity**: Gap should shrink from ~20% to ~5%
5. **Accuracy comparison**: Bar chart of all configurations
6. **Meaning scores**: Should show positive trajectory

### Comparison Table

The comparison output shows:
- RAVANA's performance on all paper metrics
- Baseline performance (accuracy, fairness, generalization)
- Key advantages of cognitive architecture

### Expected Results

**RAVANA-Pro should show:**
- Fast dissonance reduction
- Strong identity strength increase
- High generalization on held-out tasks
- Low demographic parity gap
- High wisdom score

**Baselines should show:**
- Naive RL: Decent accuracy but poor generalization, no dissonance handling
- LLM Stub: Moderate accuracy but inconsistent identity
- Rule-Based: Lower accuracy, rigid behavior, no learning

## Hyperparameters

### Environment
- `max_episodes`: 1000 (default), 100 (quick)
- `held_out_ratio`: 0.2 (20% held-out for evaluation)
- `fairness_weight`: 0.3 (importance of fairness in reward)
- `window_size`: 100 (for metric smoothing)

### RAVANA-0
- MCTS depth: 3
- MCTS simulations: 20
- All pressures enabled

### RAVANA-Pro
- MCTS depth: 8
- MCTS simulations: 100
- ResNet/Wav2Vec enabled (falls back to mock if unavailable)
- Z3 solver enabled (falls back to mock if unavailable)

### Ablated Variants
- Dissonance reappraisal: Disabled
- Constraint solver: Disabled
- Dream sabotage: Disabled

## Extending the Benchmark

### Adding New Task Types

```python
from classroom_env import Task, TaskType

class NewTask(Task):
    def evaluate_answer(self, answer: str, student) -> bool:
        # Custom evaluation logic
        pass

# Add to TaskBank
task_bank.tasks["new_task_id"] = NewTask(...)
```

### Adding New Metrics

```python
from metrics.paper_metrics import RAVANAMetrics

class ExtendedMetrics(RAVANAMetrics):
    def compute_new_metric(self):
        # Custom metric computation
        pass
```

### Adding New Baselines

```python
from metrics.comparison_metrics import BaselineResult, BaselineType

def my_baseline(env, n_episodes, seed):
    # Baseline implementation
    return BaselineResult(
        baseline_type=BaselineType.CUSTOM,
        logs=logs,
        metrics=metrics,
        comparison_summary=summary,
    )
```

## Testing

### Running Tests

```bash
# All tests
python test_env_and_metrics.py

# Should output:
# Passed: 8/8
```

### Test Coverage

1. **Task Bank**: Initialization and task retrieval
2. **MCQ Evaluation**: Correct/incorrect with ability noise
3. **Open-Ended**: Keyword-based rubric evaluation
4. **Ethical Dilemma**: Value-aligned option selection
5. **Environment Step**: Full episode execution
6. **Reward Computation**: Multi-component reward calculation
7. **Metrics**: All paper metrics computation
8. **Baselines**: All three baseline implementations

## Citation

```bibtex
@software{ravana_experiments_2026,
  author = {Seemala, Likhith Sai},
  title = {RAVANA Classroom Experiments: Benchmarking Cognitive Architecture},
  year = {2026},
  url = {https://github.com/itxLikhith/RAVANA-AGI-Research}
}
```

## Troubleshooting

### Import Errors
Ensure paths are correct:
```python
sys.path.insert(0, '/path/to/ravana_core_extended')
sys.path.insert(0, '/path/to/ravana_experiments')
```

### Missing Dependencies
Optional packages (torch, transformers, z3-solver) have mock fallbacks. The code works with just NumPy.

### Out of Memory
Reduce `n_episodes` or `mcts_simulations` for RAVANA-Pro.

### Slow Execution
Use `--quick` flag for faster testing, or reduce MCTS depth/simulations.

## Contact

For questions about the experiments, refer to the main RAVANA paper or the implementation summary in the extended core package.
