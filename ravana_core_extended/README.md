# RAVANA Core Extended — Reference Implementation v0.2.0

**RAVANA**: Advanced Cognitive Architecture for AGI Development  
*A pressure-shaped developmental system integrating human psychology, emotional intelligence, Bayesian reasoning, and cognitive dissonance theory*

---

## Version 0.2.0 — Three Major Extensions

This extended implementation adds three production-ready capabilities to the base RAVANA architecture:

### 1. Perception: Pluggable Feature Extractors
- **Abstract `BaseFeatureExtractor` interface** for custom extractors
- **ResNetFeatureExtractor**: Production image feature extraction (ResNet50)
- **Wav2VecFeatureExtractor**: Production audio feature extraction (Wav2Vec2)
- **MockFeatureExtractor**: Fallback for testing without heavy dependencies
- **Configuration flags**: `use_resnet`, `use_wav2vec`, `force_mock`

### 2. Dual-Process Reasoning: Configurable MCTS
- **Configurable MCTS depth** (`system2_mcts_depth`): 3-10 levels of tree search
- **Configurable rollouts** (`system2_mcts_simulations`): 10-200 simulations
- **Pluggable `RolloutPolicy` interface**: Simple, Complex, Ethical policies
- **Complexity-adaptive depth**: Automatic adjustment based on query complexity
- **Search statistics**: Track depth, rollouts, policy usage

### 3. Critical Thinking: Symbolic Constraint Solvers
- **Abstract `ConstraintSolver` interface** for logical reasoning
- **Z3ConstraintSolver**: SMT-based constraint solving (optional Z3 dependency)
- **MockConstraintSolver**: Heuristic-based fallback (always available)
- **Constraint integration**: Validated deduction, metamorphic testing
- **MBFL pipeline**: Constraints integrated into falsification

---

## Installation

### Basic Installation (NumPy only)
```bash
pip install numpy
```

### Production Features (optional)
```bash
# For ResNet image features
pip install torch torchvision

# For Wav2Vec audio features
pip install transformers torch librosa

# For Z3 constraint solving
pip install z3-solver
```

---

## Quick Start

### Backward-Compatible Usage
```python
from ravana_core import RavanaAgent

# Same as base version — all defaults
agent = RavanaAgent(name="RAVANA-0", seed=42)
result = agent.process(text="Is this ethical?")
print(result.decision)
```

### Extended Configuration
```python
from ravana_core import (
    RavanaAgent,
    ComplexRolloutPolicy,
    EthicalRolloutPolicy,
)

# Production agent with all features
agent = RavanaAgent(
    name="RAVANA-Production",
    seed=42,
    # Perception configuration
    use_resnet=True,
    use_wav2vec=True,
    perception_device="cuda",
    # Reasoning configuration
    system2_mcts_depth=8,
    system2_mcts_simulations=100,
    system2_rollout_policy=EthicalRolloutPolicy(),
    # Critical thinking configuration
    use_z3_solver=True,
)

# Run with multimodal input
result = agent.process(
    text="Should we implement UBI?",
    visual=np.random.rand(224, 224, 3),  # Will use ResNet50
    audio=np.random.randn(16000),         # Will use Wav2Vec2
)

# Check extended status
status = agent.status()
print(f"Extractors: {status['perception']}")
print(f"MCTS config: {status['system2_config']}")
print(f"Solver: {status['critical_thinking']['solver']}")
```

---

## API Reference

### Perception Module

```python
from ravana_core import PerceptionModule

# Default (mock extractors)
perc = PerceptionModule(seed=42)

# With production extractors
perc = PerceptionModule(
    seed=42,
    use_resnet=True,       # Enable ResNet50
    use_wav2vec=True,      # Enable Wav2Vec2
    perception_device="cuda",
)

# Force mocks for testing
perc = PerceptionModule(seed=42, force_mock=True)

# Process multimodal input
result = perc.process(
    text="Input text",
    visual=np.array(...),  # Image (H, W, C)
    audio=np.array(...),   # Audio waveform
)

# Check extractor status
status = perc.get_extractor_status()
```

### Dual-Process Reasoning

```python
from ravana_core import (
    DualProcessReasoner,
    SimpleRolloutPolicy,
    ComplexRolloutPolicy,
    EthicalRolloutPolicy,
)

# Default configuration
reasoner = DualProcessReasoner(seed=42)

# Deep search for complex decisions
reasoner = DualProcessReasoner(
    seed=42,
    mcts_depth=8,
    mcts_simulations=100,
    rollout_policy=ComplexRolloutPolicy(simulation_steps=7),
    complexity_adaptive=True,
)

# Reason with context
result = reasoner.reason(
    query="Is this ethical?",
    context={"U": 0.5, "dissonance": 0.3, "ethics_signal": 0.8},
    beliefs=[],
)

print(f"System: {result.system}")
print(f"Depth: {result.mcts_depth}")
print(f"Rollouts: {result.n_rollouts}")
```

### Critical Thinking with Constraints

```python
from ravana_core import (
    CriticalThinkingModule,
    Constraint,
    Z3ConstraintSolver,
)

# With Z3 solver (if available)
ct = CriticalThinkingModule(use_z3=True)

# With mock solver (always works)
ct = CriticalThinkingModule()

# Add constraints
ct.add_constraint(Constraint(
    name="harm_minimization",
    expression="actions must minimize harm",
    description="Core ethical constraint",
))

# Check constraint satisfaction
is_sat, violations = ct.check_constraints()

# Validate claim against constraints
is_valid, violation = ct.validate_with_constraints("Always tell the truth")

# Deduction with constraint validation
conclusion = ct.deduce(["premise1", "premise2"])
```

### Custom Extractors

```python
from ravana_core import BaseFeatureExtractor

class MyExtractor(BaseFeatureExtractor):
    @property
    def output_dim(self) -> int:
        return 512
    
    def extract(self, input_data: Any) -> np.ndarray:
        # Your feature extraction logic
        features = np.random.rand(self.output_dim)
        return features / features.sum()
```

### Custom Rollout Policies

```python
from ravana_core import RolloutPolicy

class MyPolicy(RolloutPolicy):
    @property
    def name(self) -> str:
        return "MyPolicy"
    
    def rollout(self, query, action, context, rng):
        # Your rollout simulation
        return 0.75  # reward in [0, 1]
```

### Custom Constraint Solvers

```python
from ravana_core import ConstraintSolver, ConstraintViolation

class MySolver(ConstraintSolver):
    @property
    def name(self) -> str:
        return "MySolver"
    
    def check_sat(self, constraints, context):
        # Your SAT checking logic
        return True, []  # satisfiable, no violations
    
    def validate_claim(self, claim, constraints):
        # Your claim validation logic
        return True, None  # valid, no violation
```

---

## Architecture

### Module Dependencies
```
RavanaAgent
├── PerceptionModule (EXTENDED)
│   ├── BaseFeatureExtractor (ABC)
│   ├── TextFeatureExtractor
│   ├── ResNetFeatureExtractor (optional)
│   └── Wav2VecFeatureExtractor (optional)
├── EmotionModule
├── PsychologyModule
├── GlobalWorkspace
├── DualProcessReasoner (EXTENDED)
│   ├── System1Reasoner
│   └── System2Reasoner (configurable MCTS)
│       └── RolloutPolicy (pluggable)
├── BayesianBeliefTracker
└── CriticalThinkingModule (EXTENDED)
    └── ConstraintSolver (pluggable)
```

---

## Testing

### Run All Tests
```bash
cd ravana_core_extended
python test_ravana_extended.py
```

### Test Categories
- **17 comprehensive tests** covering all extensions
- **Backward compatibility** tests (base functionality)
- **Mock-based tests** (no heavy dependencies)
- **Production feature tests** (with fallback)

---

## Configuration Summary

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_resnet` | `False` | Enable ResNet50 image features |
| `use_wav2vec` | `False` | Enable Wav2Vec2 audio features |
| `force_mock` | `False` | Force mock extractors (testing) |
| `perception_device` | `"cpu"` | Device for neural extractors |
| `system2_mcts_depth` | `5` | Tree search depth (3-10) |
| `system2_mcts_simulations` | `50` | Number of rollouts (10-200) |
| `system2_rollout_policy` | `SimpleRolloutPolicy()` | Rollout simulation policy |
| `use_z3_solver` | `False` | Enable Z3 constraint solving |

---

## Citation

```bibtex
@software{ravana_core_extended_2026,
  author = {Seemala, Likhith Sai},
  title = {RAVANA Core Extended: AGI Architecture with Production Features},
  year = {2026},
  version = {0.2.0},
  url = {https://github.com/itxLikhith/RAVANA-AGI-Research}
}
```

---

## License

Creative Commons Attribution 4.0 International (CC BY 4.0)

---

**Author**: Likhith Sai Seemala  
**ORCID**: https://orcid.org/0009-0004-6416-8918  
**Based on**: Seemala, L. S. (2026). RAVANA: Advanced Cognitive Architecture for AGI Development. Zenodo. https://doi.org/10.5281/zenodo.18309746
