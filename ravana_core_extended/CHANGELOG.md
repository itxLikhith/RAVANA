# Changelog

All notable changes to the RAVANA Core project.

## [0.2.0] - 2026-03-19 — Three Major Extensions

### Extension 1: Perception — Pluggable Feature Extractors

#### Added
- `BaseFeatureExtractor` abstract base class for custom extractors
- `MockFeatureExtractor` — lightweight fallback for testing
- `ResNetFeatureExtractor` — ResNet50-based image features (requires torch/torchvision)
- `Wav2VecFeatureExtractor` — Wav2Vec2-based audio features (requires transformers/torch/librosa)
- `TextFeatureExtractor` — bag-of-words with HMM (internal)
- Configuration flags: `use_resnet`, `use_wav2vec`, `force_mock`, `perception_device`
- `get_extractor_status()` method for monitoring

#### Changed
- `PerceptionModule.__init__()` now accepts extractor configuration
- `PerceptionModule.process()` returns `extractor_info` dict
- Visual/audio processing now routes through configured extractors

#### Technical Details
- Extractors implement `extract()`, `output_dim`, `is_available()`, `initialize()`
- ResNet50 extracts 2048-dim features from final layer
- Wav2Vec2 extracts 768-dim features mean-pooled over time
- Automatic fallback to MockFeatureExtractor on import/init failure

---

### Extension 2: Dual-Process Reasoning — Configurable MCTS

#### Added
- `RolloutPolicy` abstract base class for pluggable policies
- `SimpleRolloutPolicy` — fast heuristic rollouts (default)
- `ComplexRolloutPolicy` — multi-step simulation with state transitions
- `EthicalRolloutPolicy` — ethics-weighted reward shaping
- Configurable MCTS: `mcts_depth`, `mcts_simulations`
- `complexity_adaptive` mode — automatic depth adjustment
- `get_system2_config()` — retrieve current configuration
- Search statistics tracking: `mcts_depth`, `n_rollouts` in `ReasoningResult`

#### Changed
- `System2Reasoner._mcts()` now supports configurable tree depth
- `System2Reasoner` accepts `rollout_policy` parameter
- `DualProcessReasoner` supports `system2_mcts_depth`, `system2_mcts_simulations`
- Tree expansion with depth limit and child action generation

#### Technical Details
- Depth controls tree expansion iterations (default: 5, range: 3-10)
- Rollouts control UCB1 iterations (default: 50, range: 10-200)
- Complexity estimation based on query keywords and length
- Policies receive `query`, `action`, `context`, `rng` and return reward in [0, 1]

---

### Extension 3: Critical Thinking — Symbolic Constraint Solvers

#### Added
- `ConstraintSolver` abstract base class for logical reasoning
- `Constraint` dataclass — symbolic constraint representation
- `ConstraintViolation` dataclass — violation reporting
- `MockConstraintSolver` — heuristic-based fallback (always available)
- `Z3ConstraintSolver` — SMT-based solving (optional z3-solver dependency)
- Constraint management: `add_constraint()`, `clear_constraints()`
- `check_constraints()` — SAT checking with configured solver
- `validate_with_constraints()` — claim validation
- Constraint integration in `deduce()`, `abduce()`, `metamorphic_test()`
- `audit()` now includes constraint solver status

#### Changed
- `CriticalThinkingModule` accepts `use_z3` and `constraint_solver` parameters
- Strategy selection considers constraint solver availability
- Metamorphic testing can validate against constraints

#### Technical Details
- Mock solver uses keyword-based heuristics (~90% SAT rate)
- Z3 solver uses boolean variables for claims, supports unsat cores
- Both solvers implement `check_sat()` → `(is_satisfiable, violations)`
- Constraint validation integrated into falsification pipeline

---

### Extension 4: Extended Agent Integration

#### Added
- `RavanaAgent` accepts all extension configuration parameters
- `config` dict stores all configuration for reproducibility
- Extended `status()` returns:
  - `perception`: extractor status and configuration
  - `system2_config`: MCTS depth, rollouts, policy name
  - `critical_thinking`: solver name, availability, active constraints

#### Changed
- `RavanaAgent` initializes modules with extension configurations
- All base parameters preserved for backward compatibility

---

### Testing

#### Added
- 17 comprehensive tests covering all extensions
- `test_ravana_extended.py` — full test suite
- Test categories:
  - Perception extractors (5 tests)
  - Reasoning MCTS (4 tests)
  - Constraint solvers (4 tests)
  - Full integration (4 tests)

---

### Documentation

#### Added
- Comprehensive README with API reference
- Code examples for all new features
- Configuration parameter tables
- Custom extractor/policy/solver examples

---

### Backward Compatibility

- ✅ All base tests pass with extended agent
- ✅ Default configuration identical to base version
- ✅ No breaking changes to public APIs
- ✅ New parameters are all optional with sensible defaults

---

## [0.1.0] - 2026-03-19 — Base Implementation

### Initial Release

- PerceptionModule — entropy-based uncertainty, dual-confidence
- EmotionModule — VAD dynamics with Russell (1980) equations
- PsychologyModule — ACT-R + Cognitive Dissonance Engine (CDE)
- GlobalWorkspace — softmax attention bidding
- DualProcessReasoner — System 1/2 with MCTS and falsification
- BayesianBeliefTracker — belief state with theory of mind
- CriticalThinkingModule — MBFL testing, deduction, abduction
- RavanaAgent — full cognitive cycle with dream sabotage

---

## Future Roadmap

### v0.3.0 (Planned)
- [ ] Attention visualization tools
- [ ] Export to ONNX for production deployment
- [ ] Distributed multi-agent simulation
- [ ] LLM integration for natural language reasoning
- [ ] Emotional trajectory plotting

### v1.0.0 (Planned)
- [ ] Complete research paper implementation
- [ ] Benchmarking suite against human cognitive data
- [ ] Real-time interaction interface
- [ ] Integration with robotics control
