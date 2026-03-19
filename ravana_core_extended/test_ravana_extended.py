"""
RAVANA Core Extended — Comprehensive Test Suite v0.2.0
Tests all three extensions:
  1. Perception: pluggable extractors (Base, ResNet, Wav2Vec, Mock)
  2. Reasoning: configurable MCTS depth, rollouts, rollout policies
  3. Critical Thinking: constraint solvers (Z3, Mock)

Maintains backward compatibility with all existing tests.
"""

import sys
import numpy as np

# Test both the base and extended versions
sys.path.insert(0, '/home/.z/workspaces/con_it6dxBHxJqtbptXy/ravana_core_extended')

from ravana_core import (
    # Core modules
    PerceptionModule,
    EmotionModule,
    PsychologyModule,
    Action,
    GlobalWorkspace,
    DualProcessReasoner,
    BayesianBeliefTracker,
    CriticalThinkingModule,
    RavanaAgent,
    # NEW: Perception extractors (v0.2.0)
    BaseFeatureExtractor,
    MockFeatureExtractor,
    ResNetFeatureExtractor,
    Wav2VecFeatureExtractor,
    TextFeatureExtractor,
    # NEW: Reasoning components (v0.2.0)
    RolloutPolicy,
    SimpleRolloutPolicy,
    ComplexRolloutPolicy,
    EthicalRolloutPolicy,
    System2Reasoner,
    # NEW: Critical thinking solvers (v0.2.0)
    Constraint,
    ConstraintSolver,
    MockConstraintSolver,
    Z3ConstraintSolver,
)


# =============================================================================
# EXTENSION TEST 1: Perception with Pluggable Extractors
# =============================================================================

def test_perception_mock_extractors():
    """Test perception with mock extractors (no dependencies)."""
    print("=" * 70)
    print("TEST 1a: Perception with Mock Extractors (backward compatible)")
    print("=" * 70)

    perc = PerceptionModule(
        seed=42,
        use_resnet=False,
        use_wav2vec=False,
        force_mock=True,
    )

    # Test text-only input
    result = perc.process(text="This is a test sentence")
    assert "features" in result
    assert "U" in result
    assert "extractor_info" in result
    assert result["extractor_info"]["text"] == "TextFeatureExtractor"
    assert result["extractor_info"]["visual"] == "none"
    
    # Check extractor status
    status = perc.get_extractor_status()
    assert status["text"]["available"] is True
    assert "MockFeatureExtractor" in status["visual"]["name"]
    
    print(f"✓ Text processing: entropy={result['U']:.3f}, conf={result['mean_conf']:.3f}")
    print(f"✓ Extractor status: {status['config']}")
    print("✓ Mock extractors test passed\n")
    return True


def test_perception_with_visual_audio():
    """Test perception with visual and audio inputs."""
    print("=" * 70)
    print("TEST 1b: Perception with Visual and Audio Inputs")
    print("=" * 70)

    perc = PerceptionModule(seed=42, force_mock=True)

    # Create mock visual input (64x64x3 image)
    visual_input = np.random.rand(64, 64, 3)
    
    # Create mock audio input (1000 samples)
    audio_input = np.random.randn(1000)

    result = perc.process(
        text="Testing multimodal input",
        visual=visual_input,
        audio=audio_input,
    )

    assert "features" in result
    assert result["extractor_info"]["visual"] == "MockFeatureExtractor"
    assert result["extractor_info"]["audio"] == "MockFeatureExtractor"
    
    print(f"✓ Multimodal processing: entropy={result['U']:.3f}")
    print(f"✓ Feature dims: {result['features'].shape}")
    print(f"✓ Extractors used: {result['extractor_info']}")
    print("✓ Multimodal test passed\n")
    return True


def test_resnet_extractor_stubs():
    """Test ResNet extractor stubs (without torch dependency)."""
    print("=" * 70)
    print("TEST 1c: ResNet Extractor (stubs/mock fallback)")
    print("=" * 70)

    # This will try to use ResNet but fall back to mock
    perc = PerceptionModule(
        seed=42,
        use_resnet=True,  # Will try but fallback to mock
        force_mock=True,  # Force mock for testing
    )

    visual_input = np.random.rand(224, 224, 3)
    result = perc.process(visual=visual_input)

    status = perc.get_extractor_status()
    print(f"✓ ResNet attempt status: {status['visual']}")
    print(f"✓ Visual entropy: {result['U']:.3f}")
    print("✓ ResNet stubs test passed\n")
    return True


def test_wav2vec_extractor_stubs():
    """Test Wav2Vec extractor stubs (without transformers dependency)."""
    print("=" * 70)
    print("TEST 1d: Wav2Vec Extractor (stubs/mock fallback)")
    print("=" * 70)

    perc = PerceptionModule(
        seed=42,
        use_wav2vec=True,  # Will try but fallback to mock
        force_mock=True,   # Force mock for testing
    )

    audio_input = np.random.randn(16000)  # 1 second at 16kHz
    result = perc.process(audio=audio_input)

    status = perc.get_extractor_status()
    print(f"✓ Wav2Vec attempt status: {status['audio']}")
    print(f"✓ Audio entropy: {result['U']:.3f}")
    print("✓ Wav2Vec stubs test passed\n")
    return True


def test_extractor_base_class():
    """Test BaseFeatureExtractor ABC interface."""
    print("=" * 70)
    print("TEST 1e: BaseFeatureExtractor Interface")
    print("=" * 70)

    # Test that MockFeatureExtractor properly implements the interface
    extractor = MockFeatureExtractor(output_dim=128, seed=42)
    
    assert extractor.output_dim == 128
    assert extractor.is_available() is True
    assert extractor.initialize() is True
    
    test_input = "any input"
    features = extractor.extract(test_input)
    
    assert len(features) == 128
    assert abs(features.sum() - 1.0) < 1e-5  # normalized
    
    print(f"✓ BaseFeatureExtractor interface: dim={extractor.output_dim}")
    print(f"✓ Feature extraction: shape={features.shape}, sum={features.sum():.5f}")
    print("✓ Base extractor interface test passed\n")
    return True


# =============================================================================
# EXTENSION TEST 2: Dual-Process Reasoning with Configurable MCTS
# =============================================================================

def test_system2_configurable_mcts():
    """Test System 2 with configurable MCTS depth and rollouts."""
    print("=" * 70)
    print("TEST 2a: System 2 Configurable MCTS")
    print("=" * 70)

    # Test with shallow search (fast)
    shallow_reasoner = System2Reasoner(
        mcts_simulations=10,
        mcts_depth=2,
        seed=42,
    )

    result = shallow_reasoner.reason(
        query="Should we implement UBI?",
        context={"U": 0.5, "dissonance": 0.3},
        beliefs=[],
    )

    print(f"✓ Shallow search: depth={result.mcts_depth}, rollouts={result.n_rollouts}")
    print(f"✓ System: {result.system}, confidence: {result.confidence:.3f}")

    # Test with deep search (thorough)
    deep_reasoner = System2Reasoner(
        mcts_simulations=100,
        mcts_depth=8,
        seed=42,
    )

    result_deep = deep_reasoner.reason(
        query="Should we implement UBI?",
        context={"U": 0.5, "dissonance": 0.3},
        beliefs=[],
    )

    print(f"✓ Deep search: depth={result_deep.mcts_depth}, rollouts={result_deep.n_rollouts}")
    print(f"✓ System: {result_deep.system}, confidence: {result_deep.confidence:.3f}")
    
    assert result_deep.mcts_depth >= result.mcts_depth
    print("✓ Configurable MCTS test passed\n")
    return True


def test_rollout_policies():
    """Test different rollout policies."""
    print("=" * 70)
    print("TEST 2b: Pluggable Rollout Policies")
    print("=" * 70)

    policies = [
        SimpleRolloutPolicy(),
        ComplexRolloutPolicy(simulation_steps=5),
        EthicalRolloutPolicy(),
    ]

    for policy in policies:
        reasoner = System2Reasoner(
            mcts_simulations=20,
            mcts_depth=3,
            rollout_policy=policy,
            seed=42,
        )

        result = reasoner.reason(
            query="Is this ethical?",
            context={
                "U": 0.6,
                "dissonance": 0.4,
                "ethics_signal": 0.8,
                "safety_signal": 0.5,
            },
            beliefs=[],
        )

        print(f"✓ Policy '{policy.name}': confidence={result.confidence:.3f}")
        print(f"  Steps: {result.reasoning_steps[0]}")

    # Verify stats are tracked
    stats = reasoner.get_search_stats()
    print(f"✓ Search stats: {stats}")
    print("✓ Rollout policies test passed\n")
    return True


def test_complexity_adaptive_reasoning():
    """Test complexity-adaptive System 2 depth adjustment."""
    print("=" * 70)
    print("TEST 2c: Complexity-Adaptive Reasoning")
    print("=" * 70)

    reasoner = DualProcessReasoner(
        seed=42,
        complexity_adaptive=True,
        mcts_depth=5,
        mcts_simulations=50,
    )

    # Simple query → shallow search
    simple_result = reasoner.reason(
        query="What is 2+2?",
        context={"U": 0.1, "mean_conf": 0.9, "volatility_conf": 0.05},
    )

    # Complex ethical query → deep search
    complex_result = reasoner.reason(
        query="Should we implement universal basic income given ethical trade-offs?",
        context={"U": 0.6, "mean_conf": 0.5, "volatility_conf": 0.2},
    )

    print(f"✓ Simple query: system={simple_result.system}, depth={simple_result.mcts_depth}")
    print(f"✓ Complex query: system={complex_result.system}, depth={complex_result.mcts_depth}")
    
    # Complex queries should force System 2
    assert complex_result.system == "system2"
    print("✓ Complexity-adaptive test passed\n")
    return True


def test_dual_process_reasoner_config():
    """Test getting System 2 configuration."""
    print("=" * 70)
    print("TEST 2d: System 2 Configuration Retrieval")
    print("=" * 70)

    policy = ComplexRolloutPolicy(simulation_steps=7)
    reasoner = DualProcessReasoner(
        seed=42,
        mcts_depth=7,
        mcts_simulations=75,
        rollout_policy=policy,
    )

    config = reasoner.get_system2_config()
    print(f"✓ System 2 config: {config}")
    
    assert config["mcts_depth"] == 7
    assert config["mcts_simulations"] == 75
    assert config["rollout_policy"] == policy.name
    
    print("✓ System 2 config test passed\n")
    return True


# =============================================================================
# EXTENSION TEST 3: Critical Thinking with Constraint Solvers
# =============================================================================

def test_mock_constraint_solver():
    """Test MockConstraintSolver (always available)."""
    print("=" * 70)
    print("TEST 3a: Mock Constraint Solver")
    print("=" * 70)

    solver = MockConstraintSolver(seed=42)
    
    assert solver.is_available() is True
    assert solver.name == "MockConstraintSolver"

    # Add constraints
    constraints = [
        Constraint("honesty", "must be honest", "Honesty requirement"),
        Constraint("safety", "must not cause harm", "Safety requirement"),
    ]

    # Check satisfiability
    is_sat, violations = solver.check_sat(constraints, context={})
    print(f"✓ Constraints satisfiable: {is_sat}")
    print(f"✓ Violations: {len(violations)}")

    # Validate claim
    claim = "Always tell the truth"
    is_valid, violation = solver.validate_claim(claim, constraints)
    print(f"✓ Claim validation: valid={is_valid}")

    print("✓ Mock solver test passed\n")
    return True


def test_z3_constraint_solver_stubs():
    """Test Z3 solver (will use mock fallback without z3)."""
    print("=" * 70)
    print("TEST 3b: Z3 Constraint Solver (stubs/mock fallback)")
    print("=" * 70)

    solver = Z3ConstraintSolver()
    
    # Will report availability based on z3-solver package
    available = solver.is_available()
    print(f"✓ Z3 available: {available}")
    print(f"✓ Solver name: {solver.name}")

    # If z3 not available, will use mock fallback
    constraints = [
        Constraint("mutual_exclusion", "A and not A", "Test constraint"),
    ]

    is_sat, violations = solver.check_sat(constraints)
    print(f"✓ Check result: satisfiable={is_sat}, violations={len(violations)}")
    
    print("✓ Z3 solver stubs test passed\n")
    return True


def test_critical_thinking_with_constraints():
    """Test CriticalThinkingModule with constraint integration."""
    print("=" * 70)
    print("TEST 3c: Critical Thinking with Constraint Integration")
    print("=" * 70)

    ct = CriticalThinkingModule(
        seed=42,
        use_z3=False,  # Use mock for testing
    )

    # Add facts
    ct.add_fact("all humans are mortal")
    ct.add_fact("socrates is human")

    # Add constraints
    ct.add_constraint(Constraint("consistency", "beliefs must be consistent", "Logical consistency"))
    ct.add_constraint(Constraint("evidence", "claims require evidence", "Evidence requirement"))

    # Check constraints
    is_sat, violations = ct.check_constraints()
    print(f"✓ Constraints satisfiable: {is_sat}")
    print(f"✓ Active constraints: {len(ct.active_constraints)}")

    # Validate claim against constraints
    claim = "Socrates is immortal"
    is_valid, violation = ct.validate_with_constraints(claim)
    print(f"✓ Claim '{claim[:30]}...': valid={is_valid}")

    # Test deduction with constraint validation
    conclusion = ct.deduce(["all humans are mortal", "socrates is human"])
    print(f"✓ Deduction: {conclusion}")

    # Test strategy selection with constraints
    strategy = ct.select_strategy({
        "dissonance": 0.8,
        "novelty": 0.4,
        "uncertainty": 0.5,
    })
    print(f"✓ Selected strategy: {strategy}")

    # Audit
    audit = ct.audit()
    print(f"✓ Audit: solver={audit['solver']}, n_constraints={audit['n_constraints']}")
    
    assert audit["solver"] == "MockConstraintSolver"
    assert audit["n_constraints"] == 2
    print("✓ Critical thinking with constraints test passed\n")
    return True


def test_metamorphic_with_constraints():
    """Test metamorphic testing with constraint validation."""
    print("=" * 70)
    print("TEST 3d: Metamorphic Testing with Constraints")
    print("=" * 70)

    ct = CriticalThinkingModule(seed=42)
    ct.add_constraint(Constraint("stability", "output should be stable", "Stability"))

    passed, surprise = ct.metamorphic_test(
        input_sample="original",
        expected_property="output should be consistent",
        transformation_fn=lambda x: x.upper(),
        output_validator=lambda x: True,
        check_constraints=True,
    )

    print(f"✓ Metamorphic test: passed={passed}, surprise={surprise:.3f}")
    print("✓ Metamorphic with constraints test passed\n")
    return True


# =============================================================================
# EXTENSION TEST 4: Full Extended Agent Integration
# =============================================================================

def test_extended_agent_defaults():
    """Test extended agent with default configuration (backward compatible)."""
    print("=" * 70)
    print("TEST 4a: Extended Agent with Default Configuration")
    print("=" * 70)

    agent = RavanaAgent(
        name="RAVANA-Extended-Test",
        seed=42,
        # All defaults (backward compatible)
    )

    # Run a cognitive cycle
    result = agent.process(
        text="A friend asks for help with a difficult ethical dilemma.",
        context={
            "valence_stimulus": -0.2,
            "arousal_stimulus": 0.6,
            "ethics_signal": 0.8,
        },
    )

    print(f"✓ Cycle {result.cycle} completed")
    print(f"✓ Decision: {result.decision}")
    print(f"✓ System: {result.reasoning_result.system}")
    print(f"✓ Meaning score: {result.meaning_score:.3f}")

    # Check extended status
    status = agent.status()
    print(f"✓ Perception extractors: {status['perception']['config']}")
    print(f"✓ System 2 config: {status['system2_config']['mcts_depth']} depth")
    print(f"✓ Solver: {status['critical_thinking']['solver']}")

    assert "perception" in status
    assert "system2_config" in status
    assert "critical_thinking" in status
    assert "config" in status
    
    print("✓ Extended agent defaults test passed\n")
    return True


def test_extended_agent_with_production_features():
    """Test extended agent with production feature extractors enabled."""
    print("=" * 70)
    print("TEST 4b: Extended Agent with Production Features (Mock Fallback)")
    print("=" * 70)

    agent = RavanaAgent(
        name="RAVANA-Production-Test",
        seed=42,
        use_resnet=True,   # Will use mock fallback
        use_wav2vec=True,  # Will use mock fallback
        force_mock_perception=True,
        system2_mcts_depth=8,
        system2_mcts_simulations=100,
        system2_rollout_policy=EthicalRolloutPolicy(),
        use_z3_solver=False,  # Use mock
    )

    # Run with multimodal input
    result = agent.process(
        text="Is it ethical to use AI for surveillance?",
        visual=np.random.rand(64, 64, 3),
        audio=np.random.randn(1000),
        context={
            "ethics_signal": 0.9,
            "safety_signal": 0.3,
            "valence_stimulus": -0.4,
        },
    )

    print(f"✓ Cycle {result.cycle} completed")
    print(f"✓ Perception extractors: {result.perception_output['extractor_info']}")
    print(f"✓ Reasoning depth: {result.reasoning_result.mcts_depth}")

    status = agent.status()
    print(f"✓ Config stored: use_resnet={status['config']['use_resnet']}")
    # Check configured depth in config (original setting)
    assert status["config"]["system2_mcts_depth"] == 8, f"Expected config depth 8, got {status['config']['system2_mcts_depth']}"
    # Check that system2_config exists and has the policy
    assert status["system2_config"]["rollout_policy"] == "EthicalRolloutPolicy"

    print("✓ Extended agent production features test passed\n")
    return True


def test_extended_agent_with_constraints():
    """Test extended agent with active constraints."""
    print("=" * 70)
    print("TEST 4c: Extended Agent with Symbolic Constraints")
    print("=" * 70)

    agent = RavanaAgent(
        name="RAVANA-Constrained",
        seed=42,
        use_z3_solver=False,  # Use mock
    )

    # Add constraints to critical thinking module
    agent.critical.add_constraint(Constraint(
        "harm_minimization",
        "actions must minimize harm",
        "Core ethical constraint"
    ))
    agent.critical.add_constraint(Constraint(
        "truth_seeking",
        "beliefs must be truth-seeking",
        "Epistemic constraint"
    ))

    # Run cycles
    for i in range(3):
        result = agent.process(
            text=f"Ethical scenario {i+1}: A startup offers funding for surveillance AI.",
            context={
                "ethics_signal": 0.4,
                "safety_signal": 0.3,
            },
        )
        print(f"✓ Cycle {result.cycle}: decision={result.decision[:40]}...")

    status = agent.status()
    print(f"✓ Active constraints: {status['critical_thinking']['n_constraints']}")
    print(f"✓ Solver: {status['critical_thinking']['solver']}")

    assert status["critical_thinking"]["n_constraints"] == 2
    
    print("✓ Extended agent with constraints test passed\n")
    return True


def test_backward_compatibility():
    """Test that extended agent is backward compatible with base tests."""
    print("=" * 70)
    print("TEST 4d: Backward Compatibility with Base Agent")
    print("=" * 70)

    # Create agent exactly like the base version
    agent = RavanaAgent(name="RAVANA-Compat", seed=42)

    # Run the same scenarios as base test
    scenarios = [
        {
            "text": "A friend asks you to lie about their mistake.",
            "context": {
                "valence_stimulus": -0.3,
                "arousal_stimulus": 0.6,
                "dominance_stimulus": 0.4,
                "ethics_signal": 0.3,
            },
        },
        {
            "text": "You discover a security vulnerability.",
            "context": {
                "valence_stimulus": -0.5,
                "arousal_stimulus": 0.8,
                "ethics_signal": 0.9,
            },
        },
    ]

    for scenario in scenarios:
        result = agent.process(
            text=scenario["text"],
            context=scenario["context"],
        )
        print(f"✓ Cycle {result.cycle}: {result.decision[:50]}...")
        assert result.meaning_score >= 0.0
        assert result.meaning_score <= 1.0

    # Run dream cycle
    dreams = agent.dream(n_simulations=3)
    for i, dream in enumerate(dreams):
        print(f"✓ Dream {i+1}: {dream.lesson_learned[:50]}...")

    # Check status has all base fields
    status = agent.status()
    assert "name" in status
    assert "cycle" in status
    assert "coherence_score" in status
    assert "meaning_score_avg" in status
    assert "beliefs" in status
    assert "dissonance" in status
    assert "emotion" in status

    print("✓ All base status fields present")
    print("✓ Backward compatibility test passed\n")
    return True


# =============================================================================
# TEST RUNNER
# =============================================================================

def run_all_tests():
    """Run complete extended test suite."""
    print("\n" + "=" * 70)
    print("RAVANA CORE EXTENDED — COMPREHENSIVE TEST SUITE v0.2.0")
    print("=" * 70)
    print()

    all_tests = [
        # Extension 1: Perception
        test_perception_mock_extractors,
        test_perception_with_visual_audio,
        test_resnet_extractor_stubs,
        test_wav2vec_extractor_stubs,
        test_extractor_base_class,
        # Extension 2: Reasoning
        test_system2_configurable_mcts,
        test_rollout_policies,
        test_complexity_adaptive_reasoning,
        test_dual_process_reasoner_config,
        # Extension 3: Critical Thinking
        test_mock_constraint_solver,
        test_z3_constraint_solver_stubs,
        test_critical_thinking_with_constraints,
        test_metamorphic_with_constraints,
        # Extension 4: Full Integration
        test_extended_agent_defaults,
        test_extended_agent_with_production_features,
        test_extended_agent_with_constraints,
        test_backward_compatibility,
    ]

    results = []
    for test in all_tests:
        try:
            result = test()
            results.append((test.__name__, True, None))
        except Exception as e:
            results.append((test.__name__, False, str(e)))
            print(f"\n✗ {test.__name__} FAILED: {e}\n")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"\nPassed: {passed}/{total}")

    for name, ok, error in results:
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {status}: {name}")
        if error:
            print(f"      Error: {error}")

    # Extension summary
    print("\n" + "=" * 70)
    print("EXTENSION COVERAGE")
    print("=" * 70)
    print("\n✓ Extension 1: Perception with Pluggable Extractors")
    print("  - BaseFeatureExtractor ABC")
    print("  - MockFeatureExtractor, ResNetFeatureExtractor, Wav2VecFeatureExtractor")
    print("  - Configuration flags: use_resnet, use_wav2vec, force_mock")
    print("\n✓ Extension 2: Reasoning with Configurable MCTS")
    print("  - Configurable MCTS depth and rollouts")
    print("  - Pluggable RolloutPolicy (Simple, Complex, Ethical)")
    print("  - Complexity-adaptive depth adjustment")
    print("\n✓ Extension 3: Critical Thinking with Constraint Solvers")
    print("  - ConstraintSolver ABC")
    print("  - MockConstraintSolver (always available)")
    print("  - Z3ConstraintSolver (optional Z3 dependency)")
    print("  - Constraint integration in MBFL pipeline")
    print("\n✓ Extension 4: Full Integration & Backward Compatibility")
    print("  - Extended RavanaAgent with new configuration")
    print("  - Enhanced status reporting")
    print("  - All base functionality preserved")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
