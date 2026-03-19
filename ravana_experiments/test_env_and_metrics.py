"""
Tests for classroom environment and metrics.

Sanity checks on:
- Environment initialization and stepping
- Task evaluation
- Metric computation
- Baseline agents
"""

import sys
sys.path.insert(0, '/home/.z/workspaces/con_it6dxBHxJqtbptXy/ravana_experiments')
sys.path.insert(0, '/home/.z/workspaces/con_it6dxBHxJqtbptXy/ravana_core_extended')

import numpy as np
from classroom_env import (
    ClassroomEnvironment, TaskBank, StudentProfile,
    MCQTask, OpenEndedTask, EthicalDilemmaTask,
    TaskType
)
from metrics import (
    RAVANAMetrics,
    baseline_naive_rl,
    baseline_llm_policy,
    baseline_rule_based,
)


def test_task_bank():
    """Test task bank initialization."""
    print("=" * 60)
    print("TEST: Task Bank")
    print("=" * 60)
    
    bank = TaskBank(seed=42)
    
    all_tasks = bank.get_all_task_ids()
    print(f"Total tasks: {len(all_tasks)}")
    
    mcq_tasks = bank.get_tasks_by_type(TaskType.MCQ)
    open_tasks = bank.get_tasks_by_type(TaskType.OPEN_ENDED)
    dilemma_tasks = bank.get_tasks_by_type(TaskType.ETHICAL_DILEMMA)
    
    print(f"MCQ: {len(mcq_tasks)}, Open: {len(open_tasks)}, Dilemma: {len(dilemma_tasks)}")
    
    # Get a task and verify
    task = bank.get_task("mcq_math_1")
    assert task.task_id == "mcq_math_1"
    assert task.task_type == TaskType.MCQ
    
    print("✓ Task bank test passed\n")
    return True


def test_mcq_evaluation():
    """Test MCQ task evaluation."""
    print("=" * 60)
    print("TEST: MCQ Evaluation")
    print("=" * 60)
    
    task = MCQTask(
        task_id="test_mcq",
        question="Test question?",
        options=["A", "B", "C", "D"],
        correct_option_idx=1,
    )
    
    student = StudentProfile(student_id="test", ability=0.8)
    
    # Test correct answer
    correct = task.evaluate_answer("1", student)
    print(f"Correct answer '1': {correct}")
    
    # Test incorrect answer
    wrong = task.evaluate_answer("0", student)
    print(f"Wrong answer '0': {wrong}")
    
    # Test with ability noise
    np.random.seed(42)
    results = [task.evaluate_answer("0", student) for _ in range(20)]
    print(f"Wrong answer with ability noise: {sum(results)}/20 correct (due to noise)")
    
    print("✓ MCQ evaluation test passed\n")
    return True


def test_open_ended_evaluation():
    """Test open-ended task evaluation."""
    print("=" * 60)
    print("TEST: Open-Ended Evaluation")
    print("=" * 60)
    
    task = OpenEndedTask(
        task_id="test_open",
        question="Explain X?",
        rubric_keywords={"x": 0.3, "explain": 0.3, "reason": 0.4},
    )
    
    student = StudentProfile(student_id="test", ability=0.6)
    
    # Test good answer
    good = task.evaluate_answer("I will explain x with good reasoning", student)
    print(f"Good answer: {good}")
    
    # Test poor answer
    poor = task.evaluate_answer("I don't know", student)
    print(f"Poor answer: {poor}")
    
    print("✓ Open-ended evaluation test passed\n")
    return True


def test_ethical_dilemma():
    """Test ethical dilemma evaluation."""
    print("=" * 60)
    print("TEST: Ethical Dilemma")
    print("=" * 60)
    
    task = EthicalDilemmaTask(
        task_id="test_dilemma",
        question="What do you do?",
        options=[
            ("Option A (high honesty)", {"honesty": 0.9}),
            ("Option B (low honesty)", {"honesty": 0.2}),
        ],
    )
    
    # Student who values honesty
    honest_student = StudentProfile(
        student_id="honest",
        value_preferences={"honesty": 0.9, "fairness": 0.1},
    )
    
    result_a = task.evaluate_answer("0", honest_student)
    result_b = task.evaluate_answer("1", honest_student)
    
    print(f"Honest student choosing Option A: {result_a}")
    print(f"Honest student choosing Option B: {result_b}")
    
    print("✓ Ethical dilemma test passed\n")
    return True


def test_environment_step():
    """Test environment stepping."""
    print("=" * 60)
    print("TEST: Environment Step")
    print("=" * 60)
    
    bank = TaskBank(seed=42)
    students = [
        StudentProfile(student_id="s1", ability=0.7, demographic_group="A"),
        StudentProfile(student_id="s2", ability=0.5, demographic_group="B"),
    ]
    
    env = ClassroomEnvironment(
        task_bank=bank,
        students=students,
        max_episodes=100,
        seed=42,
    )
    
    # Test step
    obs = env.reset()
    print(f"Initial observation: {obs}")
    
    agent_action = {"answer": "1", "explanation": "Test explanation"}
    agent_state = {
        "dissonance": 0.5,
        "identity_strength": 0.6,
        "mean_conf": 0.7,
        "volatility_conf": 0.1,
    }
    
    obs, rewards, done, info = env.step(agent_action, agent_state)
    
    print(f"Step result: obs keys={obs.keys()}, rewards={rewards}, done={done}")
    print(f"Info: {info}")
    print(f"Episode logs: {len(env.episode_logs)}")
    
    print("✓ Environment step test passed\n")
    return True


def test_metrics_computation():
    """Test paper metrics computation."""
    print("=" * 60)
    print("TEST: Metrics Computation")
    print("=" * 60)
    
    # Create dummy logs
    from classroom_env.environment import EpisodeLog
    from datetime import datetime
    
    logs = []
    for i in range(200):
        log = EpisodeLog(
            episode=i,
            task_id=f"task_{i % 10}",
            task_type=TaskType.MCQ if i % 3 == 0 else TaskType.OPEN_ENDED,
            student_id=f"student_{i % 4}",
            demographic_group="A" if i % 2 == 0 else "B",
            cognitive_dissonance_D=0.8 - i * 0.003,  # Decreasing
            identity_strength_index=0.3 + i * 0.003,  # Increasing
            mean_confidence=0.6,
            volatility_confidence=0.1,
            emotion_valence=0.0,
            emotion_arousal=0.5,
            response_correct=i % 3 != 0,  # 66% accuracy
            explanation_quality=0.6,
            demographic_parity_gap=None,
            reasoning_system="system2",
            mcts_depth_used=5,
            n_rollouts=50,
            falsification_triggered=False,
            dissonance_correction_triggered=False,
            dream_sabotage_triggered=False,
            meaning_scored=0.5,
            reward_primary=0.7,
            reward_social_norm=0.5,
            reward_fairness=0.6,
            hard_constraints_violated=[],
            constraints_satisfied=True,
        )
        logs.append(log)
    
    metrics = RAVANAMetrics(logs)
    
    # Test dissonance reduction
    dr = metrics.compute_dissonance_reduction(window_size=50)
    print(f"Dissonance: {dr.value:.3f} (start: ~0.8, end: ~0.2)")
    print(f"  Improvement: {dr.improvement:+.3f}")
    
    # Test identity strength
    is_result = metrics.compute_identity_strength_increase(window_size=50)
    print(f"Identity: {is_result.value:.3f} (start: ~0.3, end: ~0.85)")
    print(f"  Improvement: {is_result.improvement:+.3f}")
    
    # Test wisdom score
    ws = metrics.compute_wisdom_score()
    print(f"Wisdom: {ws.value:.3f}")
    
    # Test full summary
    summary = metrics.get_summary_table()
    print(f"\nSummary table:\n{summary[:200]}...")
    
    print("✓ Metrics computation test passed\n")
    return True


def test_baseline_agents():
    """Test baseline agent implementations."""
    print("=" * 60)
    print("TEST: Baseline Agents")
    print("=" * 60)
    
    bank = TaskBank(seed=42)
    students = [
        StudentProfile(student_id="s1", ability=0.7),
    ]
    
    env = ClassroomEnvironment(
        task_bank=bank,
        students=students,
        max_episodes=100,
        seed=42,
    )
    
    # Test naive RL
    print("Running naive RL baseline...")
    result_rl = baseline_naive_rl(env, n_episodes=100, seed=42)
    print(f"  Accuracy: {result_rl.metrics['accuracy']:.3f}")
    
    # Reset env
    env.reset()
    
    # Test LLM stub
    print("Running LLM stub baseline...")
    result_llm = baseline_llm_policy(env, n_episodes=100, seed=42)
    print(f"  Accuracy: {result_llm.metrics['accuracy']:.3f}")
    
    # Reset env
    env.reset()
    
    # Test rule-based
    print("Running rule-based baseline...")
    result_rule = baseline_rule_based(env, n_episodes=100, seed=42)
    print(f"  Accuracy: {result_rule.metrics['accuracy']:.3f}")
    
    print("✓ Baseline agents test passed\n")
    return True


def test_reward_computation():
    """Test reward computation."""
    print("=" * 60)
    print("TEST: Reward Computation")
    print("=" * 60)
    
    from classroom_env.rewards import RewardCalculator, SocialNormSignal
    
    calc = RewardCalculator(
        w_primary=1.0,
        w_social=0.2,
        w_fairness=0.3,
        social_norms=[
            SocialNormSignal("maximize_score", weight=1.0),
            SocialNormSignal("be_fair_to_group_B", target_group="B", weight=0.8),
        ],
    )
    
    reward = calc.compute(
        correctness=True,
        explanation_quality=0.7,
        demographic_group="B",
        group_performances={"A": 0.8, "B": 0.6},
    )
    
    print(f"Primary: {reward.primary:.3f}")
    print(f"Social: {reward.social_norm:.3f}")
    print(f"Fairness: {reward.fairness:.3f}")
    print(f"Composite: {reward.composite:.3f}")
    
    print("✓ Reward computation test passed\n")
    return True


def run_all_tests():
    """Run all environment and metric tests."""
    print("\n" + "=" * 60)
    print("RAVANA EXPERIMENTS — COMPREHENSIVE TEST SUITE")
    print("=" * 60 + "\n")
    
    tests = [
        test_task_bank,
        test_mcq_evaluation,
        test_open_ended_evaluation,
        test_ethical_dilemma,
        test_environment_step,
        test_reward_computation,
        test_metrics_computation,
        test_baseline_agents,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, True, None))
        except Exception as e:
            results.append((test.__name__, False, str(e)))
            print(f"\n✗ {test.__name__} FAILED: {e}\n")
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"\nPassed: {passed}/{total}")
    
    for name, ok, error in results:
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {status}: {name}")
        if error:
            print(f"      Error: {error}")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
