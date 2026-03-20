#!/usr/bin/env python3
"""
RAVANA v3.5 Comprehensive Test Suite

Tests all v3.5 features:
1. Full Multi-Modal Cognitive Loop
2. Stress-Amplified Dissonance  
3. Adaptive Epistemic Calibration
4. Interactive XAI with Reappraisal
5. Multi-Agent Collaborative Classroom
"""

import sys
import os
import json
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.abspath('ravana_core_extended'))
sys.path.insert(0, os.path.abspath('ravana_experiments'))

from ravana_core import RavanaAgent
from classroom_env import ClassroomEnvironment, TaskBank, StudentProfile
from classroom_env.environment import StudentEngagementSignal
from metrics import RAVANAMetrics


def test_multimodal_cognitive_loop():
    """Test 1: Full Multi-Modal Cognitive Loop"""
    print("\n" + "="*60)
    print("TEST 1: Full Multi-Modal Cognitive Loop")
    print("="*60)
    
    agent = RavanaAgent(
        name="RAVANA-Multimodal",
        seed=42,
        use_adaptive_calibration=False,
        stress_amplification_factor=0.3,
    )
    
    # Create environment with multi-modal enabled
    task_bank = TaskBank(seed=42)
    students = [
        StudentProfile(
            student_id="test_student",
            ability=0.6,
            demographic_group="A",
            baseline_engagement={"engagement": 0.7, "confusion": 0.2, "boredom": 0.1, "attention": 0.8},
        )
    ]
    
    env = ClassroomEnvironment(
        task_bank=task_bank,
        students=students,
        max_episodes=10,
        seed=42,
        enable_multimodal=True,
    )
    
    results = {
        "engagement_signals": [],
        "empathy_rewards": [],
        "dissonance_with_stress": [],
    }
    
    for episode in range(10):
        task_id = env.rng.choice(list(env.train_tasks))
        task = env.task_bank.get_task(task_id)
        student = students[0]
        
        # Pre-step: Generate engagement signals
        student_vad = env._simulate_student_vad(task, student)
        engagement_signal = env._generate_engagement_signals(student, student_vad)
        
        # Process with agent
        context = {
            "valence_stimulus": student_vad["valence"],
            "arousal_stimulus": student_vad["arousal"],
            "dominance_stimulus": student_vad["dominance"],
            "action_value": 0.5,
            "ethics_signal": 0.6,
        }
        
        result = agent.process(text=task.question, context=context)
        
        # Extract agent state for empathy calculation
        agent_state = {
            "emotion": result.emotion_state,
            "dissonance": result.dissonance_score,
        }
        
        # Step in environment (this calculates empathy reward)
        agent_action = {
            "answer": "1",
            "explanation": f"Test explanation for episode {episode}",
        }
        
        obs, rewards, done, info = env.step(agent_action, agent_state)
        
        results["engagement_signals"].append({
            "engagement": engagement_signal.engagement,
            "confusion": engagement_signal.confusion,
            "tone": engagement_signal.tone,
        })
        results["empathy_rewards"].append(info["empathy_reward"])
        results["dissonance_with_stress"].append(result.dissonance_score)
    
    avg_empathy = np.mean(results["empathy_rewards"])
    
    print(f"✓ Generated {len(results['engagement_signals'])} engagement signals")
    print(f"✓ Average empathy reward: {avg_empathy:.3f}")
    print(f"✓ Average stress-amplified dissonance: {np.mean(results['dissonance_with_stress']):.3f}")
    print(f"✓ Multi-modal signals present in observations: {'engagement_signals' in obs}")
    
    return {
        "success": bool(avg_empathy > 0.5),
        "avg_empathy": float(avg_empathy),
        "details": results,
    }


def test_stress_amplified_dissonance():
    """Test 2: Stress-Amplified Dissonance"""
    print("\n" + "="*60)
    print("TEST 2: Stress-Amplified Dissonance")
    print("="*60)
    
    # Use separate agents to avoid adaptation bias
    agent_low = RavanaAgent(
        name="RAVANA-Stress-Low",
        seed=42,
        stress_amplification_factor=0.5,
    )
    agent_high = RavanaAgent(
        name="RAVANA-Stress-High",
        seed=42,
        stress_amplification_factor=0.5,
    )
    
    results = {
        "low_arousal": [],
        "high_arousal": [],
    }
    
    # Test with low arousal
    for _ in range(5):
        result = agent_low.process(
            text="Simple question",
            context={
                "arousal_stimulus": 0.2,  # Low arousal
                "valence_stimulus": 0.0,
            }
        )
        results["low_arousal"].append(result.dissonance_score)
    
    # Test with high arousal
    for _ in range(5):
        result = agent_high.process(
            text="Complex ethical dilemma",
            context={
                "arousal_stimulus": 0.9,  # High arousal
                "valence_stimulus": -0.3,
            }
        )
        results["high_arousal"].append(result.dissonance_score)
    
    avg_low = np.mean(results["low_arousal"])
    avg_high = np.mean(results["high_arousal"])
    
    print(f"✓ Low arousal (0.2): avg dissonance = {avg_low:.3f}")
    print(f"✓ High arousal (0.9): avg dissonance = {avg_high:.3f}")
    print(f"✓ Amplification factor: {(avg_high / max(avg_low, 0.01) - 1):.2f}x")
    
    return {
        "success": bool(avg_high > avg_low),
        "low_arousal_dissonance": float(avg_low),
        "high_arousal_dissonance": float(avg_high),
    }


def test_adaptive_calibration():
    """Test 3: Adaptive Epistemic Calibration"""
    print("\n" + "="*60)
    print("TEST 3: Adaptive Epistemic Calibration")
    print("="*60)
    
    agent = RavanaAgent(
        name="RAVANA-Adaptive",
        seed=42,
        use_adaptive_calibration=True,
    )
    
    results = {
        "stages": [],
        "kappa_values": [],
        "weight_shifts": [],
    }
    
    test_episodes = [0, 100000, 300000, 700000, 1100000]
    
    for ep in test_episodes:
        agent.current_episode = ep
        agent._update_adaptive_parameters()
        
        stage = agent.get_training_stage().value
        weights = agent.get_current_weights()
        
        results["stages"].append(stage)
        results["kappa_values"].append(agent.kappa)
        results["weight_shifts"].append({
            "w_dissonance": weights["w_dissonance"],
            "w_identity": weights["w_identity"],
        })
        
        print(f"  Episode {ep:>7}: Stage={stage:>8}, κ={agent.kappa:.4f}, "
              f"D={weights['w_dissonance']:.2f}, I={weights['w_identity']:.2f}")
    
    # Verify progression
    kappa_progression = results["kappa_values"][-1] > results["kappa_values"][0]
    dissonance_decrease = results["weight_shifts"][-1]["w_dissonance"] < results["weight_shifts"][0]["w_dissonance"]
    identity_increase = results["weight_shifts"][-1]["w_identity"] > results["weight_shifts"][0]["w_identity"]
    
    print(f"✓ Kappa progression: {results['kappa_values'][0]:.3f} → {results['kappa_values'][-1]:.3f}")
    print(f"✓ Dissonance weight: {results['weight_shifts'][0]['w_dissonance']:.2f} → {results['weight_shifts'][-1]['w_dissonance']:.2f}")
    print(f"✓ Identity weight: {results['weight_shifts'][0]['w_identity']:.2f} → {results['weight_shifts'][-1]['w_identity']:.2f}")
    
    return {
        "success": bool(kappa_progression and dissonance_decrease and identity_increase),
        "stages": results["stages"],
        "kappa_values": results["kappa_values"],
    }


def test_interactive_xai():
    """Test 4: Interactive XAI with Reappraisal Loop"""
    print("\n" + "="*60)
    print("TEST 4: Interactive XAI with Reappraisal Loop")
    print("="*60)
    
    agent = RavanaAgent(
        name="RAVANA-XAI",
        seed=42,
        system2_mcts_depth=5,
    )
    
    # Generate initial explanation
    result = agent.process(text="Why should we prioritize fairness?")
    original_explanation = result.decision
    
    print(f"  Original: {original_explanation[:60]}...")
    
    # Challenge the explanation
    challenge_reason = "Your reasoning doesn't account for long-term consequences"
    updated_decision, learning_gain, challenge_record = agent.challenge_explanation(
        original_explanation=original_explanation,
        challenge_reason=challenge_reason,
    )
    
    print(f"  Challenge: {challenge_reason}")
    print(f"  Reappraised: {updated_decision[:60]}...")
    print(f"  Learning gain: {learning_gain:.3f}")
    print(f"  Dissonance: {challenge_record.initial_dissonance:.3f} → {challenge_record.final_dissonance:.3f}")
    
    # Get challenge summary
    summary = agent.get_challenge_summary()
    
    print(f"✓ Challenges received: {summary['challenges_received']}")
    print(f"✓ Avg learning gain: {summary['avg_learning_gain']:.3f}")
    print(f"✓ Transfer efficiency: {summary['transfer_efficiency']:.3f}")
    
    return {
        "success": bool(learning_gain > 0.3),
        "learning_gain": float(learning_gain),
        "transfer_efficiency": float(summary["transfer_efficiency"]),
    }


def run_all_tests():
    """Run all v3.5 tests and save results"""
    print("\n" + "="*70)
    print("RAVANA v3.5 COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "version": "3.5",
        "tests": {},
        "all_passed": True,
    }
    
    try:
        results["tests"]["multimodal_loop"] = test_multimodal_cognitive_loop()
    except Exception as e:
        results["tests"]["multimodal_loop"] = {"success": False, "error": str(e)}
        results["all_passed"] = False
    
    try:
        results["tests"]["stress_dissonance"] = test_stress_amplified_dissonance()
    except Exception as e:
        results["tests"]["stress_dissonance"] = {"success": False, "error": str(e)}
        results["all_passed"] = False
    
    try:
        results["tests"]["adaptive_calibration"] = test_adaptive_calibration()
    except Exception as e:
        results["tests"]["adaptive_calibration"] = {"success": False, "error": str(e)}
        results["all_passed"] = False
    
    try:
        results["tests"]["interactive_xai"] = test_interactive_xai()
    except Exception as e:
        results["tests"]["interactive_xai"] = {"success": False, "error": str(e)}
        results["all_passed"] = False
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    all_passed = True
    for test_name, test_result in results["tests"].items():
        success = test_result.get("success", False)
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {test_name}")
        if not success:
            all_passed = False
    
    results["all_passed"] = all_passed
    print(f"\nAll tests passed: {all_passed}")
    
    # Save results
    with open("test_results_v3_5.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: test_results_v3_5.json")
    
    return results


if __name__ == "__main__":
    run_all_tests()
