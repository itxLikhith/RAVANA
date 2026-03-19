"""
Experiment runner for RAVANA classroom benchmarking.

Runs RAVANA agent in three configurations:
1. RAVANA-0: All defaults, mock perception, simple rollout, mock solver
2. RAVANA-Pro: Production features, deeper MCTS, Z3 if available
3. Ablated: Specific pressures turned off

Also runs baseline comparisons and generates plots.
"""

import sys
import os
import json
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
import argparse

# Add paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ravana_core_extended')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from ravana_core import RavanaAgent
from classroom_env import ClassroomEnvironment, TaskBank, StudentProfile, TaskType
from classroom_env.tasks import MCQTask, OpenEndedTask, EthicalDilemmaTask
from metrics import RAVANAMetrics, baseline_naive_rl, baseline_llm_policy, baseline_rule_based, compare_agents


class RAVANAEnvironmentWrapper:
    """
    Wraps RAVANA agent to interface with ClassroomEnvironment.
    
    Handles:
    - Agent state extraction for logging
    - Action generation from agent processing
    - Reward signal conversion
    """
    
    def __init__(self, agent: RavanaAgent, env: ClassroomEnvironment):
        self.agent = agent
        self.env = env
        
    def run_episode(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run one episode with the RAVANA agent."""
        
        # Process through RAVANA
        result = self.agent.process(
            text=text,
            context=context,
        )
        
        # Extract action from decision
        decision = result.decision
        
        # Simple parsing: extract answer from decision string
        # Format: "reasoned: system2 → ..." or "corrected: ..."
        answer = self._extract_answer(decision)
        
        # Extract agent cognitive state for logging
        agent_state = self._extract_state(result)
        
        return {
            "answer": answer,
            "explanation": result.reasoning_result.conclusion if hasattr(result.reasoning_result, 'conclusion') else "",
            "agent_state": agent_state,
        }
    
    def _extract_answer(self, decision: str) -> str:
        """Extract answer choice from decision string."""
        # NEW: Handle "The agent chose to '...'" format from System2Reasoner
        import re
        match = re.search(r"The agent chose to '([^']+)'", decision)
        if match:
            return match.group(1)
            
        # For MCQ: look for option number at the start of reasoned conclusion
        # format: "REASONED: system2 → 1 (MCTS value: ...)"
        numbers = re.findall(r' → (\d+)', decision)
        if numbers:
            return numbers[0]
            
        # Fallback for constraint or correction strings
        if "CONSTRAINT" in decision or "corrected" in decision.lower():
            # Try to find a number if it's an MCQ task
            nums = re.findall(r'\d+', decision)
            if nums: return nums[0]
        
        # For open-ended: return the decision as-is (cleaned)
        return decision.replace("REASONED: ", "").replace("CORRECTION: ", "").strip()
    
    def _extract_state(self, result) -> Dict[str, Any]:
        """Extract cognitive state from agent result."""
        return {
            "dissonance": result.dissonance_score,
            "identity_strength": self.agent.coherence_score,
            "mean_conf": result.perception_output.get('mean_conf', 0.5),
            "volatility_conf": result.perception_output.get('volatility_conf', 0.0),
            "emotion": result.emotion_state,
            "reasoning_system": result.reasoning_result.system if hasattr(result.reasoning_result, 'system') else "system1",
            "mcts_depth": getattr(result.reasoning_result, 'mcts_depth', 5),
            "n_rollouts": getattr(result.reasoning_result, 'n_rollouts', 50),
            "falsification_triggered": not result.reasoning_result.falsification_passed if hasattr(result.reasoning_result, 'falsification_passed') else False,
            "dissonance_correction_triggered": "corrected" in result.decision,
            "dream_sabotage_triggered": False,  # Tracked separately
            "meaning_score": result.meaning_score,
            "constraints_violated": [],
            "constraints_satisfied": True,
        }


def create_ravana_0_config(seed: int = 42) -> Dict[str, Any]:
    """RAVANA-0: CPU-friendly baseline configuration."""
    return {
        "name": "RAVANA-0",
        "seed": seed,
        # Perception: mock (fastest)
        "use_resnet": False,
        "use_wav2vec": False,
        "force_mock_perception": True,
        # System 2: minimal simulations for CPU
        "system2_mcts_depth": 3,
        "system2_mcts_simulations": 15,
        # Critical thinking: mock solver
        "use_z3_solver": False,
    }


def create_ravana_pro_config(seed: int = 42) -> Dict[str, Any]:
    """RAVANA-Pro: CPU-optimized production configuration."""
    return {
        "name": "RAVANA-Pro",
        "seed": seed,
        # Perception: neural extractors on CPU
        "use_resnet": True,
        "use_wav2vec": True,
        "force_mock_perception": False,
        "perception_device": "cpu",
        # System 2: reduced simulations for CPU responsiveness
        "system2_mcts_depth": 5,
        "system2_mcts_simulations": 30,
        # Critical thinking: Z3 solver
        "use_z3_solver": True,
    }


def create_ablated_config(
    seed: int = 42,
    no_dissonance_reappraisal: bool = False,
    no_constraint_solver: bool = False,
    no_dream_sabotage: bool = False,
) -> Dict[str, Any]:
    """
    Ablated configuration with specific pressures turned off.
    
    Currently implements ablation by:
    - Reducing MCTS depth (simpler reasoning)
    - Disabling Z3 solver if no_constraint_solver
    - Setting dream rate to 0 if no_dream_sabotage
    
    Note: Full ablation of dissonance reappraisal would require 
    modifications to the core agent class.
    """
    config = create_ravana_0_config(seed)
    config["name"] = "RAVANA-Ablated"
    
    if no_dissonance_reappraisal:
        # Approximate by reducing system 2 depth (less reflective reasoning)
        config["system2_mcts_depth"] = 2
        config["system2_mcts_simulations"] = 10
    
    if no_constraint_solver:
        config["use_z3_solver"] = False
        # Force mock solver explicitly
    
    if no_dream_sabotage:
        # Set dream sabotage rate to 0
        config["dream_counterfactual_rate"] = 0.0
    
    return config


def run_experiment(
    config: Dict[str, Any],
    env: ClassroomEnvironment,
    n_episodes: int = 1000,
    log_interval: int = 100,
) -> Dict[str, Any]:
    """
    Run single experiment with given configuration.
    
    Returns:
        Dict with logs, metrics, and summary
    """
    print(f"\n{'='*70}")
    print(f"Running: {config['name']}")
    print(f"Episodes: {n_episodes}")
    print(f"{'='*70}\n")
    
    # Create agent
    agent = RavanaAgent(**config)
    wrapper = RAVANAEnvironmentWrapper(agent, env)
    
    # Reset environment
    env.reset()
    
    # Training loop
    for episode in range(n_episodes):
        # Sample a task context
        task_id = env.rng.choice(list(env.train_tasks))
        task = env.task_bank.get_task(task_id)
        student_id = env.rng.choice(env.student_ids)
        student = env.students[student_id]
        
        # Build context for agent
        context = {
            "valence_stimulus": 0.0,
            "arousal_stimulus": 0.5 + task.difficulty * 0.3,
            "dominance_stimulus": 0.5,
            "action_value": task.difficulty,
            "action_description": f"Answer {task.task_type.name} question",
            "pleasure_signal": 0.5,
            "safety_signal": 0.6,
            "ethics_signal": 0.7 if task.task_type == TaskType.ETHICAL_DILEMMA else 0.5,
        }
        
        # Run agent
        agent_action = wrapper.run_episode(task.question, context)
        
        # Execute in environment
        obs, rewards, done, info = env.step(agent_action, agent_action.get("agent_state"))
        
        # Periodic dream sabotage
        if episode % 50 == 0 and episode > 0:
            dreams = agent.dream(n_simulations=3)
            if dreams:
                # Mark dream sabotage in last log
                if env.episode_logs:
                    # Create new log entry for dream
                    pass
        
        # Logging
        if (episode + 1) % log_interval == 0:
            recent_logs = env.episode_logs[-log_interval:]
            accuracy = sum(1 for log in recent_logs if log.response_correct) / len(recent_logs)
            avg_dissonance = np.mean([log.cognitive_dissonance_D for log in recent_logs])
            print(f"Episode {episode + 1}/{n_episodes}: Acc={accuracy:.3f}, D={avg_dissonance:.3f}")
    
    # Compute metrics
    metrics = RAVANAMetrics(env.episode_logs)
    results = metrics.compute_all()
    
    # Summary
    summary = metrics.get_summary_table()
    print(f"\n{summary}\n")
    
    return {
        "config_name": config["name"],
        "logs": env.episode_logs,
        "metrics": results,
        "summary": summary,
        "agent_status": agent.status(),
    }


def run_all_experiments(
    n_episodes: int = 1000,
    output_dir: str = "results",
    seed: int = 42,
):
    """Run complete experiment suite with all configurations."""
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/logs", exist_ok=True)
    os.makedirs(f"{output_dir}/plots", exist_ok=True)
    
    # Create environment
    task_bank = TaskBank(seed=seed)
    
    # Create diverse student population
    students = [
        StudentProfile(
            student_id="student_A_high",
            ability=0.8,
            demographic_group="A",
            value_preferences={"honesty": 0.9, "achievement": 0.8, "fairness": 0.6},
        ),
        StudentProfile(
            student_id="student_A_low",
            ability=0.4,
            demographic_group="A",
            value_preferences={"honesty": 0.7, "helpfulness": 0.8, "fairness": 0.7},
        ),
        StudentProfile(
            student_id="student_B_high",
            ability=0.8,
            demographic_group="B",
            value_preferences={"honesty": 0.8, "achievement": 0.7, "autonomy": 0.8},
        ),
        StudentProfile(
            student_id="student_B_low",
            ability=0.4,
            demographic_group="B",
            value_preferences={"helpfulness": 0.9, "fairness": 0.8, "autonomy": 0.6},
        ),
        StudentProfile(
            student_id="student_A_mid",
            ability=0.6,
            demographic_group="A",
            value_preferences={"honesty": 0.8, "fairness": 0.7, "achievement": 0.6},
        ),
        StudentProfile(
            student_id="student_B_mid",
            ability=0.6,
            demographic_group="B",
            value_preferences={"autonomy": 0.8, "fairness": 0.8, "honesty": 0.7},
        ),
    ]
    
    env = ClassroomEnvironment(
        task_bank=task_bank,
        students=students,
        max_episodes=n_episodes,
        seed=seed,
        held_out_ratio=0.2,
        fairness_weight=0.3,
    )
    
    all_results = {}
    
    # 1. RAVANA-0 (baseline)
    config_0 = create_ravana_0_config(seed)
    result_0 = run_experiment(config_0, env, n_episodes)
    all_results["RAVANA-0"] = result_0
    save_results(result_0, f"{output_dir}/logs/ravana_0.csv")
    
    # Reset environment for fair comparison
    env.reset()
    
    # 2. RAVANA-Pro
    config_pro = create_ravana_pro_config(seed)
    result_pro = run_experiment(config_pro, env, n_episodes)
    all_results["RAVANA-Pro"] = result_pro
    save_results(result_pro, f"{output_dir}/logs/ravana_pro.csv")
    
    # Reset environment
    env.reset()
    
    # 3. Ablated (no dissonance reappraisal)
    config_ablated = create_ablated_config(seed, no_dissonance_reappraisal=True)
    result_ablated = run_experiment(config_ablated, env, n_episodes)
    all_results["RAVANA-Ablated"] = result_ablated
    save_results(result_ablated, f"{output_dir}/logs/ravana_ablated.csv")
    
    # Run baselines
    print(f"\n{'='*70}")
    print("Running Baselines")
    print(f"{'='*70}\n")
    
    baseline_results = []
    
    # Naive RL
    env.reset()
    baseline_rl = baseline_naive_rl(env, n_episodes=n_episodes, seed=seed)
    baseline_results.append(baseline_rl)
    save_baseline_logs(baseline_rl.logs, f"{output_dir}/logs/baseline_rl.csv")
    
    # LLM Stub
    env.reset()
    baseline_llm = baseline_llm_policy(env, n_episodes=n_episodes, seed=seed)
    baseline_results.append(baseline_llm)
    save_baseline_logs(baseline_llm.logs, f"{output_dir}/logs/baseline_llm.csv")
    
    # Rule-based
    env.reset()
    baseline_rule = baseline_rule_based(env, n_episodes=n_episodes, seed=seed)
    baseline_results.append(baseline_rule)
    save_baseline_logs(baseline_rule.logs, f"{output_dir}/logs/baseline_rule.csv")
    
    # Generate comparison
    comparison = compare_agents(result_pro["logs"], baseline_results)
    print(comparison)
    
    # Save comparison
    with open(f"{output_dir}/comparison.txt", "w", encoding="utf-8") as f:
        f.write(comparison)
    
    # Generate plots
    generate_plots(all_results, baseline_results, output_dir)
    
    # Final summary
    final_summary = {
        "timestamp": datetime.now().isoformat(),
        "n_episodes": n_episodes,
        "configurations": list(all_results.keys()),
        "baselines": [b.baseline_type.value for b in baseline_results],
        "metrics_summary": {
            name: {k: v.value for k, v in result["metrics"].items()}
            for name, result in all_results.items()
        },
    }
    
    with open(f"{output_dir}/summary.json", "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2)
    
    print(f"\n{'='*70}")
    print("EXPERIMENTS COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {output_dir}/")
    print(f"  - Logs: {output_dir}/logs/")
    print(f"  - Plots: {output_dir}/plots/")
    print(f"  - Summary: {output_dir}/summary.json")
    print(f"{'='*70}\n")
    
    return all_results, baseline_results


def save_results(result: Dict[str, Any], filepath: str):
    """Save experiment results to CSV."""
    try:
        import pandas as pd
        
        data = []
        for log in result["logs"]:
            data.append({
                "episode": log.episode,
                "task_id": log.task_id,
                "task_type": log.task_type.name,
                "student_id": log.student_id,
                "demographic_group": log.demographic_group,
                "correct": log.response_correct,
                "dissonance": log.cognitive_dissonance_D,
                "identity_strength": log.identity_strength_index,
                "explanation_quality": log.explanation_quality,
                "reasoning_system": log.reasoning_system,
                "mcts_depth": log.mcts_depth_used,
                "meaning_score": log.meaning_scored,
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        print(f"Saved: {filepath}")
    except ImportError:
        # Fallback: simple CSV writing
        import csv
        with open(filepath, 'w', newline='') as f:
            if result["logs"]:
                writer = csv.writer(f)
                writer.writerow(["episode", "task_id", "task_type", "student_id", 
                               "demographic_group", "correct", "dissonance"])
                for log in result["logs"][:1000]:  # Limit rows
                    writer.writerow([log.episode, log.task_id, log.task_type.name,
                                   log.student_id, log.demographic_group, 
                                   log.response_correct, log.cognitive_dissonance_D])
        print(f"Saved (CSV fallback): {filepath}")


def save_baseline_logs(logs: List[Any], filepath: str):
    """Save baseline logs to CSV."""
    try:
        import pandas as pd
        
        data = []
        for log in logs:
            data.append({
                "episode": log.episode,
                "task_id": log.task_id,
                "task_type": log.task_type.name,
                "student_id": log.student_id,
                "demographic_group": log.demographic_group,
                "correct": log.response_correct,
                "dissonance": log.cognitive_dissonance_D,
                "identity_strength": log.identity_strength_index,
                "reasoning_system": log.reasoning_system,
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        print(f"Saved: {filepath}")
    except ImportError:
        # Fallback: simple CSV writing
        import csv
        with open(filepath, 'w', newline='') as f:
            if logs:
                writer = csv.writer(f)
                writer.writerow(["episode", "task_id", "task_type", "student_id", 
                               "demographic_group", "correct", "dissonance"])
                for log in logs[:1000]:  # Limit rows
                    writer.writerow([log.episode, log.task_id, log.task_type.name,
                                   log.student_id, log.demographic_group, 
                                   log.response_correct, log.cognitive_dissonance_D])
        print(f"Saved (CSV fallback): {filepath}")


def generate_plots(
    ravana_results: Dict[str, Any],
    baseline_results: List[Any],
    output_dir: str,
):
    """Generate metric trajectory plots."""
    try:
        import matplotlib.pyplot as plt
        
        # Plot 1: Dissonance reduction
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Dissonance over time
        ax = axes[0, 0]
        for name, result in ravana_results.items():
            dissonance = [log.cognitive_dissonance_D for log in result["logs"]]
            window = 50
            smoothed = [np.mean(dissonance[i:i+window]) for i in range(0, len(dissonance), window)]
            ax.plot(smoothed, label=name)
        ax.set_xlabel("Episode (x50)")
        ax.set_ylabel("Dissonance D")
        ax.set_title("Dissonance Reduction")
        ax.legend()
        ax.axhline(y=0.2, color='r', linestyle='--', label='Target')
        
        # Identity strength
        ax = axes[0, 1]
        for name, result in ravana_results.items():
            identity = [log.identity_strength_index for log in result["logs"]]
            window = 50
            smoothed = [np.mean(identity[i:i+window]) for i in range(0, len(identity), window)]
            ax.plot(smoothed, label=name)
        ax.set_xlabel("Episode (x50)")
        ax.set_ylabel("Identity Strength")
        ax.set_title("Identity Strength Increase")
        ax.legend()
        ax.axhline(y=0.85, color='r', linestyle='--', label='Target')
        
        # Accuracy over time
        ax = axes[0, 2]
        for name, result in ravana_results.items():
            correct = [1 if log.response_correct else 0 for log in result["logs"]]
            window = 100
            smoothed = [np.mean(correct[i:i+window]) for i in range(0, len(correct), window)]
            ax.plot(smoothed, label=name)
        ax.set_xlabel("Episode (x100)")
        ax.set_ylabel("Accuracy")
        ax.set_title("Learning Curve")
        ax.legend()
        
        # Demographic parity gap
        ax = axes[1, 0]
        for name, result in ravana_results.items():
            gaps = [log.demographic_parity_gap for log in result["logs"] if log.demographic_parity_gap is not None]
            if gaps:
                window = 50
                smoothed = [np.mean(gaps[i:i+window]) for i in range(0, len(gaps), window)]
                ax.plot(smoothed, label=name)
        ax.set_xlabel("Episode (x50)")
        ax.set_ylabel("Parity Gap (%)")
        ax.set_title("Demographic Parity Gap Reduction")
        ax.legend()
        ax.axhline(y=5, color='r', linestyle='--', label='Target')
        
        # Comparison bar chart
        ax = axes[1, 1]
        
        all_results = list(ravana_results.items())
        for baseline in baseline_results:
            # Compute metrics for baseline
            from metrics import RAVANAMetrics
            metrics = RAVANAMetrics(baseline.logs)
            acc = sum(1 for log in baseline.logs if log.response_correct) / len(baseline.logs)
            all_results.append((baseline.baseline_type.value, {"accuracy": acc}))
        
        names = [name for name, _ in all_results]
        accs = [result.get("accuracy", 0.5) if isinstance(result, dict) else 
                sum(1 for log in result["logs"] if log.response_correct) / len(result["logs"])
                for name, result in all_results]
        
        ax.bar(names, accs)
        ax.set_ylabel("Accuracy")
        ax.set_title("Final Accuracy Comparison")
        ax.set_ylim([0, 1])
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Meaning scores
        ax = axes[1, 2]
        for name, result in ravana_results.items():
            meaning = [log.meaning_scored for log in result["logs"]]
            window = 50
            smoothed = [np.mean(meaning[i:i+window]) for i in range(0, len(meaning), window)]
            ax.plot(smoothed, label=name)
        ax.set_xlabel("Episode (x50)")
        ax.set_ylabel("Meaning Score")
        ax.set_title("Meaning Orientation")
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/plots/metrics_trajectories.png", dpi=150)
        plt.close()
        
        print(f"Saved: {output_dir}/plots/metrics_trajectories.png")
        
    except ImportError:
        print("matplotlib not available, skipping plots")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run RAVANA classroom experiments")
    parser.add_argument("--episodes", type=int, default=5000, help="Number of episodes")
    parser.add_argument("--output", type=str, default="results_cpu_optimized", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--quick", action="store_true", help="Quick test run (100 episodes)")
    
    args = parser.parse_args()
    
    n_episodes = 100 if args.quick else args.episodes
    
    run_all_experiments(
        n_episodes=n_episodes,
        output_dir=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
