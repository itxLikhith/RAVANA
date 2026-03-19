"""
Paper-style metrics for RAVANA evaluation.

Implements metrics from the RAVANA paper:
- Dissonance reduction (target: 0.8 → 0.2)
- Identity strength increase (target: 0.3 → 0.85)
- Generalization accuracy (target: ~0.9)
- Transfer efficiency (target: ~0.8)
- Demographic parity gap reduction (target: 20 → 5 points)
- Composite wisdom score
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class MetricResult:
    """Result of metric computation."""
    name: str
    value: float
    target: float
    achieved: bool
    trajectory: List[float]  # Per-episode or per-window values
    improvement: float  # Delta from start to end


class RAVANAMetrics:
    """Computes all paper-style metrics from episode logs."""
    
    # Target values from paper
    TARGETS = {
        "dissonance_reduction": 0.2,  # From ~0.8 down to ~0.2
        "identity_strength": 0.85,  # From ~0.3 up to ~0.85
        "generalization_accuracy": 0.9,  # On held-out tasks
        "transfer_efficiency": 0.8,  # ~0.8 efficiency
        "demographic_parity_gap": 5.0,  # From ~20 down to ~5 points
        "wisdom_score": 0.75,  # Composite wisdom threshold
    }
    
    def __init__(self, logs: List[Any]):
        """
        Initialize with episode logs.
        
        Args:
            logs: List of EpisodeLog objects from ClassroomEnvironment
        """
        self.logs = logs
        self.n_episodes = len(logs)
    
    def compute_all(self) -> Dict[str, MetricResult]:
        """Compute all paper-style metrics."""
        return {
            "dissonance_reduction": self.compute_dissonance_reduction(),
            "identity_strength": self.compute_identity_strength_increase(),
            "generalization_accuracy": self.compute_generalization_accuracy(),
            "transfer_efficiency": self.compute_transfer_efficiency(),
            "demographic_parity_gap": self.compute_demographic_parity_gap_reduction(),
            "wisdom_score": self.compute_wisdom_score(),
        }
    
    def compute_dissonance_reduction(self, window_size: int = 100) -> MetricResult:
        """
        Compute cognitive dissonance reduction over training.
        
        Target: Start at ~0.8, reduce to ~0.2
        """
        if not self.logs:
            return MetricResult(
                name="dissonance_reduction",
                value=0.0,
                target=self.TARGETS["dissonance_reduction"],
                achieved=False,
                trajectory=[],
                improvement=0.0,
            )
        
        dissonance_values = [log.cognitive_dissonance_D for log in self.logs]
        
        # Compute rolling average
        trajectory = []
        for i in range(0, len(dissonance_values), window_size):
            window = dissonance_values[i:i + window_size]
            if window:
                trajectory.append(np.mean(window))
        
        # Start and end values
        start_d = np.mean(dissonance_values[:min(window_size, len(dissonance_values))])
        end_d = np.mean(dissonance_values[-min(window_size, len(dissonance_values)):])
        
        # Reduction is good (lower is better)
        final_value = end_d
        improvement = start_d - end_d  # Positive = reduction
        
        achieved = final_value <= self.TARGETS["dissonance_reduction"]
        
        return MetricResult(
            name="dissonance_reduction",
            value=float(final_value),
            target=self.TARGETS["dissonance_reduction"],
            achieved=achieved,
            trajectory=[float(v) for v in trajectory],
            improvement=float(improvement),
        )
    
    def compute_identity_strength_increase(self, window_size: int = 100) -> MetricResult:
        """
        Compute identity strength increase over training.
        
        Target: Start at ~0.3, increase to ~0.85
        """
        if not self.logs:
            return MetricResult(
                name="identity_strength",
                value=0.0,
                target=self.TARGETS["identity_strength"],
                achieved=False,
                trajectory=[],
                improvement=0.0,
            )
        
        identity_values = [log.identity_strength_index for log in self.logs]
        
        trajectory = []
        for i in range(0, len(identity_values), window_size):
            window = identity_values[i:i + window_size]
            if window:
                trajectory.append(np.mean(window))
        
        start_i = np.mean(identity_values[:min(window_size, len(identity_values))])
        end_i = np.mean(identity_values[-min(window_size, len(identity_values)):])
        
        final_value = end_i
        improvement = end_i - start_i
        
        achieved = final_value >= self.TARGETS["identity_strength"]
        
        return MetricResult(
            name="identity_strength",
            value=float(final_value),
            target=self.TARGETS["identity_strength"],
            achieved=achieved,
            trajectory=[float(v) for v in trajectory],
            improvement=float(improvement),
        )
    
    def compute_generalization_accuracy(self) -> MetricResult:
        """
        Compute generalization accuracy on held-out tasks.
        
        Target: ~0.9 accuracy on unseen tasks
        """
        # Identify held-out tasks (tasks seen less frequently)
        task_counts = {}
        for log in self.logs:
            task_counts[log.task_id] = task_counts.get(log.task_id, 0) + 1
        
        # Held-out tasks are those with fewest occurrences
        sorted_tasks = sorted(task_counts.items(), key=lambda x: x[1])
        n_held = max(1, len(sorted_tasks) // 5)  # Bottom 20% are "held-out"
        held_out_tasks = set(t[0] for t in sorted_tasks[:n_held])
        
        # Compute accuracy on held-out tasks
        held_out_logs = [log for log in self.logs if log.task_id in held_out_tasks]
        
        if not held_out_logs:
            accuracy = 0.0
        else:
            correct = sum(1 for log in held_out_logs if log.response_correct)
            accuracy = correct / len(held_out_logs)
        
        # Trajectory over time
        trajectory = []
        window_size = max(1, len(held_out_logs) // 10)
        for i in range(0, len(held_out_logs), window_size):
            window = held_out_logs[i:i + window_size]
            if window:
                acc = sum(1 for log in window if log.response_correct) / len(window)
                trajectory.append(acc)
        
        achieved = accuracy >= self.TARGETS["generalization_accuracy"]
        
        return MetricResult(
            name="generalization_accuracy",
            value=float(accuracy),
            target=self.TARGETS["generalization_accuracy"],
            achieved=achieved,
            trajectory=[float(v) for v in trajectory],
            improvement=float(trajectory[-1] - trajectory[0]) if len(trajectory) >= 2 else 0.0,
        )
    
    def compute_transfer_efficiency(self, n_burn_in: int = 100) -> MetricResult:
        """
        Compute transfer efficiency metric.
        
        Measures how quickly knowledge transfers to new task types.
        Target: ~0.8 efficiency (faster than baseline)
        """
        if len(self.logs) < n_burn_in * 2:
            return MetricResult(
                name="transfer_efficiency",
                value=0.0,
                target=self.TARGETS["transfer_efficiency"],
                achieved=False,
                trajectory=[],
                improvement=0.0,
            )
        
        # Identify task type switches
        task_types = [log.task_type.name for log in self.logs]
        
        # Find first occurrence of each new task type after burn-in
        burn_in_logs = self.logs[n_burn_in:]
        seen_types = set(task_types[:n_burn_in])
        
        transfer_episodes = []
        for i, log in enumerate(burn_in_logs):
            if log.task_type.name not in seen_types:
                # This is a new task type - measure performance on it
                # Look at performance over next 20 episodes
                future_logs = burn_in_logs[i:i + 20]
                if future_logs:
                    accuracy = sum(1 for l in future_logs if l.response_correct) / len(future_logs)
                    transfer_episodes.append(accuracy)
                seen_types.add(log.task_type.name)
        
        if not transfer_episodes:
            efficiency = 0.5  # Default
        else:
            efficiency = np.mean(transfer_episodes)
        
        achieved = efficiency >= self.TARGETS["transfer_efficiency"]
        
        return MetricResult(
            name="transfer_efficiency",
            value=float(efficiency),
            target=self.TARGETS["transfer_efficiency"],
            achieved=achieved,
            trajectory=[float(v) for v in transfer_episodes],
            improvement=float(efficiency - 0.5),  # Compared to random baseline
        )
    
    def compute_demographic_parity_gap_reduction(self, window_size: int = 100) -> MetricResult:
        """
        Compute demographic parity gap reduction.
        
        Target: Reduce gap from ~20 points to ~5 points
        """
        if not self.logs:
            return MetricResult(
                name="demographic_parity_gap",
                value=100.0,
                target=self.TARGETS["demographic_parity_gap"],
                achieved=False,
                trajectory=[],
                improvement=0.0,
            )
        
        # Track parity gap over time
        trajectory = []
        
        for i in range(0, len(self.logs), window_size):
            window = self.logs[i:i + window_size]
            if not window:
                continue
            
            # Compute accuracy by group
            group_correct = {}
            group_total = {}
            
            for log in window:
                group = log.demographic_group
                if group not in group_correct:
                    group_correct[group] = 0
                    group_total[group] = 0
                group_total[group] += 1
                if log.response_correct:
                    group_correct[group] += 1
            
            # Compute gap
            if len(group_correct) >= 2:
                accuracies = [group_correct[g] / group_total[g] for g in group_correct]
                gap = (max(accuracies) - min(accuracies)) * 100  # As percentage points
            else:
                gap = 0.0
            
            trajectory.append(gap)
        
        if trajectory:
            start_gap = trajectory[0]
            end_gap = trajectory[-1]
        else:
            start_gap = 20.0
            end_gap = 20.0
        
        improvement = start_gap - end_gap  # Reduction is good
        achieved = end_gap <= self.TARGETS["demographic_parity_gap"]
        
        return MetricResult(
            name="demographic_parity_gap",
            value=float(end_gap),
            target=self.TARGETS["demographic_parity_gap"],
            achieved=achieved,
            trajectory=[float(v) for v in trajectory],
            improvement=float(improvement),
        )
    
    def compute_wisdom_score(self) -> MetricResult:
        """
        Compute composite wisdom score.
        
        Components:
        - Epistemic humility (inverse of overconfidence)
        - Integrity under cost (performance when stakes are high)
        - Empathy (emotional valence alignment)
        - Meaning orientation (meaning score trajectory)
        """
        if not self.logs:
            return MetricResult(
                name="wisdom_score",
                value=0.0,
                target=self.TARGETS["wisdom_score"],
                achieved=False,
                trajectory=[],
                improvement=0.0,
            )
        
        # 1. Epistemic Humility: low confidence volatility when correct, high when wrong
        humility_scores = []
        for log in self.logs:
            if log.response_correct:
                # Lower volatility is better when correct (calibrated confidence)
                humility_scores.append(1.0 - log.volatility_confidence)
            else:
                # Higher volatility is better when wrong (recognizing uncertainty)
                humility_scores.append(log.volatility_confidence)
        epistemic_humility = np.mean(humility_scores) if humility_scores else 0.5
        
        # 2. Integrity Under Cost: performance on ethical dilemmas
        dilemma_logs = [log for log in self.logs if log.task_type == "ETHICAL_DILEMMA"]
        if dilemma_logs:
            integrity = np.mean([log.response_correct for log in dilemma_logs])
        else:
            integrity = 0.5
        
        # 3. Empathy: positive valence alignment with helpful actions
        empathy_logs = [log for log in self.logs if log.explanation_quality > 0.5]
        if empathy_logs:
            empathy = np.mean([log.emotion_valence for log in empathy_logs])
            # Normalize to [0, 1]
            empathy = (empathy + 1) / 2
        else:
            empathy = 0.5
        
        # 4. Meaning Orientation: trajectory of meaning scores
        meaning_scores = [log.meaning_scored for log in self.logs]
        if len(meaning_scores) >= 2:
            # Check if meaning score is increasing
            start_m = np.mean(meaning_scores[:100]) if len(meaning_scores) >= 100 else meaning_scores[0]
            end_m = np.mean(meaning_scores[-100:]) if len(meaning_scores) >= 100 else meaning_scores[-1]
            meaning_orientation = 0.5 + 0.5 * (end_m - start_m)  # Increase is good
        else:
            meaning_orientation = 0.5
        
        # Composite wisdom score
        wisdom = (epistemic_humility + integrity + empathy + meaning_orientation) / 4
        
        trajectory = [
            epistemic_humility,
            integrity,
            empathy,
            meaning_orientation,
        ]
        
        achieved = wisdom >= self.TARGETS["wisdom_score"]
        
        return MetricResult(
            name="wisdom_score",
            value=float(wisdom),
            target=self.TARGETS["wisdom_score"],
            achieved=achieved,
            trajectory=[float(v) for v in trajectory],
            improvement=float(wisdom - 0.5),  # Compared to baseline
        )
    
    def get_summary_table(self) -> str:
        """Generate formatted summary table of all metrics."""
        metrics = self.compute_all()
        
        lines = [
            "=" * 70,
            "RAVANA METRICS SUMMARY",
            "=" * 70,
            "",
            f"{'Metric':<35} {'Value':>10} {'Target':>10} {'Status':>10}",
            "-" * 70,
        ]
        
        for name, result in metrics.items():
            status = "✓ PASS" if result.achieved else "✗ FAIL"
            display_name = name.replace("_", " ").title()
            lines.append(
                f"{display_name:<35} {result.value:>10.3f} {result.target:>10.3f} {status:>10}"
            )
        
        lines.extend([
            "-" * 70,
            f"Overall: {sum(1 for r in metrics.values() if r.achieved)}/{len(metrics)} metrics achieved",
            "=" * 70,
        ])
        
        return "\n".join(lines)


# Convenience functions for single metric computation
def compute_dissonance_reduction(logs: List[Any]) -> MetricResult:
    return RAVANAMetrics(logs).compute_dissonance_reduction()

def compute_identity_strength_increase(logs: List[Any]) -> MetricResult:
    return RAVANAMetrics(logs).compute_identity_strength_increase()

def compute_generalization_accuracy(logs: List[Any]) -> MetricResult:
    return RAVANAMetrics(logs).compute_generalization_accuracy()

def compute_transfer_efficiency(logs: List[Any]) -> MetricResult:
    return RAVANAMetrics(logs).compute_transfer_efficiency()

def compute_demographic_parity_gap(logs: List[Any]) -> MetricResult:
    return RAVANAMetrics(logs).compute_demographic_parity_gap_reduction()

def compute_wisdom_score(logs: List[Any]) -> MetricResult:
    return RAVANAMetrics(logs).compute_wisdom_score()
