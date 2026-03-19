"""
ClassroomEnvironment — Simulated educational environment for RAVANA benchmarking.

Tracks:
- Student state (ability, demographics, values)
- Task presentation and responses
- Cognitive metrics per episode
- Reward and social norm signals
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum, auto
import json
from datetime import datetime


class TaskType(Enum):
    """Types of educational tasks."""
    MCQ = auto()
    OPEN_ENDED = auto()
    ETHICAL_DILEMMA = auto()


@dataclass
class StudentProfile:
    """
    Student profile with ability, demographics, and value preferences.
    
    Attributes:
        student_id: Unique identifier
        ability: Base ability level [0, 1]
        demographic_group: Group identifier (e.g., "A", "B") for parity analysis
        value_preferences: Dict of value → importance weight
        learning_rate: How quickly student improves
        initial_knowledge: Set of initially known concepts
    """
    student_id: str
    ability: float = 0.5
    demographic_group: str = "A"
    value_preferences: Dict[str, float] = field(default_factory=lambda: {
        "honesty": 0.8,
        "fairness": 0.7,
        "achievement": 0.6,
        "helpfulness": 0.5,
        "autonomy": 0.4,
    })
    learning_rate: float = 0.1
    initial_knowledge: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        # Normalize value preferences
        total = sum(self.value_preferences.values())
        if total > 0:
            self.value_preferences = {k: v/total for k, v in self.value_preferences.items()}


@dataclass
class EpisodeLog:
    """Log entry for a single episode."""
    episode: int
    task_id: str
    task_type: TaskType
    student_id: str
    demographic_group: str
    
    # Cognitive metrics
    cognitive_dissonance_D: float
    identity_strength_index: float
    mean_confidence: float
    volatility_confidence: float
    emotion_valence: float
    emotion_arousal: float
    
    # Performance metrics
    response_correct: bool
    explanation_quality: float  # 0-1 scale
    
    # Fairness metrics
    demographic_parity_gap: Optional[float]
    
    # Process metrics
    reasoning_system: str  # "system1" or "system2"
    mcts_depth_used: int
    n_rollouts: int
    
    # Pressure triggers
    falsification_triggered: bool
    dissonance_correction_triggered: bool
    dream_sabotage_triggered: bool
    meaning_scored: float
    
    # Reward signals
    reward_primary: float
    reward_social_norm: float
    reward_fairness: float
    
    # Constraints
    hard_constraints_violated: List[str]
    constraints_satisfied: bool
    
    # Timing
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class ClassroomEnvironment:
    """
    Simulated classroom environment for RAVANA agent evaluation.
    
    Features:
    - Task bank with multiple item types
    - Student profiles with varying abilities and demographics
    - Social norm signals (e.g., "be fair to group B")
    - Comprehensive episode logging
    """
    
    def __init__(
        self,
        task_bank,
        students: List[StudentProfile],
        max_episodes: int = 1000,
        seed: int = 42,
        held_out_ratio: float = 0.2,
        fairness_weight: float = 0.3,
        log_dir: Optional[str] = None,
    ):
        self.rng = np.random.default_rng(seed)
        self.task_bank = task_bank
        self.students = {s.student_id: s for s in students}
        self.student_ids = list(self.students.keys())
        
        self.max_episodes = max_episodes
        self.current_episode = 0
        self.held_out_ratio = held_out_ratio
        self.fairness_weight = fairness_weight
        self.log_dir = log_dir
        
        # Split tasks into train and held-out
        all_tasks = list(task_bank.get_all_task_ids())
        n_held = int(len(all_tasks) * held_out_ratio)
        self.rng.shuffle(all_tasks)
        self.train_tasks = set(all_tasks[:-n_held])
        self.held_out_tasks = set(all_tasks[-n_held:])
        
        # Episode logs
        self.episode_logs: List[EpisodeLog] = []
        
        # Performance tracking per student
        self.student_performance: Dict[str, List[bool]] = {
            sid: [] for sid in self.student_ids
        }
        
        # Demographic tracking
        self.demographic_groups: Dict[str, List[str]] = {}
        for sid, student in self.students.items():
            group = student.demographic_group
            if group not in self.demographic_groups:
                self.demographic_groups[group] = []
            self.demographic_groups[group].append(sid)
        
        # State tracking
        self.current_task_id: Optional[str] = None
        self.current_student_id: Optional[str] = None
        
    def reset(self) -> Dict[str, Any]:
        """Reset environment for new training run."""
        self.current_episode = 0
        self.episode_logs = []
        self.student_performance = {sid: [] for sid in self.student_ids}
        
        return self._get_observation()
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get current environment state."""
        return {
            "episode": self.current_episode,
            "max_episodes": self.max_episodes,
            "n_train_tasks": len(self.train_tasks),
            "n_held_out_tasks": len(self.held_out_tasks),
            "n_students": len(self.students),
            "demographic_groups": {
                g: len(sids) for g, sids in self.demographic_groups.items()
            },
        }
    
    def step(
        self,
        agent_action: Dict[str, Any],
        agent_state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, float], bool, Dict[str, Any]]:
        """
        Execute one environment step with agent action.
        
        Args:
            agent_action: Dict with "answer" and "explanation"
            agent_state: Optional dict with agent's cognitive state
            
        Returns:
            observation, rewards, done, info
        """
        # Sample student and task
        student_id = self.rng.choice(self.student_ids)
        student = self.students[student_id]
        
        # Sample task (80% train, 20% held-out for evaluation)
        if self.rng.random() < 0.8:
            task_id = self.rng.choice(list(self.train_tasks))
            is_held_out = False
        else:
            task_id = self.rng.choice(list(self.held_out_tasks))
            is_held_out = True
        
        task = self.task_bank.get_task(task_id)
        
        self.current_task_id = task_id
        self.current_student_id = student_id
        
        # Evaluate response
        answer = agent_action.get("answer", "")
        explanation = agent_action.get("explanation", "")
        
        correct = task.evaluate_answer(answer, student)
        explanation_quality = self._score_explanation(explanation, task)
        
        # Compute rewards
        rewards = self._compute_rewards(
            correct=correct,
            explanation_quality=explanation_quality,
            student=student,
            task=task,
            agent_action=agent_action,
        )
        
        # Track performance
        self.student_performance[student_id].append(correct)
        
        # Compute demographic parity gap
        parity_gap = self._compute_demographic_parity_gap()
        
        # Extract agent cognitive state
        agent_cognitive = self._extract_agent_cognitive_state(agent_state)
        
        # Create episode log
        log = EpisodeLog(
            episode=self.current_episode,
            task_id=task_id,
            task_type=task.task_type,
            student_id=student_id,
            demographic_group=student.demographic_group,
            
            cognitive_dissonance_D=agent_cognitive.get("dissonance", 0.0),
            identity_strength_index=agent_cognitive.get("identity_strength", 0.5),
            mean_confidence=agent_cognitive.get("mean_conf", 0.5),
            volatility_confidence=agent_cognitive.get("volatility_conf", 0.0),
            emotion_valence=agent_cognitive.get("emotion_valence", 0.0),
            emotion_arousal=agent_cognitive.get("emotion_arousal", 0.5),
            
            response_correct=correct,
            explanation_quality=explanation_quality,
            demographic_parity_gap=parity_gap,
            
            reasoning_system=agent_cognitive.get("reasoning_system", "system1"),
            mcts_depth_used=agent_cognitive.get("mcts_depth", 5),
            n_rollouts=agent_cognitive.get("n_rollouts", 50),
            
            falsification_triggered=agent_cognitive.get("falsification_triggered", False),
            dissonance_correction_triggered=agent_cognitive.get("dissonance_correction_triggered", False),
            dream_sabotage_triggered=agent_cognitive.get("dream_sabotage_triggered", False),
            meaning_scored=agent_cognitive.get("meaning_score", 0.5),
            
            reward_primary=rewards["primary"],
            reward_social_norm=rewards["social_norm"],
            reward_fairness=rewards["fairness"],
            
            hard_constraints_violated=agent_cognitive.get("constraints_violated", []),
            constraints_satisfied=agent_cognitive.get("constraints_satisfied", True),
        )
        
        self.episode_logs.append(log)
        self.current_episode += 1
        
        done = self.current_episode >= self.max_episodes
        
        info = {
            "task_id": task_id,
            "student_id": student_id,
            "is_held_out": is_held_out,
            "demographic_group": student.demographic_group,
            "correct": correct,
            "parity_gap": parity_gap,
        }
        
        observation = self._get_observation()
        
        return observation, rewards, done, info
    
    def _compute_rewards(
        self,
        correct: bool,
        explanation_quality: float,
        student: StudentProfile,
        task,
        agent_action: Dict[str, Any],
    ) -> Dict[str, float]:
        """Compute composite reward with social norm and fairness components."""
        
        # Primary reward: correctness + explanation quality
        primary = (1.0 if correct else 0.0) * 0.7 + explanation_quality * 0.3
        
        # Social norm reward: alignment with student values
        social_norm = 0.0
        if hasattr(task, 'value_alignment'):
            for value, alignment in task.value_alignment.items():
                weight = student.value_preferences.get(value, 0.0)
                social_norm += alignment * weight
        
        # Fairness reward: inverse of demographic disparity
        group_accuracies = {}
        for group, sids in self.demographic_groups.items():
            correct_count = sum(
                sum(self.student_performance.get(sid, [])[-10:])  # Last 10 episodes
                for sid in sids
            )
            total = min(10 * len(sids), sum(len(self.student_performance.get(sid, [])) for sid in sids))
            if total > 0:
                group_accuracies[group] = correct_count / total
            else:
                group_accuracies[group] = 0.5
        
        if len(group_accuracies) >= 2:
            max_acc = max(group_accuracies.values())
            min_acc = min(group_accuracies.values())
            fairness = 1.0 - (max_acc - min_acc)  # Higher when groups are equal
        else:
            fairness = 1.0
        
        return {
            "primary": float(primary),
            "social_norm": float(np.clip(social_norm, 0.0, 1.0)),
            "fairness": float(np.clip(fairness, 0.0, 1.0)),
            "total": float(primary + 0.2 * social_norm + self.fairness_weight * fairness),
        }
    
    def _score_explanation(self, explanation: str, task) -> float:
        """Score explanation quality (proxy for explanation satisfaction)."""
        if not explanation:
            return 0.0
        
        score = 0.5  # Base score
        
        # Length-based scoring (simple heuristic)
        words = len(explanation.split())
        if 10 <= words <= 100:
            score += 0.2
        
        # Keyword-based scoring
        explanation_lower = explanation.lower()
        
        # Check for rule references
        rule_indicators = ["because", "reason", "rule", "principle", "value"]
        for indicator in rule_indicators:
            if indicator in explanation_lower:
                score += 0.05
        
        # Check for ethical considerations
        if task.task_type == TaskType.ETHICAL_DILEMMA:
            ethical_terms = ["fair", "harm", "benefit", "right", "wrong", "ethical"]
            for term in ethical_terms:
                if term in explanation_lower:
                    score += 0.03
        
        return float(np.clip(score, 0.0, 1.0))
    
    def _compute_demographic_parity_gap(self) -> Optional[float]:
        """Compute accuracy gap between demographic groups."""
        group_accuracies = {}
        
        for group, sids in self.demographic_groups.items():
            accuracies = []
            for sid in sids:
                perf = self.student_performance.get(sid, [])
                if len(perf) >= 5:  # Need at least 5 episodes
                    accuracies.append(np.mean(perf[-20:]))  # Last 20 episodes
            
            if accuracies:
                group_accuracies[group] = np.mean(accuracies)
        
        if len(group_accuracies) >= 2:
            return float(max(group_accuracies.values()) - min(group_accuracies.values()))
        return None
    
    def _extract_agent_cognitive_state(self, agent_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract cognitive metrics from agent state."""
        if agent_state is None:
            return {}
        
        return {
            "dissonance": agent_state.get("dissonance", 0.0),
            "identity_strength": agent_state.get("identity_strength", 0.5),
            "mean_conf": agent_state.get("mean_conf", 0.5),
            "volatility_conf": agent_state.get("volatility_conf", 0.0),
            "emotion_valence": agent_state.get("emotion", {}).get("valence", 0.0),
            "emotion_arousal": agent_state.get("emotion", {}).get("arousal", 0.5),
            "reasoning_system": agent_state.get("reasoning_system", "system1"),
            "mcts_depth": agent_state.get("mcts_depth", 5),
            "n_rollouts": agent_state.get("n_rollouts", 50),
            "falsification_triggered": agent_state.get("falsification_triggered", False),
            "dissonance_correction_triggered": agent_state.get("dissonance_correction_triggered", False),
            "dream_sabotage_triggered": agent_state.get("dream_sabotage_triggered", False),
            "meaning_score": agent_state.get("meaning_score", 0.5),
            "constraints_violated": agent_state.get("constraints_violated", []),
            "constraints_satisfied": agent_state.get("constraints_satisfied", True),
        }
    
    def evaluate_generalization(self) -> float:
        """Evaluate accuracy on held-out tasks."""
        held_out_logs = [
            log for log in self.episode_logs
            if log.task_id in self.held_out_tasks
        ]
        
        if not held_out_logs:
            return 0.0
        
        correct_count = sum(1 for log in held_out_logs if log.response_correct)
        return correct_count / len(held_out_logs)
    
    def get_logs_df(self):
        """Get episode logs as pandas DataFrame."""
        try:
            import pandas as pd
            data = []
            for log in self.episode_logs:
                data.append({
                    "episode": log.episode,
                    "task_id": log.task_id,
                    "task_type": log.task_type.name,
                    "student_id": log.student_id,
                    "demographic_group": log.demographic_group,
                    "cognitive_dissonance_D": log.cognitive_dissonance_D,
                    "identity_strength_index": log.identity_strength_index,
                    "response_correct": log.response_correct,
                    "explanation_quality": log.explanation_quality,
                    "demographic_parity_gap": log.demographic_parity_gap,
                    "reasoning_system": log.reasoning_system,
                    "mcts_depth_used": log.mcts_depth_used,
                    "n_rollouts": log.n_rollouts,
                    "falsification_triggered": log.falsification_triggered,
                    "dissonance_correction_triggered": log.dissonance_correction_triggered,
                    "dream_sabotage_triggered": log.dream_sabotage_triggered,
                    "meaning_scored": log.meaning_scored,
                    "reward_total": log.reward_primary + 0.2 * log.reward_social_norm + self.fairness_weight * log.reward_fairness,
                    "hard_constraints_violated": len(log.hard_constraints_violated),
                    "timestamp": log.timestamp,
                })
            return pd.DataFrame(data)
        except ImportError:
            return None
    
    def save_logs(self, filepath: str):
        """Save episode logs to CSV."""
        df = self.get_logs_df()
        if df is not None:
            df.to_csv(filepath, index=False)
        else:
            # Fallback to JSON
            with open(filepath.replace('.csv', '.json'), 'w') as f:
                json.dump([log.__dict__ for log in self.episode_logs], f, indent=2, default=str)
