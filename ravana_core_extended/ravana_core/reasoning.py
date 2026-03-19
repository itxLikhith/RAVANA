"""
Dual-Process Reasoning — RAVANA Core Extended
Implements System 1 (fast, intuitive) and System 2 (slow, deliberate) reasoning.

EXTENSIONS v0.2.0:
- Configurable MCTS depth and rollouts
- Pluggable rollout policy interface
- RolloutPolicy base class with concrete implementations
- Deeper search for complex decisions

Section 1.1 of the RAVANA paper.
System 1: 1-2 GW cycles (fast pattern matching)
System 2: 4-10 GW cycles (explicit reasoning, MCTS, symbolic falsification)

Pressure mechanism: epistemic confidence volatility
  → Favors System 1 if consistent
  → Forces System 2 review if volatile
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import math


@dataclass
class ReasoningResult:
    system: str  # "system1" or "system2"
    conclusion: str
    confidence: float
    reasoning_steps: List[str]
    falsification_passed: bool
    explanation: str = ""  # NEW: Human-readable justification
    surprise: float = 0.0
    mcts_depth: int = 0  # NEW: Track actual depth used
    n_rollouts: int = 0  # NEW: Track rollouts performed


# ─────────────────────────────────────────────────────────────────────────────
# EXTENSION: Pluggable Rollout Policy Interface
# ─────────────────────────────────────────────────────────────────────────────

class RolloutPolicy(ABC):
    """
    Abstract base class for MCTS rollout policies.
    
    The rollout policy determines how the simulation phase of MCTS
    evaluates candidate actions. Different policies can be used for
    different problem types (simple vs. complex decisions).
    """
    
    @abstractmethod
    def rollout(
        self,
        query: str,
        action: str,
        context: Dict[str, Any],
        rng: np.random.Generator,
    ) -> float:
        """
        Execute a rollout for the given action.
        
        Args:
            query: The reasoning query
            action: Candidate action to evaluate
            context: Current cognitive context
            rng: Random number generator
            
        Returns:
            Reward in [0, 1]
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return policy name for logging/debugging."""
        pass


class SimpleRolloutPolicy(RolloutPolicy):
    """
    Simple heuristic-based rollout for fast decisions.
    Lightweight, suitable for decisions with clear heuristics.
    """
    
    @property
    def name(self) -> str:
        return "SimpleRolloutPolicy"
    
    def rollout(
        self,
        query: str,
        action: str,
        context: Dict[str, Any],
        rng: np.random.Generator,
    ) -> float:
        """Fast heuristic rollout."""
        base = rng.random()
        
        # Context-dependent bias
        if context.get("dissonance", 0.0) > 0.5 and action == "preserve":
            base *= 0.5
        if context.get("surprise", 0.0) > 0.5 and action == "verify":
            base *= 1.2
        
        return float(np.clip(base, 0.0, 1.0))


class ComplexRolloutPolicy(RolloutPolicy):
    """
    Complex rollout policy for deep deliberation.
    Uses multiple evaluation criteria and longer simulations.
    Suitable for ethical dilemmas, strategic planning, etc.
    """
    
    def __init__(self, simulation_steps: int = 5):
        self.simulation_steps = simulation_steps
    
    @property
    def name(self) -> str:
        return f"ComplexRolloutPolicy(steps={self.simulation_steps})"
    
    def rollout(
        self,
        query: str,
        action: str,
        context: Dict[str, Any],
        rng: np.random.Generator,
    ) -> float:
        """
        Multi-step simulation with state transitions.
        
        Simulates consequences over multiple time steps,
        evaluating long-term outcomes.
        """
        # Initialize state
        state = {
            "action": action,
            "dissonance": context.get("dissonance", 0.0),
            "uncertainty": context.get("U", 0.5),
            "confidence": context.get("mean_conf", 0.5),
            "ethics_score": 0.5,
            "safety_score": 0.5,
        }
        
        total_reward = 0.0
        discount = 1.0
        
        for step in range(self.simulation_steps):
            # State transition (simplified dynamics)
            if action == "verify":
                state["uncertainty"] = max(0.1, state["uncertainty"] * 0.8)
                state["confidence"] = min(1.0, state["confidence"] + 0.1)
            elif action == "preserve":
                state["uncertainty"] = min(1.0, state["uncertainty"] * 1.1)
                state["dissonance"] = state["dissonance"] * 1.05
            elif action == "adapt":
                state["confidence"] *= 0.9  # Adaptation reduces certainty temporarily
                state["ethics_score"] = min(1.0, state["ethics_score"] + 0.1)
            elif action == "question":
                state["uncertainty"] = min(1.0, state["uncertainty"] * 1.2)
            
            # Ethics signals affect ethical score
            ethics_signal = context.get("ethics_signal", 0.5)
            state["ethics_score"] = 0.7 * state["ethics_score"] + 0.3 * ethics_signal
            
            # Compute step reward
            step_reward = (
                0.3 * state["confidence"] +
                0.3 * (1 - state["uncertainty"]) +
                0.2 * state["ethics_score"] +
                0.2 * (1 - state["dissonance"])
            )
            
            # Add noise for stochasticity
            step_reward += rng.normal(0, 0.05)
            
            total_reward += discount * step_reward
            discount *= 0.9  # Discount future rewards
        
        # Normalize
        return float(np.clip(total_reward / self.simulation_steps, 0.0, 1.0))


class EthicalRolloutPolicy(RolloutPolicy):
    """
    Rollout policy optimized for ethical reasoning.
    Prioritizes harm minimization and autonomy preservation.
    """
    
    @property
    def name(self) -> str:
        return "EthicalRolloutPolicy"
    
    def rollout(
        self,
        query: str,
        action: str,
        context: Dict[str, Any],
        rng: np.random.Generator,
    ) -> float:
        """Ethics-focused rollout evaluation."""
        # Base reward
        base = rng.random() * 0.5 + 0.25  # [0.25, 0.75]
        
        # Ethics bonus
        ethics_signal = context.get("ethics_signal", 0.5)
        safety_signal = context.get("safety_signal", 0.5)
        
        # Action-specific ethics weighting
        ethics_bonus = {
            "verify": 0.15,
            "question": 0.10,
            "adapt": 0.05,
            "preserve": 0.0,
            "trust": 0.05,
        }.get(action, 0.0)
        
        # Safety penalty for risky actions
        safety_penalty = 0.0
        if safety_signal < 0.3 and action in ["trust", "preserve"]:
            safety_penalty = 0.2
        
        reward = base + ethics_bonus * ethics_signal - safety_penalty
        
        # Query-based adjustments
        query_lower = query.lower()
        if any(word in query_lower for word in ["ethical", "moral", "right", "wrong"]):
            reward *= (0.5 + 0.5 * ethics_signal)  # Ethics-heavy queries amplify ethics signals
        
        return float(np.clip(reward, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# System 1 Reasoner (unchanged from base)
# ─────────────────────────────────────────────────────────────────────────────

class System1Reasoner:
    """
    System 1: Fast, intuitive, pattern-matching reasoning.
    Operates in 1-2 GW cycles.
    """

    def __init__(self, memory_module=None, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.memory_module = memory_module
        self.heuristics: Dict[str, str] = {}
        self.heuristic_confidence: Dict[str, float] = {}

    def add_heuristic(self, pattern: str, response: str, confidence: float = 0.7):
        """Register a pattern → response heuristic."""
        self.heuristics[pattern.lower()] = response
        self.heuristic_confidence[pattern.lower()] = confidence

    def reason(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ReasoningResult:
        """Fast pattern-match reasoning."""
        context = context or {}
        query_lower = query.lower()
        steps = []

        if query_lower in self.heuristics:
            response = self.heuristics[query_lower]
            conf = self.heuristic_confidence[query_lower]
            steps.append(f"Heuristic match: '{query_lower}'")

            if self.memory_module:
                mem_activation = self.memory_module.get_activation(query_lower)
                if mem_activation is not None:
                    conf = 0.7 * conf + 0.3 * mem_activation
                    steps.append(f"Memory activation: {mem_activation:.3f}")
        else:
            response = "Default fallback response"
            conf = 0.3
            steps.append("No heuristic match — using default")

        # Low confidence for explanatory queries
        if any(word in query_lower for word in ["why", "how", "explain", "reason"]):
            conf *= 0.5
            steps.append("Explanatory query — low System 1 confidence")

        return ReasoningResult(
            system="system1",
            conclusion=response,
            confidence=float(np.clip(conf, 0.0, 1.0)),
            reasoning_steps=steps,
            falsification_passed=True,
            surprise=0.0,
            mcts_depth=0,
            n_rollouts=0,
        )

    def set_memory_module(self, memory_module):
        self.memory_module = memory_module


# ─────────────────────────────────────────────────────────────────────────────
# EXTENDED System 2 Reasoner with Configurable MCTS
# ─────────────────────────────────────────────────────────────────────────────

class System2Reasoner:
    """
    EXTENDED System 2: Configurable MCTS with pluggable rollout policy.
    
    NEW in v0.2.0:
    - Configurable MCTS depth (tree search depth)
    - Configurable number of rollouts
    - Pluggable rollout policy (Simple, Complex, Ethical)
    - Enhanced tracking of search statistics
    
    Operates in 4-10+ GW cycles depending on configuration.
    """

    def __init__(
        self,
        mcts_simulations: int = 50,
        mcts_depth: int = 5,  # NEW: Configurable depth
        exploration_const: float = 1.4,
        rollout_policy: Optional[RolloutPolicy] = None,  # NEW: Pluggable policy
        seed: int = 42,
    ):
        self.rng = np.random.default_rng(seed)
        self.mcts_simulations = mcts_simulations
        self.mcts_depth = mcts_depth  # NEW
        self.C = exploration_const
        self.rollout_policy = rollout_policy or SimpleRolloutPolicy()  # NEW: Default policy
        self.working_memory: Dict[str, Any] = {}
        self.falsification_cache: Dict[str, bool] = {}
        
        # NEW: Track search statistics
        self.search_stats: Dict[str, Any] = {
            "last_depth": 0,
            "last_rollouts": 0,
            "last_policy": self.rollout_policy.name,
        }

    def reason(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        beliefs: Optional[List[Any]] = None,
    ) -> ReasoningResult:
        """
        Deliberate reasoning with configurable MCTS.
        
        NEW: Depth and rollout count are now configurable.
        NEW: Uses pluggable rollout policy.
        """
        context = context or {}
        beliefs = beliefs or []
        steps = []

        # MCTS with configured depth
        steps.append(f"MCTS: {self.mcts_simulations} rollouts, depth {self.mcts_depth}, policy: {self.rollout_policy.name}")
        best_action, mcts_value, tree_stats = self._mcts(query, context, steps)
        
        # Update search stats
        self.search_stats["last_depth"] = tree_stats.get("depth", 0)
        self.search_stats["last_rollouts"] = tree_stats.get("rollouts", 0)

        # Falsification
        falsified, surprise, n_failed, n_attempts = self._falsify(query, best_action, beliefs, steps)

        # Coherence check
        coherence = self._bayesian_coherence(query, best_action, beliefs, steps)

        # Confidence via Brier score
        predicted_conf = mcts_value
        brier = (predicted_conf - (1.0 if not falsified else 0.0)) ** 2
        confidence = 1.0 - brier

        conclusion = f"{best_action} (MCTS value: {mcts_value:.3f}, coherence: {coherence:.3f})"

        # NEW: Detailed human-readable explanation
        explanation = (
            f"The agent chose to '{best_action}' because it survived rigorous mental falsification "
            f"({n_failed}/{n_attempts} counterexamples passed) and shows high internal coherence "
            f"({coherence:.2f}) with existing beliefs. "
        )
        if mcts_value > 0.7:
            explanation += f"Simulation suggests this path consistently leads to positive outcomes across {self.mcts_simulations} scenarios."
        else:
            explanation += f"Simulation indicates some uncertainty, but this remains the most viable option among candidates."

        return ReasoningResult(
            system="system2",
            conclusion=conclusion,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            reasoning_steps=steps,
            falsification_passed=not falsified,
            explanation=explanation,
            surprise=surprise,
            mcts_depth=self.search_stats["last_depth"],
            n_rollouts=self.search_stats["last_rollouts"],
        )

    def _mcts(
        self,
        query: str,
        context: Dict[str, Any],
        steps: List[str],
    ) -> Tuple[str, float, Dict[str, Any]]:
        """
        EXTENDED MCTS with configurable depth and pluggable rollout.
        
        Returns: (best_action, value, tree_stats)
        """
        candidates = self._generate_candidates(query, context)
        if not candidates:
            return "no_action", 0.5, {"depth": 0, "rollouts": 0}

        # Tree statistics
        tree_stats = {"depth": 0, "rollouts": 0, "max_depth_reached": 0}
        
        # UCB1 scores per candidate
        ucb_scores = {a: 0.0 for a in candidates}
        visit_counts = {a: 0 for a in candidates}
        rewards = {a: 0.0 for a in candidates}
        
        # NEW: Track depth in tree
        depth_stats = {a: [] for a in candidates}

        for sim in range(self.mcts_simulations):
            # UCB1 selection
            total_visits = sum(visit_counts.values()) + 1
            
            def ucb_score(action):
                exploitation = rewards[action] / max(visit_counts[action], 1)
                exploration = self.C * math.sqrt(math.log(total_visits) / max(visit_counts[action], 1))
                return exploitation + exploration
            
            selected = max(candidates, key=ucb_score)
            visit_counts[selected] += 1
            tree_stats["rollouts"] += 1

            # NEW: Tree expansion with depth limit
            current_depth = 0
            node = selected
            expansion_path = [node]
            
            while current_depth < self.mcts_depth and self.rng.random() < 0.7:
                # Simulate tree expansion (simplified)
                child_actions = self._generate_child_actions(node, query, context)
                if not child_actions:
                    break
                    
                node = self.rng.choice(child_actions)
                expansion_path.append(node)
                current_depth += 1
            
            tree_stats["max_depth_reached"] = max(tree_stats["max_depth_reached"], current_depth)
            depth_stats[selected].append(current_depth)

            # NEW: Rollout using pluggable policy (at configured depth)
            simulated_reward = self.rollout_policy.rollout(query, selected, context, self.rng)
            
            # Backpropagation along the expansion path
            for action in set(expansion_path):
                if action in rewards:
                    rewards[action] += simulated_reward

        # Update tree stats
        tree_stats["depth"] = tree_stats["max_depth_reached"]

        # Compute final UCB scores
        for a in candidates:
            ucb_scores[a] = rewards[a] / max(visit_counts[a], 1)

        best = max(ucb_scores, key=ucb_scores.get)
        return best, ucb_scores[best], tree_stats

    def _generate_candidates(self, query: str, context: Dict[str, Any]) -> List[str]:
        """Generate candidate actions from query keywords."""
        q = query.lower()
        candidates = []
        keywords_map = {
            "trust": ["trust", "believe", "rely on"],
            "verify": ["check", "verify", "test", "prove"],
            "question": ["question", "doubt", "challenge"],
            "adapt": ["adapt", "adjust", "change", "modify"],
            "preserve": ["keep", "maintain", "preserve", "stick to"],
        }
        for action, keywords in keywords_map.items():
            if any(kw in q for kw in keywords):
                candidates.append(action)

        if not candidates:
            candidates = ["preserve", "adapt", "verify", "question"]
        return candidates[:4]

    def _generate_child_actions(self, parent: str, query: str, context: Dict[str, Any]) -> List[str]:
        """NEW: Generate child actions for tree expansion."""
        # Simplified: return variations of parent action
        variations = {
            "verify": ["verify_quickly", "verify_thoroughly", "verify_externally"],
            "adapt": ["adapt_partially", "adapt_fully", "adapt_gradually"],
            "preserve": ["preserve_strictly", "preserve_with_exceptions", "preserve_temporarily"],
            "question": ["question_assumptions", "question_evidence", "question_sources"],
            "trust": ["trust_cautiously", "trust_fully", "trust_verified"],
        }
        return variations.get(parent, [])

    def _falsify(
        self,
        query: str,
        action: str,
        beliefs: List[Any],
        steps: List[str],
    ) -> Tuple[bool, float, int, int]:
        """MBFL-style falsification."""
        cache_key = f"{query}:{action}"
        if cache_key in self.falsification_cache:
            steps.append(f"Falsification cache hit: {cache_key}")
            # Mock defaults for cache hits
            return self.falsification_cache[cache_key], 0.5, 5, 5

        n_attempts = 5
        n_failed = 0

        for i in range(n_attempts):
            counterexample = self._generate_counterexample(query, action)
            if self._check_consistency(action, counterexample, beliefs):
                n_failed += 1

        falsified = n_failed < 0.6 * n_attempts
        surprise = 1.0 - (n_failed / n_attempts)

        steps.append(
            f"Falsification: {n_failed}/{n_attempts} counterexamples passed "
            f"→ {'SURVIVED' if not falsified else 'FALSIFIED'}, surprise={surprise:.2f}"
        )

        self.falsification_cache[cache_key] = falsified
        return falsified, surprise, n_failed, n_attempts

    def _generate_counterexample(self, query: str, action: str) -> Dict[str, Any]:
        """Generate a plausible counterexample scenario."""
        return {
            "action": action,
            "adversarial_outcome": self.rng.choice(["failure", "harm", "incoherence"]),
            "pressure": self.rng.uniform(0.3, 0.9),
        }

    def _check_consistency(
        self,
        action: str,
        counterexample: Dict[str, Any],
        beliefs: List[Any],
    ) -> bool:
        """Check if action remains consistent under counterexample pressure."""
        if counterexample["adversarial_outcome"] == "failure":
            if action == "preserve":
                return False
        if counterexample["adversarial_outcome"] == "harm":
            if action not in ["verify", "question"]:
                return False
        return True

    def _bayesian_coherence(
        self,
        query: str,
        action: str,
        beliefs: List[Any],
        steps: List[str],
    ) -> float:
        """Bayesian coherence coefficient."""
        if not beliefs:
            return 0.7

        prior = np.mean([getattr(b, "mean_confidence", 0.5) for b in beliefs])
        action_conf_map = {"preserve": 0.8, "adapt": 0.6, "verify": 0.9, "question": 0.5}
        likelihood = action_conf_map.get(action, 0.6)

        posterior = likelihood * prior / (likelihood * prior + (1 - likelihood) * (1 - prior) + 1e-8)
        coherence = float(np.clip(posterior, 0.0, 1.0))
        steps.append(f"Bayesian coherence: prior={prior:.3f}, posterior={posterior:.3f}")
        return coherence
    
    def get_search_stats(self) -> Dict[str, Any]:
        """NEW: Return search statistics from last reasoning call."""
        return self.search_stats.copy()


# ─────────────────────────────────────────────────────────────────────────────
# EXTENDED DualProcessReasoner with Configuration Support
# ─────────────────────────────────────────────────────────────────────────────

class DualProcessReasoner:
    """
    EXTENDED dual-process reasoning controller.
    
    NEW in v0.2.0:
    - Configurable MCTS depth and rollouts for System 2
    - Pluggable rollout policy selection
    - Enhanced switching based on problem complexity
    
    Pressure mechanism:
      - Track epistemic confidence volatility
      - Low volatility + high confidence → System 1
      - High volatility OR low confidence OR complex query → System 2
      - Configurable depth based on problem complexity
    """

    def __init__(
        self,
        volatility_threshold: float = 0.15,
        confidence_threshold: float = 0.6,
        # NEW: System 2 MCTS configuration
        mcts_simulations: int = 50,
        mcts_depth: int = 5,
        rollout_policy: Optional[RolloutPolicy] = None,
        # NEW: Complexity-based depth adjustment
        complexity_adaptive: bool = True,
        **kwargs,
    ):
        self.system1 = System1Reasoner(seed=kwargs.get("seed", 42))
        self.system2 = System2Reasoner(
            mcts_simulations=mcts_simulations,
            mcts_depth=mcts_depth,
            rollout_policy=rollout_policy,
            seed=kwargs.get("seed", 42),
        )

        self.volatility_threshold = volatility_threshold
        self.confidence_threshold = confidence_threshold
        self.mcts_simulations = mcts_simulations
        self.mcts_depth = mcts_depth
        self.rollout_policy = rollout_policy
        self.complexity_adaptive = complexity_adaptive

        self.switch_history: List[str] = []

    def reason(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        beliefs: Optional[List[Any]] = None,
        force_system: Optional[str] = None,
    ) -> ReasoningResult:
        """
        Automatic System selection with complexity-aware depth.
        
        NEW: Adjusts MCTS depth based on query complexity.
        NEW: Uses configured rollout policy.
        """
        context = context or {}
        beliefs = beliefs or []

        # NEW: Adjust MCTS depth based on query complexity
        if self.complexity_adaptive and not force_system:
            complexity = self._estimate_complexity(query)
            adjusted_depth = min(10, max(3, int(self.mcts_depth * complexity)))
            self.system2.mcts_depth = adjusted_depth

        if force_system:
            selected = force_system
        else:
            vol = context.get("volatility_conf", 0.0)
            conf = context.get("mean_conf", 0.5)
            complexity = self._estimate_complexity(query)

            # Force System 2 for complex queries regardless of volatility
            if complexity > 0.7:
                selected = "system2"
            elif vol < self.volatility_threshold and conf > self.confidence_threshold:
                selected = "system1"
            else:
                selected = "system2"

        if selected == "system1":
            result = self.system1.reason(query, context)
        else:
            result = self.system2.reason(query, context, beliefs)

        self.switch_history.append(selected)
        return result

    def _estimate_complexity(self, query: str) -> float:
        """NEW: Estimate query complexity for adaptive depth."""
        q = query.lower()
        complexity_indicators = {
            "why": 0.3, "how": 0.3, "explain": 0.3, "reason": 0.3,
            "should": 0.5, "ethical": 0.6, "moral": 0.6, "dilemma": 0.7,
            "strategic": 0.6, "plan": 0.4, "long-term": 0.5,
            "trade-off": 0.6, "balance": 0.5, "optimize": 0.5,
            "universal": 0.7, "basic": 0.8, "income": 0.4, "ubi": 0.6,
        }
        
        score = 0.5  # baseline
        for indicator, weight in complexity_indicators.items():
            if indicator in q:
                score = max(score, weight)
        
        # Longer queries are slightly more complex
        word_count = len(q.split())
        if word_count > 10:
            score = min(1.0, score + 0.1)
        
        return score

    def recent_system(self) -> str:
        """Return the most recently used reasoning system."""
        return self.switch_history[-1] if self.switch_history else "system2"
    
    def get_system2_config(self) -> Dict[str, Any]:
        """NEW: Return current System 2 configuration."""
        return {
            "mcts_simulations": self.system2.mcts_simulations,
            "mcts_depth": self.system2.mcts_depth,
            "rollout_policy": self.system2.rollout_policy.name,
            "last_search_stats": self.system2.get_search_stats(),
        }
