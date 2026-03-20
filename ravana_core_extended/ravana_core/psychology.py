"""
Psychology Module — RAVANA Core
Implements ACT-R production rules, Cognitive Dissonance Engine (CDE),
and Social Norm Simulation (Freud's Id-Ego-Superego drive arbitration).

Section 3.2 of the RAVANA paper.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field


@dataclass
class Belief:
    """A symbolic belief proposition."""
    proposition: str
    mean_confidence: float = 0.5
    volatility: float = 0.0
    emotional_weight: float = 0.5
    activation: float = 0.0
    retrieval_prob: float = 0.0
    identity_strength: float = 0.0  # how central to self-concept


@dataclass
class Action:
    """A candidate or executed action."""
    description: str
    value: float = 0.0
    dissonance_cost: float = 0.0
    drive_source: str = "ego"  # id | ego | superego


@dataclass
class IdentityCommitment:
    """Cross-context identity commitment (Section 5.4)."""
    name: str
    strength: float = 0.5
    beliefs: List[str] = field(default_factory=list)
    consistency_threshold: float = 0.7


class CognitiveDissonanceEngine:
    """
    Computes cognitive dissonance D (Eq. 3.3) and triggers resolution mechanisms.

    D = Σ|belief_i − action_j| × mean_conf_i × emotional_weight_k
      + Σ|identity_strength − action_alignment| × (1 + context_variance)
      + commitment_strength × |commitment_intent − action| × free_choice_multiplier
    """

    DISSONANCE_THRESHOLD = 0.5

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.dissonance_history: list[float] = []

    def compute_dissonance(
        self,
        beliefs: List[Belief],
        action: Action,
        identity_commitments: List[IdentityCommitment],
        context_variance: float = 0.1,
        free_choice_multiplier: float = 1.0,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute total dissonance score.

        Returns:
            (dissonance_score, breakdown_dict)
        """
        # Component 1: Belief-action conflict
        belief_component = 0.0
        for b in beliefs:
            conflict = abs(b.mean_confidence - action.value) * b.mean_confidence * b.emotional_weight
            belief_component += conflict

        # Component 2: Identity-action conflict
        identity_component = 0.0
        for ic in identity_commitments:
            alignment = 1.0 - action.dissonance_cost  # proxy
            identity_component += abs(ic.strength - alignment) * (1.0 + context_variance)

        # Component 3: Commitment violation
        commitment_component = 0.0
        for ic in identity_commitments:
            if ic.name in action.description.lower():
                commitment_component += ic.strength * free_choice_multiplier * abs(0.5 - action.value)

        total_dissonance = belief_component + identity_component + commitment_component
        self.dissonance_history.append(total_dissonance)

        breakdown = {
            "belief_conflict": belief_component,
            "identity_conflict": identity_component,
            "commitment_violation": commitment_component,
            "total": total_dissonance,
        }
        return float(total_dissonance), breakdown

    def is_dissonant(self, D: float) -> bool:
        """Check if dissonance exceeds threshold."""
        return D > self.threshold

    def trigger_resolution(
        self,
        D: float,
        current_action: Action,
        reappraise_available: bool = True,
    ) -> Dict[str, Any]:
        """
        Returns resolution strategy based on dissonance magnitude.

        Dissonance Triggers (Section 3.2.2):
        1. D < τ: No action needed
        2. D < 1.5τ: Reappraisal (reframe situation)
        3. D < 2.5τ: Behavioral correction
        4. D ≥ 2.5τ: Belief change forced
        """
        if D < self.threshold:
            return {"strategy": "none", "action": None, "description": "Dissonance within tolerance"}

        ratio = D / self.threshold

        if ratio < 1.5:
            return {
                "strategy": "reappraisal",
                "action": "reframe",
                "description": "Mild dissonance — reframe situation to reduce conflict",
                "dissonance_cost": D * 0.2,
            }
        elif ratio < 2.5:
            return {
                "strategy": "behavioral_correction",
                "action": "modify_action",
                "description": "Moderate dissonance — correct behavior to align with beliefs",
                "dissonance_cost": D * 0.5,
            }
        else:
            return {
                "strategy": "belief_change",
                "action": "update_belief",
                "description": "High dissonance — belief must be updated or commitment weakened",
                "dissonance_cost": D * 0.8,
            }


class ProductionSystem:
    """
    ACT-R-style production rule system (Section 3.2.1).

    Rules fire based on utility = base_utility + global_U × noise
    Memory chunks have activation levels updated by relevance × confidence.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.productions: List[Tuple[callable, callable, float]] = []
        self.declarative_memory: Dict[str, Dict[str, Any]] = {}
        self.current_confidence: float = 0.5

    def add_production(
        self,
        condition_fn: callable,
        action_fn: callable,
        base_utility: float = 0.0,
        description: str = "",
    ):
        """Register a (condition, action, utility) production rule."""
        self.productions.append((condition_fn, action_fn, base_utility, description))

    def match_and_fire(
        self,
        state: Dict[str, Any],
        global_U: float = 0.0,
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Evaluate all productions, add noise, fire the highest-utility rule.

        Returns:
            (new_state, fired_rule_description)
        """
        utilities = []
        for cond, action, base_util, desc in self.productions:
            try:
                matches = cond(state)
            except Exception:
                matches = False

            if matches:
                noise = global_U * self.rng.normal(0, 0.1)
                utilities.append(base_util + noise)
            else:
                utilities.append(-float("inf"))

        if not utilities or max(utilities) == -float("inf"):
            return None, "no_match"

        best_idx = int(np.argmax(utilities))
        _, action_fn, _, desc = self.productions[best_idx]

        try:
            new_state = action_fn(state)
        except Exception as e:
            return state, f"error: {e}"

        return new_state, desc

    def add_memory_chunk(
        self,
        name: str,
        proposition: str,
        base_activation: float = 0.0,
    ):
        """Add a chunk to declarative memory."""
        self.declarative_memory[name] = {
            "proposition": proposition,
            "activation": base_activation,
            "retrieval_prob": 0.0,
        }

    def update_memory(self, context: Dict[str, Any]):
        """
        Update chunk activations: A = base + Σ(relevance × conf)
        and recompute retrieval probabilities (softmax).
        """
        for name, chunk in self.declarative_memory.items():
            relevance = self._compute_relevance(chunk, context)
            chunk["activation"] += relevance * self.current_confidence

        # Retrieval probability via softmax
        activations = np.array([c["activation"] for c in self.declarative_memory.values()])
        if len(activations) > 0 and np.ptp(activations) > 0:
            exp_act = np.exp(activations - np.max(activations))
            probs = exp_act / exp_act.sum()
            for i, name in enumerate(self.declarative_memory):
                self.declarative_memory[name]["retrieval_prob"] = float(probs[i])

    def _compute_relevance(self, chunk: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Heuristic relevance: shared keys between chunk proposition and context."""
        prop_words = set(chunk["proposition"].lower().split())
        context_values = set(str(v).lower() for v in context.values())
        overlap = len(prop_words & context_values)
        return float(overlap) / (len(prop_words) + 1)


class SocialNormModule:
    """
    Freud's Id-Ego-Superego drive arbitration (Section 3.2.3).

    Three independent RL policies compete:
      — Id:      pleasure maximisation
      — Ego:     safety maximisation
      — Superego: ethics / social norm maximisation
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.weights = {
            "id": 0.0,
            "ego": 0.0,
            "superego": 0.0,
        }
        # Each drive has a Q-value (learned via interaction)
        self.Q = {k: 0.0 for k in self.weights}

    def value(self, state: Dict[str, Any], drive: str) -> float:
        """Return Q-value for a drive given current state."""
        # Simplified linear value function
        base = self.Q[drive]
        if drive == "id":
            return base + state.get("pleasure_signal", 0.0) * 0.5
        elif drive == "ego":
            return base + state.get("safety_signal", 0.0) * 0.5
        else:  # superego
            return base + state.get("ethics_signal", 0.0) * 0.5

    def arbitrate(
        self,
        state: Dict[str, Any],
    ) -> Tuple[str, float, np.ndarray]:
        """
        Probabilistic drive selection via softmax over Q-values.

        Returns:
            (winning_drive, post_action_dissonance, all_action_probs)
        """
        bids = {d: self.value(state, d) for d in self.weights}
        bid_array = np.array(list(bids.values()))
        probs = self._softmax(bid_array, temperature=1.0)

        drive_order = list(bids.keys())
        chosen_idx = int(np.argmax(probs))
        winning_drive = drive_order[chosen_idx]

        # Dissonance = if superego was overridden
        dissonance = 0.0
        if winning_drive != "superego":
            dissonance = probs[2] * 0.5  # superego probability × penalty

        return winning_drive, float(dissonance), probs

    def update_Q(self, drive: str, reward: float, alpha: float = 0.1):
        """Simple Q-learning update."""
        self.Q[drive] += alpha * (reward - self.Q[drive])

    def _softmax(self, x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        x = x / temperature
        x = x - np.max(x)
        exp_x = np.exp(x)
        return exp_x / exp_x.sum()


class PsychologyModule:
    """
    Top-level psychology module integrating:
      — ProductionSystem (ACT-R)
      — CognitiveDissonanceEngine (CDE)
      — SocialNormModule (Id-Ego-Superego)
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.productions = ProductionSystem(seed=seed)
        self.cde = CognitiveDissonanceEngine()
        self.social = SocialNormModule(seed=seed)

        self.beliefs: List[Belief] = []
        self.commitments: List[IdentityCommitment] = []
        self.dissonance_score: float = 0.0

        self._setup_default_productions()

    def _setup_default_productions(self):
        """Register RAVANA's core ACT-R productions."""

        # P1: When high dissonance → trigger reappraisal
        def cond_high_dissonance(state):
            return state.get("dissonance", 0.0) > self.cde.threshold

        def act_reappraise(state):
            return {**state, "strategy": "reappraisal"}

        self.productions.add_production(
            cond_high_dissonance, act_reappraise,
            base_utility=0.8,
            description="Trigger reappraisal on high dissonance"
        )

        # P2: When superego override → record ethical violation
        def cond_superego_override(state):
            return state.get("winning_drive") == "id" and state.get("dissonance", 0.0) > 0.3

        def act_ethical_violation(state):
            return {**state, "ethical_violation": True, "dissonance": state.get("dissonance", 0.0) + 0.3}

        self.productions.add_production(
            cond_superego_override, act_ethical_violation,
            base_utility=0.6,
            description="Record ethical violation when Id overrides Superego"
        )

    def add_belief(self, proposition: str, confidence: float = 0.5, emotional_weight: float = 0.5):
        self.beliefs.append(Belief(
            proposition=proposition,
            mean_confidence=confidence,
            emotional_weight=emotional_weight,
        ))

    def add_commitment(self, name: str, strength: float = 0.5, beliefs: List[str] = None):
        self.commitments.append(IdentityCommitment(
            name=name,
            strength=strength,
            beliefs=beliefs or [],
        ))

    def process(self, state: Dict[str, Any], global_U: float = 0.0) -> Dict[str, Any]:
        """
        Full psychology pipeline for one cognitive cycle.

        1. Social drive arbitration
        2. Cognitive dissonance computation
        3. ACT-R production firing
        4. Memory update
        """
        # Step 1: Drive arbitration
        winning_drive, dissonance, probs = self.social.arbitrate(state)

        # Step 2: Dissonance computation
        if self.beliefs and "action_value" in state:
            action = Action(
                description=state.get("action_description", ""),
                value=state["action_value"],
                dissonance_cost=1.0 - state["action_value"],
                drive_source=winning_drive,
            )
            D, breakdown = self.cde.compute_dissonance(
                self.beliefs, action, self.commitments
            )
            self.dissonance_score = D
        else:
            D, breakdown = 0.0, {}
            action = None

        # Step 3: Production firing
        enriched_state = {
            **state,
            "winning_drive": winning_drive,
            "drive_probs": {k: float(v) for k, v in zip(self.weights.keys(), probs)},
            "dissonance": float(D),
            "dissonance_breakdown": breakdown,
        }
        new_state, fired = self.productions.match_and_fire(enriched_state, global_U)

        # Step 4: Memory update
        self.productions.update_memory(enriched_state)

        return new_state or enriched_state

    def forge_or_dissolve_commitments(self, meaning_score: float, current_beliefs: List[Belief]):
        """
        Dynamically update identity commitments (Section 5.4).
        
        Meaning > 0.8 → Forge new commitment or strengthen existing.
        Meaning < 0.2 → Weaken or dissolve commitment.
        """
        # Strengthen/Weaken existing
        for ic in self.commitments:
            if meaning_score > 0.8:
                ic.strength = min(1.0, ic.strength + 0.05)
            elif meaning_score < 0.2:
                ic.strength = max(0.0, ic.strength - 0.005)
        
        # Remove dissolved commitments
        self.commitments = [ic for ic in self.commitments if ic.strength > 0.01]
        
        # Forge new commitment if high meaning and new strong beliefs exist
        if meaning_score > 0.9 and len(self.commitments) < 5:
            strong_beliefs = [b for b in current_beliefs if b.mean_confidence > 0.8]
            if strong_beliefs:
                # Pick a random strong belief not already in commitments
                existing_beliefs = set()
                for ic in self.commitments:
                    existing_beliefs.update(ic.beliefs)
                
                candidates = [b for b in strong_beliefs if b.proposition not in existing_beliefs]
                if candidates:
                    new_b = candidates[0]
                    self.add_commitment(
                        name=f"identity_{new_b.proposition[:20]}",
                        strength=0.5,
                        beliefs=[new_b.proposition]
                    )

    @property
    def weights(self):
        return self.social.weights
