"""
Emotion Module — RAVANA Core
Implements Russell's VAD (Valence-Arousal-Dominance) emotional model
with anticipation-driven emotion and reappraisal-based regulation.

Four-branch EI (Mayer-Salovey):
  Branch 1 — Perception:     handled externally via PerceptionModule
  Branch 2 — Use of Emotion: emotion-scaled GW bidding
  Branch 3 — Understanding:  causal emotion models (dV/dA/dD dynamics)
  Branch 4 — Regulation:     reappraisal (NOT suppression)
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple


class EmotionModule:
    """
    VAD emotional dynamics with coupled differential equations
    (Eq. 3.4–3.6 in the RAVANA paper).

    Supports:
    - Emotion perception (VAD state)
    - Anticipation-driven emotion (Eq. 3.7)
    - Empathy via Euclidean distance in VAD space
    - Reappraisal-focused regulation
    """

    # Default VAD parameters (Ekman/Russell conventions)
    VALENCE_RANGE = (-1.0, 1.0)
    AROUSAL_RANGE = (0.0, 1.0)
    DOMINANCE_RANGE = (0.0, 1.0)

    # VAD baseline (neutral state)
    BASELINE_VAD = np.array([0.0, 0.2, 0.5])

    # Euler integration step
    DT = 0.5

    def __init__(
        self,
        eta_v: float = 0.6,
        eta_a: float = 0.8,
        eta_d: float = 0.3,
        lambda_v: float = 0.2,
        lambda_a: float = 0.3,
        lambda_d: float = 0.2,
        seed: int = 42,
    ):
        self.eta_v, self.eta_a, self.eta_d = eta_v, eta_a, eta_d
        self.lambda_v, self.lambda_a, self.lambda_d = lambda_v, lambda_a, lambda_d
        self.rng = np.random.default_rng(seed)

        # Current VAD state
        self.vad = self.BASELINE_VAD.copy()

        # Emotion history for empathy
        self.vad_history: list[np.ndarray] = []

        # Emotional weights per situation (set by agent context)
        self.emotional_weights: Dict[str, float] = {
            "threat": 0.8,
            "opportunity": 0.5,
            "social": 0.6,
            "achievement": 0.4,
            "loss": 0.7,
            "neutral": 0.1,
        }

        # Registered emotion categories (for interpretation)
        self._emotion_lookup = {
            "joy":        np.array([ 0.9,  0.7,  0.8]),
            "anger":      np.array([-0.7,  0.9,  0.6]),
            "fear":       np.array([-0.8,  0.8,  0.2]),
            "sadness":    np.array([-0.6,  0.3,  0.3]),
            "surprise":   np.array([ 0.2,  0.8,  0.5]),
            "disgust":    np.array([-0.7,  0.5,  0.4]),
            "trust":      np.array([ 0.6,  0.4,  0.7]),
            "anticipation": np.array([ 0.5,  0.6,  0.6]),
            "neutral":    np.array([ 0.0,  0.2,  0.5]),
        }

    # ── VAD Dynamics ─────────────────────────────────────────────────────────

    def update(
        self,
        stimulus_valence: float,
        stimulus_arousal: float,
        stimulus_dominance: float,
        global_U: float = 0.0,
        dt: float = DT,
    ) -> Dict[str, float]:
        """
        Apply Russell's VAD differential equations (Eq. 3.4–3.6):
          dV/dt = ηv(stimulus_valence − V) − λvV
          dA/dt = ηa(stimulus_arousal + 0.3×global_U) − λa(A − baseline_A)
          dD/dt = ηd(stimulus_dominance − D) − λdD

        Args:
            stimulus_valence:   Desired valence (−1 to 1)
            stimulus_arousal:    Desired arousal (0 to 1)
            stimulus_dominance:  Desired dominance (0 to 1)
            global_U:            Perceptual uncertainty (increases arousal)
            dt:                  Integration timestep

        Returns:
            Updated VAD state vector
        """
        V, A, D = self.vad

        dV = self.eta_v * (stimulus_valence - V) - self.lambda_v * V
        dA = self.eta_a * (stimulus_arousal + 0.3 * global_U) - self.lambda_a * (A - self.BASELINE_VAD[1])
        dD = self.eta_d * (stimulus_dominance - D) - self.lambda_d * D

        self.vad += np.array([dV, dA, dD]) * dt

        # Clamp to valid ranges
        self.vad[0] = np.clip(self.vad[0], -1.0, 1.0)
        self.vad[1] = np.clip(self.vad[1], 0.0, 1.0)
        self.vad[2] = np.clip(self.vad[2], 0.0, 1.0)

        return self.current_state()

    def anticipate(
        self,
        positive_outcome_prob: float,
        negative_outcome_prob: float,
        arousal_positive: float = 0.5,
        arousal_negative: float = 0.7,
    ) -> float:
        """
        Anticipation-driven emotion (Eq. 3.7):
        A(t+Δt) = P(+) × arousal_positive + P(−) × arousal_negative

        Returns the anticipated arousal level.
        """
        return float(
            positive_outcome_prob * arousal_positive +
            negative_outcome_prob * arousal_negative
        )

    # ── Emotion Regulation (Branch 4) ───────────────────────────────────────

    def reappraise(
        self,
        original_interpretation: str,
        new_interpretation: str,
    ) -> Tuple[float, str]:
        """
        Reappraisal-based regulation (Section 3.3.4).
        Changes the emotional impact by reframing the situation.

        This reduces V (valence shift) WITHOUT blocking the emotion — cooler, not cold.

        Returns:
            (valence_shift, revised_description)
        """
        # Map interpretations to valences (simplified heuristic lookup)
        orig_vad = self._interpret_to_vad(original_interpretation)
        new_vad  = self._interpret_to_vad(new_interpretation)

        valence_shift = new_vad[0] - orig_vad[0]

        # Apply shift to current valence
        self.vad[0] = np.clip(self.vad[0] + valence_shift * 0.5, -1.0, 1.0)

        revised = (
            f"Reframed: {new_interpretation} "
            f"(valence shift: {valence_shift:+.2f})"
        )
        return float(valence_shift), revised

    # ── Empathy (Theory of Mind) ──────────────────────────────────────────────

    def empathy_distance(self, other_vad: np.ndarray) -> float:
        """
        Euclidean distance in VAD space (Section 3.3.3).
        Lower distance → higher empathy reward.

        empathy_reward = 1 − distance
        """
        distance = float(np.linalg.norm(self.vad - other_vad))
        return distance

    def empathy_reward(self, other_vad: np.ndarray) -> float:
        """Empathy reward: 1 − normalised distance (bounded [0, 1])."""
        max_dist = np.sqrt(3)  # max VAD distance (corner to corner of cube)
        dist = self.empathy_distance(other_vad)
        return float(np.clip(1.0 - dist / max_dist, 0.0, 1.0))

    # ── GW Bid Component ─────────────────────────────────────────────────────

    def emotion_bid_component(self) -> float:
        """
        Emotion intensity for GW bid (Eq. 2.1).
        High arousal + extreme valence = strong emotional salience.
        """
        V, A, D = self.vad
        intensity = A * (1.0 - 0.5 * abs(V))  # arousal modulated by valence extremity
        return float(intensity)

    # ── Utility ──────────────────────────────────────────────────────────────

    def current_state(self) -> Dict[str, float]:
        """Return human-readable VAD state."""
        V, A, D = self.vad
        return {
            "valence": float(V),
            "arousal": float(A),
            "dominance": float(D),
            "emotion_label": self._vad_to_label(self.vad),
            "intensity": self.emotion_bid_component(),
        }

    def apply_emotional_perturbation(self, magnitude: float) -> np.ndarray:
        """
        Add Gaussian noise scaled by dissonance magnitude
        (Dissonance Trigger 1 from Section 3.2.2).
        """
        noise = self.rng.normal(0, magnitude * 0.3, 3)
        self.vad = np.clip(self.vad + noise, -1.0, 1.0)
        return self.vad.copy()

    # ── Internal ─────────────────────────────────────────────────────────────

    def _interpret_to_vad(self, interpretation: str) -> np.ndarray:
        """Map free-text interpretation to VAD vector (simplified)."""
        interp_lower = interpretation.lower()
        for label, vad in self._emotion_lookup.items():
            if label in interp_lower:
                return vad

        # Default: positive/negative heuristics
        if any(w in interp_lower for w in ["good", "great", "win", "success", "helpful"]):
            return np.array([0.6, 0.5, 0.6])
        if any(w in interp_lower for w in ["bad", "lose", "fail", "threat", "harm"]):
            return np.array([-0.6, 0.6, 0.4])
        return self.BASELINE_VAD.copy()

    def _vad_to_label(self, vad: np.ndarray) -> str:
        """Map VAD vector to nearest emotion category."""
        V, A, D = vad
        best_label, best_dist = "neutral", float("inf")
        for label, ref_vad in self._emotion_lookup.items():
            dist = np.linalg.norm(vad - ref_vad)
            if dist < best_dist:
                best_dist = dist
                best_label = label
        return best_label

    def set_weight(self, category: str, weight: float):
        """Set emotional weight for a category."""
        self.emotional_weights[category] = np.clip(weight, 0.0, 1.0)
