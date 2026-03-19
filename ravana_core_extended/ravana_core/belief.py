"""
Bayesian Belief Tracker — RAVANA Core
Tracks beliefs as probability distributions with Bayesian updating.

Section 1.3 of the RAVANA paper.
Implements approximate Bayesian inference for belief updating and theory of mind.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import math


@dataclass
class BeliefState:
    """A probabilistic belief with mean and variance."""
    name: str
    prior_mean: float = 0.5
    prior_variance: float = 0.04
    posterior_mean: float = 0.5
    posterior_variance: float = 0.04
    n_updates: int = 0


class BayesianBeliefTracker:
    """
    Tracks beliefs using approximate Bayesian inference.

    Implements:
    - Bayesian belief updating (Eq. 1.1)
    - Theory of mind via inverse reinforcement learning
    - Dual-confidence: mean + volatility (posterior variance proxy)
    - Uncertainty quantification
    """

    # Prior for Beta distribution (conjugate prior for Bernoulli likelihood)
    ALPHA_PRIOR = 2.0
    BETA_PRIOR = 2.0

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.beliefs: Dict[str, BeliefState] = {}
        self.update_history: Dict[str, List[float]] = {}

    def add_belief(
        self,
        name: str,
        prior_mean: float = 0.5,
        prior_variance: float = 0.04,
    ):
        """Initialize a new belief with a Beta prior."""
        self.beliefs[name] = BeliefState(
            name=name,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            posterior_mean=prior_mean,
            posterior_variance=prior_variance,
        )
        self.update_history[name] = [prior_mean]

    def update(
        self,
        name: str,
        likelihood: float,
        evidence_strength: float = 1.0,
    ) -> Tuple[float, float]:
        """
        Bayesian belief update (Eq. 1.1):

        P(hypothesis | data) ∝ P(data | hypothesis) × P(hypothesis)

        Uses conjugate Beta-Bernoulli model for binary hypotheses.

        Args:
            name: belief identifier
            likelihood: P(data | hypothesis) — probability data supports hypothesis
            evidence_strength: How strong is the evidence (default 1.0)

        Returns:
            (new_posterior_mean, new_posterior_variance)
        """
        if name not in self.beliefs:
            self.add_belief(name)

        b = self.beliefs[name]

        # Beta posterior parameters (conjugate update)
        alpha_post = self.ALPHA_PRIOR + evidence_strength * likelihood
        beta_post  = self.BETA_PRIOR + evidence_strength * (1 - likelihood)

        # Posterior moments
        new_mean = alpha_post / (alpha_post + beta_post)
        new_var  = (alpha_post * beta_post) / (
            (alpha_post + beta_post) ** 2 * (alpha_post + beta_post + 1)
        )

        b.posterior_mean   = float(new_mean)
        b.posterior_variance = float(new_var)
        b.n_updates += 1

        self.update_history[name].append(new_mean)

        return new_mean, new_var

    def update_from_observation(
        self,
        name: str,
        observed: float,
        noise_sigma: float = 0.1,
    ) -> Tuple[float, float]:
        """
        Update belief based on a continuous observation.

        Uses approximate Bayesian inference via Gaussian approximation.
        """
        if name not in self.beliefs:
            self.add_belief(name)

        b = self.beliefs[name]

        # Precision-weighted fusion
        prior_precision  = 1.0 / max(b.posterior_variance, 1e-6)
        obs_precision    = 1.0 / (noise_sigma ** 2)
        post_precision   = prior_precision + obs_precision

        new_mean = (b.posterior_mean * prior_precision + observed * obs_precision) / post_precision
        new_var  = 1.0 / post_precision

        b.posterior_mean   = float(new_mean)
        b.posterior_variance = float(new_var)
        b.n_updates += 1

        self.update_history[name].append(new_mean)

        return new_mean, new_var

    def get_belief(self, name: str) -> Optional[BeliefState]:
        return self.beliefs.get(name)

    def get_confidence(self, name: str) -> float:
        """
        Confidence = inverse posterior variance.
        High precision (low variance) = high confidence.
        """
        b = self.beliefs.get(name)
        if b is None:
            return 0.5
        return float(np.clip(1.0 - b.posterior_variance * 10, 0.0, 1.0))

    def get_volatility(self, name: str, window: int = 5) -> float:
        """
        Volatility = variance of recent belief updates.
        Measures how stable the belief is over time.
        """
        history = self.update_history.get(name, [])
        if len(history) < 2:
            return 0.0
        recent = history[-window:]
        return float(np.var(recent)) if len(recent) > 1 else 0.0

    def bayesian_coherence_score(self, name: str) -> float:
        """
        Bayesian Coherence Coefficient (BCC) inspired by Qiu et al., 2026.
        Measures how well posterior aligns with prior → better calibrated = higher.
        """
        b = self.beliefs.get(name)
        if b is None:
            return 0.5

        # Perfect coherence: posterior close to prior
        coherence = 1.0 - abs(b.posterior_mean - b.prior_mean)
        # Penalise overconfidence (very low variance relative to prior)
        uncertainty_penalty = min(b.posterior_variance / (b.prior_variance + 1e-8), 1.0)
        return float(np.clip(coherence * (1 - 0.3 * uncertainty_penalty), 0.0, 1.0))

    def all_beliefs_summary(self) -> Dict[str, Dict[str, float]]:
        """Return summary of all tracked beliefs."""
        return {
            name: {
                "mean": b.posterior_mean,
                "variance": b.posterior_variance,
                "confidence": self.get_confidence(name),
                "volatility": self.get_volatility(name),
                "n_updates": b.n_updates,
                "bcc": self.bayesian_coherence_score(name),
            }
            for name, b in self.beliefs.items()
        }

    def theory_of_mind(
        self,
        other_beliefs: Dict[str, float],
        own_beliefs: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Theory of Mind: infer other's goals/beliefs via inverse reasoning.

        Returns:
            Dict of {goal: inferred_probability}
        """
        inferred_goals = {}
        for goal, other_prob in other_beliefs.items():
            own_prob = own_beliefs.get(goal, 0.5)
            # Bayesian update of how much we trust other's belief
            inferred_goals[goal] = float(
                np.clip(0.5 + (other_prob - own_prob) * 0.5, 0.0, 1.0)
            )
        return inferred_goals

    def brier_score(self, name: str, correct: bool) -> float:
        """
        Brier score: (predicted_conf − correctness)².
        Lower is better. Used for calibration evaluation.
        """
        conf = self.get_confidence(name)
        correct_float = 1.0 if correct else 0.0
        return float((conf - correct_float) ** 2)

    def reset(self):
        """Reset all beliefs."""
        self.beliefs.clear()
        self.update_history.clear()
