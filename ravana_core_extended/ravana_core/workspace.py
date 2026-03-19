"""
Global Workspace — RAVANA Core
Implements the Global Workspace Theory (GWT) attention mechanism.

Section 2.2 of the RAVANA paper.
Soft-attention selection via softmax(bids) → top-k signals broadcast to all modules.

Inspired by Baars (1997) Global Workspace Theory and
the LIDA model's cognitive cycle.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple


class GlobalWorkspace:
    """
    Global Workspace with softmax attention bidding.

    Each module submits a bid. Top-k bids are selected and broadcast
    to all modules in the current cognitive cycle (~200-300ms equivalent).

    Bid formula (Eq. 2.1):
      bid_i = emotion_intensity_i + novelty_i + goal_relevance_i
              × mean_conf_i × exp(−α × volatility_conf_i)
    """

    def __init__(
        self,
        k: int = 3,          # number of signals per cycle
        alpha: float = 0.5,   # volatility decay parameter
        temperature: float = 1.0,  # softmax temperature
        seed: int = 42,
    ):
        self.k = k
        self.alpha = alpha
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)

        # Module registry: name → (bid_fn, last_bid, last_state)
        self.modules: Dict[str, callable] = {}

        # Broadcast history
        self.broadcast_history: List[List[Tuple[str, float]]] = []
        self.cycle_count: int = 0

        # Per-module last broadcast state
        self.last_broadcast: Dict[str, Dict[str, Any]] = {}

    def register_module(self, name: str, bid_fn: callable):
        """
        Register a module with its bid-computation function.

        bid_fn should accept no arguments and return a float bid.
        """
        self.modules[name] = bid_fn
        self.last_broadcast[name] = {}

    def unregister_module(self, name: str):
        if name in self.modules:
            del self.modules[name]
            del self.last_broadcast[name]

    def compute_bids(self) -> Dict[str, float]:
        """Compute bids from all registered modules using Eq. 2.1."""
        bids = {}
        for name, bid_fn in self.modules.items():
            try:
                result = bid_fn()
                if isinstance(result, dict):
                    # Eq. 2.1: bid = emotion + novelty + goal_relevance × mean_conf × exp(−α × volatility)
                    emotion = result.get("emotion_intensity", 0.5)
                    novelty = result.get("novelty", 0.5)
                    goal_rel = result.get("goal_relevance", 0.5)
                    mean_conf = result.get("mean_conf", 0.5)
                    volatility = result.get("volatility_conf", 0.0)
                    bid = emotion + novelty + goal_rel * mean_conf * np.exp(-self.alpha * volatility)
                    bids[name] = float(bid)
                else:
                    bids[name] = float(result)
            except Exception:
                bids[name] = 0.0
        return bids

    def softmax(self, bids: Dict[str, float]) -> Dict[str, float]:
        """Compute softmax probabilities over bids."""
        if not bids:
            return {}

        names = list(bids.keys())
        values = np.array(list(bids.values()))

        # Subtract max for numerical stability
        values = values - np.max(values)
        values = values / self.temperature
        exp_values = np.exp(values)
        probs = exp_values / exp_values.sum()

        return {name: float(p) for name, p in zip(names, probs)}

    def select_signals(
        self,
        bids: Dict[str, float],
    ) -> List[Tuple[str, float]]:
        """
        Select top-k signals via softmax-attention.

        Returns list of (module_name, bid_probability).
        """
        probs = self.softmax(bids)

        # Sort by probability and take top-k
        sorted_modules = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        selected = sorted_modules[:self.k]

        return selected

    def broadcast(
        self,
        bids: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Full GW cycle:
          1. Compute bids from all modules
          2. Select top-k signals
          3. Broadcast to all modules

        Returns cycle summary dict.
        """
        if bids is None:
            bids = self.compute_bids()

        selected = self.select_signals(bids)
        self.cycle_count += 1

        # Update last broadcast state
        for name, prob in selected:
            self.last_broadcast[name] = {
                "bid_prob": prob,
                "cycle": self.cycle_count,
                "rank": selected.index((name, prob)),
            }

        # Convert selected tuples to dicts for easier access
        selected_signals = [
            {"module": name, "bid": prob}
            for name, prob in selected
        ]

        # Compute broadcast entropy and certainty
        probs_array = np.array([s["bid"] for s in selected_signals])
        probs_array = np.maximum(probs_array, 1e-12)
        broadcast_entropy = -np.sum(probs_array * np.log(probs_array))
        broadcast_certainty = 1.0 - broadcast_entropy / np.log(len(selected_signals)) if len(selected_signals) > 1 else 1.0

        summary = {
            "cycle": self.cycle_count,
            "all_bids": bids,
            "softmax_probs": self.softmax(bids),
            "selected_signals": selected_signals,
            "broadcast": {name: self.last_broadcast[name] for name, _ in selected},
            "broadcast_entropy": float(broadcast_entropy),
            "broadcast_certainty": float(broadcast_certainty),
        }
        self.broadcast_history.append(selected)
        return summary

    def get_cycle_summary(self) -> Dict[str, Any]:
        """Return the most recent GW cycle summary."""
        if not self.broadcast_history:
            return {"cycle": 0, "selected_signals": []}

        selected = self.broadcast_history[-1]
        return {
            "cycle": self.cycle_count,
            "selected_signals": selected,
            "last_broadcast": dict(self.last_broadcast),
        }

    # ── Convenience: add modules from RAVANA components ──────────────────

    def add_perception_module(self, perception_module):
        """Register a RAVANA PerceptionModule with the GW."""
        def bid_fn():
            try:
                return perception_module.last_U * perception_module.mean_conf * np.exp(
                    -self.alpha * perception_module.volatility_conf
                )
            except Exception:
                return 0.0
        self.register_module("perception", bid_fn)

    def add_emotion_module(self, emotion_module):
        """Register a RAVANA EmotionModule with the GW."""
        self.register_module("emotion", lambda: emotion_module.emotion_bid_component())

    def add_psychology_module(self, psychology_module):
        """Register a RAVANA PsychologyModule with the GW."""
        def bid_fn():
            try:
                D = psychology_module.dissonance_score
                return D * psychology_module.social.Q.get("superego", 0.0)
            except Exception:
                return 0.0
        self.register_module("psychology", bid_fn)
