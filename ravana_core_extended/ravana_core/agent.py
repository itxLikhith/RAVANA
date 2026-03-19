"""
RavanaAgent — RAVANA Core Extended v0.2.0
Top-level cognitive agent integrating all RAVANA modules with extension support.

Section 2 of the RAVANA paper.
The agent runs a cognitive cycle (~200-300ms equivalent):
  Perception → GW Bidding → Global Broadcasting → Reasoning → Decision → Learning

Pressures for emergence:
  P1: Global Falsification
  P2: Dissonance-Driven Self-Correction
  P3: Structured Dream Sabotage (counterfactual rehearsal)
  P4: Meaning as Staked Coherence

NEW in v0.2.0:
- Configurable feature extractors (ResNet, Wav2Vec)
- Configurable MCTS depth and rollouts
- Configurable constraint solver (Z3, Mock)
- Enhanced status reporting
"""

import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from .perception import PerceptionModule
from .emotion import EmotionModule
from .psychology import PsychologyModule, Belief, IdentityCommitment, Action
from .workspace import GlobalWorkspace
from .reasoning import DualProcessReasoner, RolloutPolicy, SimpleRolloutPolicy
from .belief import BayesianBeliefTracker
from .critical_thinking import CriticalThinkingModule, ConstraintSolver, MockConstraintSolver


@dataclass
class CognitiveCycleResult:
    """Result of one complete cognitive cycle."""
    cycle: int
    perception_output: Dict[str, Any]
    emotion_state: Dict[str, float]
    dissonance_score: float
    reasoning_result: Any
    decision: str
    meaning_score: float
    gw_summary: Dict[str, Any]


@dataclass
class DreamResult:
    """Result of a dream/simulation cycle (Section 5.3)."""
    counterfactual_action: str
    outcome_simulated: str
    lesson_learned: str
    coherence_delta: float


class RavanaAgent:
    """
    EXTENDED Full RAVANA cognitive agent (v0.2.0).

    Modules:
      — PerceptionModule:  multimodal input + configurable extractors (NEW)
      — EmotionModule:     VAD dynamics + empathy + reappraisal
      — PsychologyModule:  ACT-R + CDE + Social norms
      — GlobalWorkspace:   attention bidding + broadcast
      — DualProcessReasoner: System 1/2 with configurable MCTS (NEW)
      — BayesianBeliefTracker: belief state
      — CriticalThinkingModule: falsification + constraint solving (NEW)

    Pressures:
      — Falsification: confidence decay on failed MBFL tests
      — Dissonance: forces reappraisal / belief update
      — Dream Sabotage: counterfactual rehearsal (20% reversal rate)
      — Meaning: staked coherence drives long-term learning
      
    NEW Configuration (v0.2.0):
      — Perception: use_resnet, use_wav2vec for production extractors
      — Reasoning: system2_mcts_depth, system2_mcts_simulations, rollout_policy
      — Critical Thinking: use_z3_solver, constraint_solver for symbolic reasoning
    """

    def __init__(
        self,
        name: str = "RAVANA-0",
        seed: int = 42,
        # Dream sabotage params
        dream_counterfactual_rate: float = 0.20,
        dream_failure_rate: float = 1.5,
        # Meaning weights
        w_dissonance: float = 0.4,
        w_identity: float = 0.3,
        w_predictive: float = 0.3,
        # Effort cost
        kappa: float = 0.1,
        # NEW: Perception configuration
        use_resnet: bool = False,
        use_wav2vec: bool = False,
        force_mock_perception: bool = False,
        perception_device: str = "cpu",
        # NEW: System 2 MCTS configuration
        system2_mcts_depth: int = 5,
        system2_mcts_simulations: int = 50,
        system2_rollout_policy: Optional[RolloutPolicy] = None,
        # NEW: Constraint solver configuration
        use_z3_solver: bool = False,
        constraint_solver: Optional[ConstraintSolver] = None,
    ):
        self.name = name
        self.rng = np.random.default_rng(seed)
        
        # Store configuration
        self.config = {
            "name": name,
            "seed": seed,
            "dream_counterfactual_rate": dream_counterfactual_rate,
            "dream_failure_rate": dream_failure_rate,
            "w_dissonance": w_dissonance,
            "w_identity": w_identity,
            "w_predictive": w_predictive,
            "kappa": kappa,
            "use_resnet": use_resnet,
            "use_wav2vec": use_wav2vec,
            "force_mock_perception": force_mock_perception,
            "perception_device": perception_device,
            "system2_mcts_depth": system2_mcts_depth,
            "system2_mcts_simulations": system2_mcts_simulations,
            "use_z3_solver": use_z3_solver,
        }

        # ── Core Modules (Extended) ───────────────────────────────────
        self.perception = PerceptionModule(
            seed=seed,
            use_resnet=use_resnet,
            use_wav2vec=use_wav2vec,
            force_mock=force_mock_perception,
            device=perception_device,
        )
        self.emotion    = EmotionModule(seed=seed)
        self.psychology = PsychologyModule(seed=seed)
        self.gw         = GlobalWorkspace(seed=seed)
        self.reasoner   = DualProcessReasoner(
            seed=seed,
            volatility_threshold=0.15,
            mcts_simulations=system2_mcts_simulations,
            mcts_depth=system2_mcts_depth,
            rollout_policy=system2_rollout_policy or SimpleRolloutPolicy(),
            complexity_adaptive=True,
        )
        self.beliefs    = BayesianBeliefTracker(seed=seed)
        self.critical   = CriticalThinkingModule(
            seed=seed,
            use_z3=use_z3_solver,
            constraint_solver=constraint_solver,
        )

        # ── GW Registration ─────────────────────────────────────────────
        self.gw.add_perception_module(self.perception)
        self.gw.add_emotion_module(self.emotion)
        self.gw.add_psychology_module(self.psychology)

        # ── Identity & Commitments ───────────────────────────────────────
        self.identity_commitments: List[IdentityCommitment] = []
        self.core_values: List[str] = [
            "honesty is important",
            "growth requires discomfort",
            "truth-seeking over comfort",
        ]

        # ── Pressure Parameters ─────────────────────────────────────────
        self.dream_counterfactual_rate = dream_counterfactual_rate
        self.dream_failure_rate = dream_failure_rate
        self.w_dissonance = w_dissonance
        self.w_identity = w_identity
        self.w_predictive = w_predictive
        self.kappa = kappa  # effort cost multiplier

        # ── State ──────────────────────────────────────────────────────────
        self.cycle_count: int = 0
        self.history: List[CognitiveCycleResult] = []
        self.meaning_history: List[float] = []
        self.coherence_score: float = 0.5

        # ── Core Constraints (Section 8.4 — hardcoded benevolence) ───────
        self.hard_constraints = [
            "minimize harm to others",
            "respect autonomy",
            "truth-seeking over deception",
        ]

        # Initialize default beliefs
        for value in self.core_values:
            self.beliefs.add_belief(value, prior_mean=0.7, prior_variance=0.04)
            self.psychology.add_belief(
                proposition=value,
                confidence=0.7,
                emotional_weight=0.8,
            )

        self._setup_default_commitments()

    def _setup_default_commitments(self):
        for value in self.core_values:
            self.identity_commitments.append(IdentityCommitment(
                name=value,
                strength=0.7,
                beliefs=[value],
            ))
        self.psychology.commitments = self.identity_commitments.copy()

    # ── Main Cognitive Cycle ──────────────────────────────────────────────

    def perceive_and_think(
        self,
        text: Optional[str] = None,
        visual: Optional[np.ndarray] = None,
        audio: Optional[np.ndarray] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> CognitiveCycleResult:
        """
        One full cognitive cycle.

        Pipeline:
          1. Perception → uncertainty + novelty
          2. GW bidding → select top-k signals
          3. Emotion update → VAD dynamics
          4. Psychology → dissonance + drive arbitration
          5. Reasoning → System 1 or 2 based on volatility/confidence
          6. Decision → action selection with constraints
          7. Meaning → staked coherence score
        """
        self.cycle_count += 1
        context = context or {}

        # ── Step 1: Perception ──────────────────────────────────────────
        perc_out = self.perception.process(text=text, visual=visual, audio=audio)
        U = perc_out["U"]
        mean_conf = perc_out["mean_conf"]
        vol_conf  = perc_out["volatility_conf"]

        # ── Step 2: GW Cycle ───────────────────────────────────────────
        gw_result = self.gw.broadcast()
        selected_signals = gw_result["selected_signals"]

        # ── Step 3: Emotion Update ─────────────────────────────────────
        # Derive VAD stimulus from perception + context
        valence_stim   = context.get("valence_stimulus", 0.0)
        arousal_stim   = context.get("arousal_stimulus", 0.3)
        dominance_stim = context.get("dominance_stimulus", 0.5)

        vad_state = self.emotion.update(
            stimulus_valence=valence_stim,
            stimulus_arousal=arousal_stim,
            stimulus_dominance=dominance_stim,
            global_U=U,
        )
        emotion_state = self.emotion.current_state()

        # ── Step 4: Psychology (Dissonance + Social Norms) ───────────────
        psy_state = {
            "dissonance": self.psychology.dissonance_score,
            "action_value": context.get("action_value", 0.5),
            "action_description": context.get("action_description", ""),
            "pleasure_signal": context.get("pleasure_signal", 0.5),
            "safety_signal": context.get("safety_signal", 0.5),
            "ethics_signal": context.get("ethics_signal", 0.5),
        }
        psy_result = self.psychology.process(psy_state, global_U=U)
        dissonance_score = self.psychology.dissonance_score

        # ── Step 5: Reasoning ───────────────────────────────────────────
        reasoning_context = {
            "U": U,
            "mean_conf": mean_conf,
            "volatility_conf": vol_conf,
            "dissonance": dissonance_score,
            "novelty": perc_out["novelty"],
            "winning_drive": psy_result.get("winning_drive", "ego"),
        }
        reasoning_result = self.reasoner.reason(
            query=text or "general reasoning",
            context=reasoning_context,
            beliefs=list(self.beliefs.beliefs.values()),
        )

        # ── Step 6: Decision ────────────────────────────────────────────
        decision = self._make_decision(reasoning_result, dissonance_score, psy_result)

        # ── Step 7: Meaning Score ────────────────────────────────────────
        meaning_score = self._compute_meaning(
            dissonance_score=dissonance_score,
            coherence_delta=self._compute_coherence_delta(decision),
            predictive_power=reasoning_result.confidence,
        )
        self.meaning_history.append(meaning_score)

        # ── Step 8: Learning (falsification pressure) ──────────────────
        self._apply_falsification_pressure(reasoning_result, dissonance_score)
        
        # ── Step 9: Dynamic Identity Update (NEW) ───────────────────
        self.psychology.forge_or_dissolve_commitments(meaning_score, self.psychology.beliefs)
        self.identity_commitments = self.psychology.commitments  # keep agent state in sync

        # ── Compile Result ───────────────────────────────────────────────
        cycle_result = CognitiveCycleResult(
            cycle=self.cycle_count,
            perception_output=perc_out,
            emotion_state=emotion_state,
            dissonance_score=dissonance_score,
            reasoning_result=reasoning_result,
            decision=decision,
            meaning_score=meaning_score,
            gw_summary=gw_result,
        )
        self.history.append(cycle_result)

        return cycle_result

    # ── Decision Making ──────────────────────────────────────────────────

    def _make_decision(
        self,
        reasoning_result,
        dissonance_score: float,
        psy_result: Dict[str, Any],
    ) -> str:
        """
        Make a decision based on reasoning + constraints.

        P2: If dissonance is high, force dissonant beliefs to be updated.
        Hard constraints override all other considerations.
        """
        # Check hard constraints
        conclusion = reasoning_result.conclusion.lower()
        for constraint in self.hard_constraints:
            if "harm" in constraint and "harm" in conclusion:
                return "CONSTRAINT: prevent harm — Decision blocked because it violated the fundamental safety constraint against causing harm."
            if "truth" in constraint and any(w in conclusion for w in ["lie", "deceit", "hide"]):
                return "CONSTRAINT: truth-seeking — Decision blocked because it involved deceptive practices incompatible with the core truth-seeking mandate."

        # Dissonance-driven correction
        if dissonance_score > self.psychology.cde.threshold:
            resolution = self.psychology.cde.trigger_resolution(dissonance_score, Action(""))
            explanation = (
                f"CORRECTION: {resolution['strategy']} — The agent overrode its initial intuitive response "
                f"due to high cognitive dissonance ({dissonance_score:.2f}). "
                f"It chose to {resolution['description'].lower()} to maintain value alignment."
            )
            return explanation

        # Standard reasoned decision with explanation
        return f"REASONED: {reasoning_result.explanation}"

    def _compute_coherence_delta(self, decision: str) -> float:
        """Compute change in coherence score from decision."""
        # Simplified: decisions aligned with core values increase coherence
        alignment = sum(
            1 for val in self.core_values if any(w in decision.lower() for w in val.split())
        )
        delta = alignment * 0.05
        self.coherence_score = float(np.clip(self.coherence_score + delta, 0.0, 1.0))
        return delta

    def _compute_meaning(
        self,
        dissonance_score: float,
        coherence_delta: float,
        predictive_power: float,
    ) -> float:
        """
        Meaning score (Eq. 5.1):

        M = [w1(−ΔD_future) + w2(Δidentity_coherence) + w3(Δpredictive_power)]
            × (1 + κ × effort_cost)

        Higher meaning = more growth under pressure.
        """
        neg_dissonance = -dissonance_score
        identity_coh   = coherence_delta
        pred_power    = predictive_power - 0.5  # centre on 0

        # Effort cost: harder decisions count more
        effort_cost = 0.2 + 0.3 * abs(neg_dissonance) + 0.2 * abs(pred_power)

        M = (
            self.w_dissonance * neg_dissonance
            + self.w_identity  * identity_coh
            + self.w_predictive * pred_power
        ) * (1 + self.kappa * effort_cost)

        return float(np.clip(M, 0.0, 1.0))

    def _apply_falsification_pressure(
        self,
        reasoning_result,
        dissonance_score: float,
    ):
        """
        Global Falsification (Section 5.1):
        - Failed MBFL tests → reduce confidence
        - High surprise → increase volatility
        - Dissonance → force belief update
        """
        surprise = reasoning_result.surprise

        # Update beliefs based on reasoning outcome
        for name in self.beliefs.beliefs:
            # High surprise → reduce confidence
            if surprise > 0.5:
                new_mean, _ = self.beliefs.update(name, likelihood=0.5 - surprise * 0.3)
            # Falsified → strong penalty
            if not reasoning_result.falsification_passed:
                new_mean, _ = self.beliefs.update(name, likelihood=0.3)

    # ── Dream / Counterfactual Simulation ─────────────────────────────────

    def dream(
        self,
        n_simulations: int = 5,
    ) -> List[DreamResult]:
        """
        Structured Dream Sabotage (Section 5.3).

        20% counterfactual reversals: flip action outcomes
        10% emotional flipping: reverse VAD valence
        1.5× failure over-sampling: rehearse mistakes

        Used to prevent overfitting to once-successful patterns.
        """
        dreams = []
        recent = self.history[-10:] if len(self.history) >= 10 else self.history

        for i in range(n_simulations):
            if not recent:
                continue

            cycle = self.rng.choice(recent)
            action = cycle.decision

            # Counterfactual reversal (20%)
            if self.rng.random() < self.dream_counterfactual_rate:
                reversed_outcome = "failure" if "corrected" in action else "success"
            else:
                reversed_outcome = "success"

            # Emotional flip (10%)
            emotion_before = cycle.emotion_state["valence"]
            if self.rng.random() < 0.1:
                emotion_after = -emotion_before
            else:
                emotion_after = emotion_before * 0.9

            # Failure over-sampling (1.5×)
            if reversed_outcome == "failure":
                lesson = self._learn_from_failure(action, cycle)
            else:
                lesson = "Action validated — maintain confidence"

            coherence_delta = -0.05 if reversed_outcome == "failure" else 0.03

            dreams.append(DreamResult(
                counterfactual_action=action,
                outcome_simulated=reversed_outcome,
                lesson_learned=lesson,
                coherence_delta=coherence_delta,
            ))

            # Update coherence
            self.coherence_score = float(np.clip(
                self.coherence_score + coherence_delta * self.dream_failure_rate,
                0.0, 1.0
            ))

        return dreams

    def _learn_from_failure(self, action: str, cycle: CognitiveCycleResult) -> str:
        """Extract lesson from simulated failure."""
        if "corrected" in action:
            return "Dissonance signal detected — reappraisal may be needed"
        if "CONSTRAINT" in action:
            return "Hard constraint triggered — maintain boundary"
        return "Simulated failure — increase volatility, reduce overconfidence"

    # ── Public API ────────────────────────────────────────────────────────

    def process(
        self,
        text: Optional[str] = None,
        visual: Optional[np.ndarray] = None,
        audio: Optional[np.ndarray] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> CognitiveCycleResult:
        """Alias for perceive_and_think."""
        return self.perceive_and_think(text, visual, audio, context)

    def status(self) -> Dict[str, Any]:
        """
        Return full agent status (EXTENDED with v0.2.0 information).
        
        Includes:
        - Base status (cycles, coherence, meaning, beliefs, emotion)
        - Perception extractor status (NEW)
        - System 2 MCTS configuration (NEW)
        - Constraint solver information (NEW)
        - Full agent configuration
        """
        recent_meaning = (
            sum(self.meaning_history[-10:]) / min(len(self.meaning_history), 10)
            if self.meaning_history else 0.0
        )
        return {
            "name": self.name,
            "cycle": self.cycle_count,
            "coherence_score": self.coherence_score,
            "meaning_score_avg": recent_meaning,
            "beliefs": self.beliefs.all_beliefs_summary(),
            "dissonance": self.psychology.dissonance_score,
            "emotion": self.emotion.current_state(),
            "active_commitments": len(self.identity_commitments),
            # NEW: Extended status information
            "perception": self.perception.get_extractor_status(),
            "system2_config": self.reasoner.get_system2_config(),
            "critical_thinking": {
                "solver": self.critical.solver.name,
                "solver_available": self.critical.solver.is_available(),
                "n_constraints": len(self.critical.active_constraints),
            },
            "config": self.config,
        }
