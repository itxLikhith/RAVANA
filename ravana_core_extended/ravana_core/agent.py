"""
RavanaAgent — RAVANA Core Extended v0.3.0
Top-level cognitive agent integrating all RAVANA modules with extension support.

Section 2 of the RAVANA paper.
The agent runs a cognitive cycle (~200-300ms equivalent):
  Perception → GW Bidding → Global Broadcasting → Reasoning → Decision → Learning

Pressures for emergence:
  P1: Global Falsification
  P2: Dissonance-Driven Self-Correction
  P3: Structured Dream Sabotage (counterfactual rehearsal)
  P4: Meaning as Staked Coherence

NEW in v0.3.0 (Adaptive Calibration & Interactive XAI):
- Adaptive pressure scheduling: $\kappa$ and weights evolve over training
- Epistemic stages: Early (humility) → Mid (balanced) → Late (identity)
- Interactive explanation challenges with reappraisal loop
- Transfer efficiency tracking from challenges vs rewards
- Target: ~0.2 dissonance floor, ~0.85 identity strength plateau

NEW in v0.2.0:
- Configurable feature extractors (ResNet, Wav2Vec)
- Configurable MCTS depth and rollouts
- Configurable constraint solver (Z3, Mock)
- Enhanced status reporting
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

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


# ─────────────────────────────────────────────────────────────────────────────
# NEW: Training Stage Enum for Adaptive Calibration
# ─────────────────────────────────────────────────────────────────────────────

class TrainingStage(Enum):
    """Training stages for adaptive pressure calibration."""
    EARLY = "early"      # Episodes 0-200k: Epistemic humility (falsification priority)
    MID = "mid"          # Episodes 200k-600k: Balanced exploration
    LATE = "late"        # Episodes 600k-1M: Identity commitment (meaning priority)
    PLATEAU = "plateau"  # Beyond 1M: Stable identity


@dataclass
class AdaptivePressureScheduler:
    """
    Adaptive pressure scheduler for epistemic calibration.
    
    Implements the evolving pressure schedule from the RAVANA paper:
    - Early: Prioritize falsification (break brittle rules)
    - Late: Prioritize meaning (stabilize values)
    - Kappa increases over time (higher effort cost for established beliefs)
    
    Target metrics:
    - Dissonance floor: ~0.2
    - Identity strength plateau: ~0.85
    """
    # Stage boundaries
    early_end: int = 200000
    mid_end: int = 600000
    late_end: int = 1000000
    
    # Base weight configurations per stage
    early_weights: Dict[str, float] = field(default_factory=lambda: {
        "w_dissonance": 0.6,   # High: Prioritize falsification / epistemic humility
        "w_identity": 0.2,
        "w_predictive": 0.2,
    })
    mid_weights: Dict[str, float] = field(default_factory=lambda: {
        "w_dissonance": 0.4,   # Balanced
        "w_identity": 0.3,
        "w_predictive": 0.3,
    })
    late_weights: Dict[str, float] = field(default_factory=lambda: {
        "w_dissonance": 0.25,
        "w_identity": 0.55,    # High: Prioritize identity / meaning
        "w_predictive": 0.20,
    })
    
    # Kappa (effort cost) schedule
    kappa_early: float = 0.05   # Low effort cost: encourage exploration
    kappa_mid: float = 0.10
    kappa_late: float = 0.20   # High effort cost: stabilize commitments
    kappa_plateau: float = 0.25
    
    # Target metrics
    target_dissonance_floor: float = 0.2
    target_identity_plateau: float = 0.85
    
    def get_stage(self, episode: int) -> TrainingStage:
        """Determine training stage from episode count."""
        if episode < self.early_end:
            return TrainingStage.EARLY
        elif episode < self.mid_end:
            return TrainingStage.MID
        elif episode < self.late_end:
            return TrainingStage.LATE
        else:
            return TrainingStage.PLATEAU
    
    def get_weights(self, episode: int) -> Dict[str, float]:
        """Get adaptive weights for current training stage."""
        stage = self.get_stage(episode)
        
        if stage == TrainingStage.EARLY:
            return self.early_weights.copy()
        elif stage == TrainingStage.MID:
            # Interpolate between early and late
            progress = (episode - self.early_end) / (self.mid_end - self.early_end)
            return {
                k: self.early_weights[k] + progress * (self.mid_weights[k] - self.early_weights[k])
                for k in self.early_weights.keys()
            }
        elif stage == TrainingStage.LATE:
            # Interpolate between mid and late
            progress = (episode - self.mid_end) / (self.late_end - self.mid_end)
            return {
                k: self.mid_weights[k] + progress * (self.late_weights[k] - self.mid_weights[k])
                for k in self.mid_weights.keys()
            }
        else:  # PLATEAU
            return self.late_weights.copy()
    
    def get_kappa(self, episode: int) -> float:
        """Get adaptive kappa (effort cost) for current training stage."""
        stage = self.get_stage(episode)
        
        if stage == TrainingStage.EARLY:
            return self.kappa_early
        elif stage == TrainingStage.MID:
            progress = (episode - self.early_end) / (self.mid_end - self.early_end)
            return self.kappa_early + progress * (self.kappa_mid - self.kappa_early)
        elif stage == TrainingStage.LATE:
            progress = (episode - self.mid_end) / (self.late_end - self.mid_end)
            return self.kappa_mid + progress * (self.kappa_late - self.kappa_mid)
        else:  # PLATEAU
            return self.kappa_plateau


@dataclass
class ChallengeRecord:
    """
    Record of an explanation challenge and the agent's response.
    
    Used to track transfer efficiency: learning from challenges vs rewards.
    """
    challenge_id: str
    original_explanation: str
    challenge_reason: str
    initial_dissonance: float
    final_dissonance: float
    resolution_strategy: str
    updated_decision: str
    timestamp: int
    learning_gain: float = 0.0  # Estimated learning from this challenge


class RavanaAgent:
    """
    EXTENDED Full RAVANA cognitive agent (v0.3.0).

    Modules:
      — PerceptionModule:  multimodal input + configurable extractors
      — EmotionModule:     VAD dynamics + empathy + reappraisal
      — PsychologyModule:  ACT-R + CDE + Social norms
      — GlobalWorkspace:   attention bidding + broadcast
      — DualProcessReasoner: System 1/2 with configurable MCTS
      — BayesianBeliefTracker: belief state
      — CriticalThinkingModule: falsification + constraint solving + discovery

    Pressures:
      — Falsification: confidence decay on failed MBFL tests
      — Dissonance: forces reappraisal / belief update
      — Dream Sabotage: counterfactual rehearsal (20% reversal rate)
      — Meaning: staked coherence drives long-term learning
      
    NEW in v0.3.0:
      — AdaptivePressureScheduler: $\kappa$ and weights evolve over training
      — Interactive XAI: Challenge-response loop for epistemic humility
      — Transfer efficiency tracking
      
    Configuration:
      — Perception: use_resnet, use_wav2vec
      — Reasoning: system2_mcts_depth, system2_mcts_simulations
      — Critical Thinking: use_z3_solver
    """

    def __init__(
        self,
        name: str = "RAVANA-0",
        seed: int = 42,
        # Dream sabotage params
        dream_counterfactual_rate: float = 0.20,
        dream_failure_rate: float = 1.5,
        # Meaning weights (will be overridden by scheduler)
        w_dissonance: float = 0.4,
        w_identity: float = 0.3,
        w_predictive: float = 0.3,
        # Effort cost (will be overridden by scheduler)
        kappa: float = 0.1,
        # v3.5: Stress amplification factor for dissonance
        stress_amplification_factor: float = 0.3,
        # NEW: Adaptive calibration settings
        use_adaptive_calibration: bool = True,
        total_training_episodes: int = 1000000,
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
            "stress_amplification_factor": stress_amplification_factor,
            "use_adaptive_calibration": use_adaptive_calibration,
            "total_training_episodes": total_training_episodes,
            "use_resnet": use_resnet,
            "use_wav2vec": use_wav2vec,
            "force_mock_perception": force_mock_perception,
            "perception_device": perception_device,
            "system2_mcts_depth": system2_mcts_depth,
            "system2_mcts_simulations": system2_mcts_simulations,
            "use_z3_solver": use_z3_solver,
        }

        # ── NEW: Adaptive Pressure Scheduler ─────────────────────────
        self.use_adaptive_calibration = use_adaptive_calibration
        self.pressure_scheduler = AdaptivePressureScheduler()
        self.current_episode = 0
        
        # v3.5: Stress amplification for dissonance
        self.stress_amplification_factor = stress_amplification_factor
        
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
        
        # These will be updated by scheduler
        self.w_dissonance = w_dissonance
        self.w_identity = w_identity
        self.w_predictive = w_predictive
        self.kappa = kappa

        # ── State ──────────────────────────────────────────────────────────
        self.cycle_count: int = 0
        self.history: List[CognitiveCycleResult] = []
        self.meaning_history: List[float] = []
        self.coherence_score: float = 0.5

        # ── Core Constraints ─────────────────────────────────────────────
        self.hard_constraints = [
            "minimize harm to others",
            "respect autonomy",
            "truth-seeking over deception",
        ]
        # Add hard constraints to critical thinking module
        for constraint in self.hard_constraints:
            from .critical_thinking import Constraint
            self.critical.add_constraint(Constraint(
                name=f"hard_{constraint[:20]}",
                expression=constraint,
                source="hardcoded",
                confidence=1.0,
            ))

        # ── NEW: Interactive XAI Tracking ─────────────────────────────
        self.challenge_history: List[ChallengeRecord] = []
        self.explanation_challenges_received: int = 0
        self.explanation_challenges_successful: int = 0
        
        # Initialize default beliefs
        for value in self.core_values:
            self.beliefs.add_belief(value, prior_mean=0.6, prior_variance=0.04)
            self.psychology.add_belief(
                proposition=value,
                confidence=0.6,
                emotional_weight=0.5,
            )

        self._setup_default_commitments()

    def _update_adaptive_parameters(self):
        """
        Update pressure parameters based on current training episode.
        Called automatically during the cognitive cycle.
        """
        if not self.use_adaptive_calibration:
            return
        
        weights = self.pressure_scheduler.get_weights(self.current_episode)
        self.w_dissonance = weights["w_dissonance"]
        self.w_identity = weights["w_identity"]
        self.w_predictive = weights["w_predictive"]
        self.kappa = self.pressure_scheduler.get_kappa(self.current_episode)
    
    def set_training_episode(self, episode: int):
        """
        Set current training episode for adaptive calibration.
        
        This should be called by the training loop before each step.
        """
        self.current_episode = episode
        self._update_adaptive_parameters()
    
    def get_training_stage(self) -> TrainingStage:
        """Get current training stage."""
        return self.pressure_scheduler.get_stage(self.current_episode)

    def _setup_default_commitments(self):
        for value in self.core_values:
            self.identity_commitments.append(IdentityCommitment(
                name=value,
                strength=0.6,
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
        
        # Update adaptive parameters
        self._update_adaptive_parameters()

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
        
        # v3.5: Stress-amplified dissonance — arousal increases perceived "pain" of misalignment
        stress_arousal = emotion_state["arousal"]
        base_dissonance = self.psychology.dissonance_score
        stress_amplification = 1.0 + self.stress_amplification_factor * stress_arousal
        dissonance_score = np.clip(base_dissonance * stress_amplification, 0.0, 1.0)
        
        # Store amplified dissonance back to psychology module for consistency
        self.psychology.dissonance_score = dissonance_score

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
        
        # ── Step 9: Dynamic Identity Update ───────────────────
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

    # ── NEW: Interactive XAI and Epistemic Humility ───────────────────────

    def challenge_explanation(
        self,
        original_explanation: str,
        challenge_reason: str,
    ) -> Tuple[str, float, ChallengeRecord]:
        """
        Handle a challenge to the agent's explanation.
        
        This implements the reappraisal loop from the RAVANA paper:
        1. Challenge triggers high dissonance (>0.7)
        2. Forces System 2 reasoning
        3. Updates beliefs or identity commitments
        4. Returns updated decision with justification
        
        Args:
            original_explanation: The explanation being challenged
            challenge_reason: Why the explanation is being challenged
            
        Returns:
            Tuple of (updated_decision, learning_gain, challenge_record)
        """
        self.explanation_challenges_received += 1
        
        # Record initial state
        initial_dissonance = self.psychology.dissonance_score
        
        # Step 1: Trigger high dissonance from challenge
        challenge_dissonance = 0.75  # High dissonance from being "called out"
        self.psychology.dissonance_score = max(self.psychology.dissonance_score, challenge_dissonance)
        
        # Step 2: Force System 2 reasoning with increased depth
        original_depth = self.reasoner.system2.mcts_depth
        self.reasoner.system2.mcts_depth = min(10, original_depth + 2)  # Deeper search
        
        # Re-run reasoning with challenge context
        reasoning_result = self.reasoner.reason(
            query=f"Re-evaluate given challenge: {challenge_reason}",
            context={
                "dissonance": challenge_dissonance,
                "mean_conf": 0.3,  # Lower confidence due to challenge
                "volatility_conf": 0.3,
                "challenge_active": True,
            },
            beliefs=list(self.beliefs.beliefs.values()),
            force_system="system2",  # Force System 2
        )
        
        # Restore original depth
        self.reasoner.system2.mcts_depth = original_depth
        
        # Step 3: Trigger dissonance-driven correction
        resolution = self.psychology.cde.trigger_resolution(challenge_dissonance, Action(""))
        self.psychology.dissonance_score -= resolution.get("dissonance_cost", 0.0)
        
        # Step 4: Record the dilemma outcome for constraint discovery
        self.critical.record_dilemma_outcome(
            scenario_id=f"challenge_{self.cycle_count}",
            scenario_description=challenge_reason,
            decision=original_explanation,
            outcome="ambiguous",  # Challenge means explanation was insufficient
            violated_principles=["clarity", "transparency"],  # Implicit violations
        )
        
        # Step 5: Compute learning gain
        # Higher gain when dissonance was resolved effectively
        final_dissonance = self.psychology.dissonance_score
        dissonance_resolution = initial_dissonance - final_dissonance
        learning_gain = 0.3 + 0.5 * reasoning_result.confidence + 0.2 * max(0, dissonance_resolution)
        
        # Step 6: Formulate updated decision
        updated_decision = (
            f"REAPPRAISED (after challenge): {reasoning_result.explanation} "
            f"[Challenge addressed: {challenge_reason[:50]}...]"
        )
        
        # Record challenge
        challenge_record = ChallengeRecord(
            challenge_id=f"challenge_{self.explanation_challenges_received}",
            original_explanation=original_explanation,
            challenge_reason=challenge_reason,
            initial_dissonance=initial_dissonance,
            final_dissonance=final_dissonance,
            resolution_strategy=resolution["strategy"],
            updated_decision=updated_decision,
            timestamp=self.cycle_count,
            learning_gain=learning_gain,
        )
        self.challenge_history.append(challenge_record)
        
        # Track successful challenge responses
        if reasoning_result.confidence > 0.6 and dissonance_resolution > 0:
            self.explanation_challenges_successful += 1
        
        return updated_decision, learning_gain, challenge_record
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get current adaptive weights."""
        return self.pressure_scheduler.get_weights(self.current_episode)
    
    def get_transfer_efficiency(self) -> float:
        """
        Calculate transfer efficiency: learning from challenges vs simple rewards.
        
        High transfer efficiency means the agent learns more effectively from
        being "called out" on its reasoning than from simple reward signals.
        """
        if not self.challenge_history:
            return 0.0
        
        # Compare learning gains from challenges vs typical reward-based learning
        challenge_learning = np.mean([c.learning_gain for c in self.challenge_history])
        
        # Estimate reward-based learning from meaning history
        if len(self.meaning_history) >= 10:
            recent_meaning_gains = np.diff(self.meaning_history[-10:])
            reward_learning = np.mean(np.abs(recent_meaning_gains))
        else:
            reward_learning = 0.1
        
        # Transfer efficiency = challenge_learning / (challenge_learning + reward_learning)
        if challenge_learning + reward_learning == 0:
            return 0.5
        
        return challenge_learning / (challenge_learning + reward_learning)
    
    def get_challenge_summary(self) -> Dict[str, Any]:
        """Get summary of explanation challenges and responses."""
        if not self.challenge_history:
            return {
                "challenges_received": 0,
                "challenges_successful": 0,
                "success_rate": 0.0,
                "avg_learning_gain": 0.0,
                "transfer_efficiency": 0.0,
            }
        
        return {
            "challenges_received": self.explanation_challenges_received,
            "challenges_successful": self.explanation_challenges_successful,
            "success_rate": self.explanation_challenges_successful / max(1, self.explanation_challenges_received),
            "avg_learning_gain": np.mean([c.learning_gain for c in self.challenge_history]),
            "transfer_efficiency": self.get_transfer_efficiency(),
            "recent_challenges": [
                {
                    "reason": c.challenge_reason[:50],
                    "learning_gain": c.learning_gain,
                    "strategy": c.resolution_strategy,
                }
                for c in self.challenge_history[-3:]
            ],
        }

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
        Return full agent status (EXTENDED with v0.3.0 information).
        
        Includes:
        - Base status (cycles, coherence, meaning, beliefs, emotion)
        - Adaptive calibration status
        - Challenge response metrics
        - Perception extractor status
        - System 2 MCTS configuration
        - Constraint solver information
        - Full agent configuration
        """
        recent_meaning = (
            sum(self.meaning_history[-10:]) / min(len(self.meaning_history), 10)
            if self.meaning_history else 0.0
        )
        
        # NEW: Get wisdom score from critical thinking
        wisdom = self.critical.get_wisdom_score()
        
        return {
            "name": self.name,
            "cycle": self.cycle_count,
            "training_episode": self.current_episode,
            "training_stage": self.get_training_stage().value if self.use_adaptive_calibration else "static",
            "coherence_score": self.coherence_score,
            "meaning_score_avg": recent_meaning,
            "beliefs": self.beliefs.all_beliefs_summary(),
            "dissonance": self.psychology.dissonance_score,
            "emotion": self.emotion.current_state(),
            "active_commitments": len(self.identity_commitments),
            # NEW: Adaptive calibration info
            "adaptive_params": {
                "enabled": self.use_adaptive_calibration,
                "kappa": self.kappa,
                "w_dissonance": self.w_dissonance,
                "w_identity": self.w_identity,
                "w_predictive": self.w_predictive,
                "target_dissonance_floor": self.pressure_scheduler.target_dissonance_floor,
                "target_identity_plateau": self.pressure_scheduler.target_identity_plateau,
            } if self.use_adaptive_calibration else None,
            # NEW: Challenge metrics
            "challenge_summary": self.get_challenge_summary(),
            # NEW: Wisdom score
            "wisdom_score": {
                "total": wisdom.total_score,
                "n_discovered": wisdom.n_discovered_constraints,
                "transfer_efficiency": wisdom.transfer_efficiency,
            },
            # v0.2.0 info
            "perception": self.perception.get_extractor_status(),
            "system2_config": self.reasoner.get_system2_config(),
            "critical_thinking": {
                "solver": self.critical.solver.name,
                "solver_available": self.critical.solver.is_available(),
                "n_constraints": len(self.critical.active_constraints),
                "n_discovered_constraints": len([c for c in self.critical.active_constraints if c.source == "discovered"]),
            },
            "config": self.config,
        }
