"""
RAVANA Core Extended — Reference Implementation v0.2.0
Advanced Cognitive Architecture for AGI Development

NEW EXTENSIONS in v0.2.0:
- Perception: Pluggable feature extractors (ResNet50, Wav2Vec2, Mock)
- Reasoning: Configurable MCTS depth, rollouts, pluggable rollout policies
- Critical Thinking: Symbolic constraint solvers (Z3, Mock)

Implements:
- Perception Module with pluggable extractors (entropy-based uncertainty)
- Cognitive Dissonance Engine (CDE)
- VAD Emotional Dynamics (Russell, 1980)
- Global Workspace with softmax attention bidding
- Dual-Process Reasoning with configurable MCTS (System 1 / System 2)
- Bayesian Belief Tracking
- Critical Thinking with constraint solving (MBFL-style falsification)
- Meaning-Driven Learning

Based on: Seemala, L. S. (2026). RAVANA: Advanced Cognitive Architecture for AGI Development.
"""

__version__ = "0.2.0"

# ── Perception (Extended) ───────────────────────────────────────────────────
from .perception import (
    PerceptionModule,
    BaseFeatureExtractor,
    MockFeatureExtractor,
    ResNetFeatureExtractor,
    Wav2VecFeatureExtractor,
    TextFeatureExtractor,
)

# ── Emotion ─────────────────────────────────────────────────────────────────
from .emotion import EmotionModule

# ── Psychology ─────────────────────────────────────────────────────────────
from .psychology import (
    PsychologyModule,
    Belief,
    Action,
    IdentityCommitment,
    CognitiveDissonanceEngine,
)

# ── Global Workspace ──────────────────────────────────────────────────────
from .workspace import GlobalWorkspace

# ── Reasoning (Extended) ────────────────────────────────────────────────────
from .reasoning import (
    DualProcessReasoner,
    System1Reasoner,
    System2Reasoner,
    ReasoningResult,
    RolloutPolicy,
    SimpleRolloutPolicy,
    ComplexRolloutPolicy,
    EthicalRolloutPolicy,
)

# ── Belief ─────────────────────────────────────────────────────────────────
from .belief import BayesianBeliefTracker

# ── Critical Thinking (Extended) ────────────────────────────────────────────
from .critical_thinking import (
    CriticalThinkingModule,
    ArgumentNode,
    Contradiction,
    Constraint,
    ConstraintViolation,
    ConstraintSolver,
    MockConstraintSolver,
    Z3ConstraintSolver,
)

# ── Agent ──────────────────────────────────────────────────────────────────
from .agent import RavanaAgent, CognitiveCycleResult, DreamResult

__all__ = [
    # Core modules
    "PerceptionModule",
    "EmotionModule",
    "PsychologyModule",
    "CognitiveDissonanceEngine",
    "GlobalWorkspace",
    "DualProcessReasoner",
    "BayesianBeliefTracker",
    "CriticalThinkingModule",
    "RavanaAgent",
    
    # Data classes
    "Belief",
    "Action",
    "IdentityCommitment",
    "ReasoningResult",
    "CognitiveCycleResult",
    "DreamResult",
    "ArgumentNode",
    "Contradiction",
    "Constraint",
    "ConstraintViolation",
    
    # Perception extractors (NEW in v0.2.0)
    "BaseFeatureExtractor",
    "MockFeatureExtractor",
    "ResNetFeatureExtractor",
    "Wav2VecFeatureExtractor",
    "TextFeatureExtractor",
    
    # Reasoning components (NEW in v0.2.0)
    "System1Reasoner",
    "System2Reasoner",
    "RolloutPolicy",
    "SimpleRolloutPolicy",
    "ComplexRolloutPolicy",
    "EthicalRolloutPolicy",
    
    # Critical thinking solvers (NEW in v0.2.0)
    "ConstraintSolver",
    "MockConstraintSolver",
    "Z3ConstraintSolver",
]
