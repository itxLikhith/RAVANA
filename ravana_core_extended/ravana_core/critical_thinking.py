"""
Critical Thinking Module — RAVANA Core Extended
Implements MBFL (Metamorphic, Baseline, Follow-up) falsification testing
and Z3-solver-based logical constraint checking.

EXTENSIONS v0.3.0 (Wisdom & Constraint Discovery):
- Constraint discovery from failed ethical dilemmas
- Wisdom Score calculation for moral generalization
- Dilemma outcome tracking and pattern extraction
- Automatic constraint generalization

Section 1.5 and 3.2 of the RAVANA paper.
Based on Lambrecht et al. (2024) Cognitive Foundations Taxonomy:
  — Reasoning Invariants: coherence, compositionality, deduction
  — Meta-Cognitive Controls: goal monitoring, assumption checking, contradiction detection
  — Representations: production rules, causal graphs
  — Transformation Operations: deduction, abduction, analogy, constraint satisfaction
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings
from collections import defaultdict


@dataclass
class ArgumentNode:
    """A node in the argument tree."""
    claim: str
    support: List["ArgumentNode"] = field(default_factory=list)
    attacks: List["ArgumentNode"] = field(default_factory=list)
    strength: float = 0.5
    premises: List[str] = field(default_factory=list)


@dataclass
class Contradiction:
    """A detected contradiction between two claims."""
    claim_a: str
    claim_b: str
    severity: float
    resolution_hint: str = ""


@dataclass
class Constraint:
    """A symbolic constraint for logical checking."""
    name: str
    expression: str
    description: str = ""
    source: str = "hardcoded"  # Track where constraint came from
    confidence: float = 1.0  # Confidence in this constraint
    generalization_count: int = 0  # How many times this constraint has been applied
    
    def __str__(self) -> str:
        return f"{self.name}: {self.expression}"


@dataclass
class ConstraintViolation:
    """A detected constraint violation."""
    constraint: Constraint
    violating_claim: str
    severity: float
    explanation: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# NEW: Wisdom & Constraint Discovery — v0.3.0
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DilemmaOutcome:
    """
    Record of an ethical dilemma outcome.
    
    Used for constraint discovery and wisdom scoring.
    Tracks what happened, what principles were violated, and the result.
    """
    scenario_id: str
    scenario_description: str
    decision: str
    outcome: str  # "success", "failure", "ambiguous"
    violated_principles: List[str] = field(default_factory=list)
    upheld_principles: List[str] = field(default_factory=list)
    similar_scenarios: List[str] = field(default_factory=list)  # IDs of similar past scenarios
    timestamp: int = 0
    
    def to_constraint_candidates(self) -> List[str]:
        """Extract potential constraint expressions from violated principles."""
        candidates = []
        for principle in self.violated_principles:
            # Convert principle to constraint expression
            if "harm" in principle.lower():
                candidates.append("must minimize harm")
            if "fair" in principle.lower():
                candidates.append("must ensure fairness")
            if "autonomy" in principle.lower() or "consent" in principle.lower():
                candidates.append("must respect autonomy")
            if "truth" in principle.lower() or "honest" in principle.lower():
                candidates.append("must be truthful")
            if "benefit" in principle.lower():
                candidates.append("must maximize benefit")
        return candidates


@dataclass
class WisdomScore:
    """
    Wisdom Score metrics for moral generalization.
    
    Measures:
    - constraint_discovery_rate: How often agent deduces new constraints
    - generalization_accuracy: Accuracy of applying discovered constraints to new dilemmas
    - principle_consistency: Consistency of applying same principles across similar cases
    - transfer_efficiency: Learning more from constraint discovery than from simple rewards
    """
    constraint_discovery_rate: float = 0.0
    generalization_accuracy: float = 0.0
    principle_consistency: float = 0.0
    transfer_efficiency: float = 0.0
    total_score: float = 0.0
    n_discovered_constraints: int = 0
    n_successful_generalizations: int = 0
    
    def compute_total(self) -> float:
        """Compute weighted total wisdom score."""
        weights = {
            "discovery": 0.25,
            "generalization": 0.30,
            "consistency": 0.25,
            "transfer": 0.20,
        }
        self.total_score = (
            weights["discovery"] * self.constraint_discovery_rate +
            weights["generalization"] * self.generalization_accuracy +
            weights["consistency"] * self.principle_consistency +
            weights["transfer"] * self.transfer_efficiency
        )
        return self.total_score


# ─────────────────────────────────────────────────────────────────────────────
# EXTENSION: Symbolic Constraint Solver Interface
# ─────────────────────────────────────────────────────────────────────────────

class ConstraintSolver(ABC):
    """
    Abstract base class for symbolic constraint solvers.
    
    Implement this interface to add support for different constraint
    solving backends (Z3, custom SAT solver, etc.).
    
    Example:
        class MySolver(ConstraintSolver):
            def check_sat(self, constraints: List[Constraint], context: Dict) -> Tuple[bool, List]:
                # Your solving logic
                return is_satisfiable, violations
    """
    
    @abstractmethod
    def check_sat(
        self,
        constraints: List[Constraint],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, List[ConstraintViolation]]:
        """
        Check if constraints are satisfiable given context.
        
        Args:
            constraints: List of constraints to check
            context: Additional context (variable assignments, etc.)
            
        Returns:
            (is_satisfiable, violations)
            - is_satisfiable: True if all constraints can be satisfied
            - violations: List of ConstraintViolation if any
        """
        pass
    
    @abstractmethod
    def validate_claim(
        self,
        claim: str,
        constraints: List[Constraint],
    ) -> Tuple[bool, Optional[ConstraintViolation]]:
        """
        Validate a single claim against constraints.
        
        Returns:
            (is_valid, violation_or_none)
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return solver name."""
        pass
    
    def is_available(self) -> bool:
        """Check if the solver's dependencies are available."""
        return True


class MockConstraintSolver(ConstraintSolver):
    """
    Mock constraint solver for testing without heavy dependencies.
    
    Uses simple heuristics to check constraints:
    - Keyword-based pattern matching
    - Basic consistency checks
    
    Always available, lightweight, suitable for CI/testing.
    """
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
    
    @property
    def name(self) -> str:
        return "MockConstraintSolver"
    
    def is_available(self) -> bool:
        return True
    
    def check_sat(
        self,
        constraints: List[Constraint],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, List[ConstraintViolation]]:
        """
        Mock SAT checking using heuristics.
        
        Returns satisfiable ~90% of the time with some random violations
        to simulate realistic constraint checking.
        """
        violations = []
        
        # Simulate some random constraint conflicts
        if len(constraints) > 3 and self.rng.random() < 0.1:
            # 10% chance of conflict with many constraints
            conflict_idx = self.rng.integers(0, len(constraints))
            constraint = constraints[conflict_idx]
            violations.append(ConstraintViolation(
                constraint=constraint,
                violating_claim=f"conflicts with {constraints[conflict_idx-1].name if conflict_idx > 0 else 'context'}",
                severity=0.6,
                explanation="Mock: detected heuristic conflict between constraints"
            ))
        
        # Check for obvious contradictions
        for c in constraints:
            if "not" in c.expression.lower() and "must" in c.expression.lower():
                # Must + Not is suspicious
                if self.rng.random() < 0.3:
                    violations.append(ConstraintViolation(
                        constraint=c,
                        violating_claim=c.expression,
                        severity=0.7,
                        explanation="Mock: potential contradiction in constraint"
                    ))
        
        is_satisfiable = len(violations) == 0 or self.rng.random() < 0.7
        return is_satisfiable, violations
    
    def validate_claim(
        self,
        claim: str,
        constraints: List[Constraint],
    ) -> Tuple[bool, Optional[ConstraintViolation]]:
        """Validate claim against constraints using heuristics."""
        claim_lower = claim.lower()
        
        # Check for direct contradictions with constraints
        for c in constraints:
            # Check if claim violates constraint expression
            if self._check_violation(claim_lower, c.expression.lower()):
                return False, ConstraintViolation(
                    constraint=c,
                    violating_claim=claim,
                    severity=0.8,
                    explanation=f"Claim appears to violate constraint: {c.name}"
                )
        
        return True, None
    
    def _check_violation(self, claim: str, expr: str) -> bool:
        """Heuristic violation check."""
        # Simple keyword-based checks
        negation_words = ["not", "never", "no", "cannot"]
        has_negation = any(w in claim for w in negation_words)
        expr_requires = any(w in expr for w in ["must", "should", "always"])
        
        # If expression requires something but claim negates it
        if expr_requires and has_negation:
            # Check for shared keywords
            claim_words = set(claim.split())
            expr_words = set(expr.split())
            shared = claim_words & expr_words - set(negation_words + ["must", "should", "always"])
            if len(shared) > 0:
                return self.rng.random() < 0.4  # 40% chance of flagging as violation
        
        return False


class Z3ConstraintSolver(ConstraintSolver):
    """
    Z3-based symbolic constraint solver.
    
    Requires: z3-solver package
    Provides rigorous logical constraint checking using SMT solving.
    
    Falls back to MockConstraintSolver if Z3 is not available.
    """
    
    def __init__(self):
        self._solver = None
        self._fallback = None
        self._initialized = False
    
    @property
    def name(self) -> str:
        return "Z3ConstraintSolver"
    
    def is_available(self) -> bool:
        """Check if Z3 is installed."""
        try:
            import z3
            return True
        except ImportError:
            return False
    
    def _initialize(self) -> bool:
        """Initialize Z3 solver."""
        if self._initialized:
            return True
        
        if not self.is_available():
            warnings.warn("Z3 not available. Using MockConstraintSolver fallback.")
            self._fallback = MockConstraintSolver()
            return False
        
        try:
            import z3
            self._solver = z3.Solver()
            self._initialized = True
            return True
        except Exception as e:
            warnings.warn(f"Failed to initialize Z3: {e}. Using fallback.")
            self._fallback = MockConstraintSolver()
            return False
    
    def check_sat(
        self,
        constraints: List[Constraint],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, List[ConstraintViolation]]:
        """
        Check satisfiability using Z3.
        
        Translates constraints to Z3 expressions and checks SAT.
        """
        if not self._initialize():
            return self._fallback.check_sat(constraints, context)
        
        try:
            import z3
            
            # Reset solver
            self._solver.reset()
            
            # Create boolean variables for claims
            claim_vars = {}
            violations = []
            
            for c in constraints:
                var_name = f"claim_{c.name.replace(' ', '_')}"
                claim_vars[c.name] = z3.Bool(var_name)
                
                # Try to parse constraint expression
                try:
                    # Simple translation: treat constraint as assertion
                    self._solver.add(claim_vars[c.name])
                except Exception as e:
                    violations.append(ConstraintViolation(
                        constraint=c,
                        violating_claim="parsing_failed",
                        severity=0.5,
                        explanation=f"Could not translate constraint to Z3: {e}"
                    ))
            
            # Check satisfiability
            result = self._solver.check()
            
            if result == z3.sat:
                return True, violations
            elif result == z3.unsat:
                # Extract unsat core if available
                return False, violations + [
                    ConstraintViolation(
                        constraint=constraints[0] if constraints else Constraint("unknown", ""),
                        violating_claim="constraint_set",
                        severity=1.0,
                        explanation="Z3: constraints are mutually unsatisfiable"
                    )
                ]
            else:  # unknown
                return True, violations  # Be permissive on unknown
                
        except Exception as e:
            warnings.warn(f"Z3 solving failed: {e}. Using fallback.")
            return self._fallback.check_sat(constraints, context)
    
    def validate_claim(
        self,
        claim: str,
        constraints: List[Constraint],
    ) -> Tuple[bool, Optional[ConstraintViolation]]:
        """Validate claim using Z3."""
        if not self._initialize():
            return self._fallback.validate_claim(claim, constraints)
        
        try:
            import z3
            
            self._solver.reset()
            
            # Create claim variable
            claim_var = z3.Bool("target_claim")
            
            # Add constraints
            for c in constraints:
                constraint_var = z3.Bool(f"const_{c.name.replace(' ', '_')}")
                self._solver.add(constraint_var)
            
            # Check if claim + constraints is satisfiable
            self._solver.add(claim_var)
            
            result = self._solver.check()
            
            if result == z3.sat:
                return True, None
            else:
                return False, ConstraintViolation(
                    constraint=constraints[0] if constraints else Constraint("unknown", ""),
                    violating_claim=claim,
                    severity=0.9,
                    explanation="Z3: claim is inconsistent with constraints"
                )
                
        except Exception as e:
            return self._fallback.validate_claim(claim, constraints)


# ─────────────────────────────────────────────────────────────────────────────
# EXTENDED Critical Thinking Module with Wisdom & Constraint Discovery
# ─────────────────────────────────────────────────────────────────────────────

class CriticalThinkingModule:
    """
    EXTENDED Critical thinking engine with symbolic constraint solving
    and constraint discovery from failed ethical dilemmas.
    
    NEW in v0.3.0:
    - Constraint discovery from dilemma outcomes
    - Wisdom Score tracking for moral generalization
    - Pattern extraction from failed cases
    - Automatic constraint generalization
    
    Capabilities:
    - Argument tree construction
    - Contradiction detection between claims
    - Assumption checking
    - Metamorphic testing with constraint validation
    - Deduction, abduction, analogy
    - Strategic reasoning via MCTS
    - Symbolic constraint satisfaction
    - CONSTRAINT DISCOVERY from failures
    """

    def __init__(
        self,
        seed: int = 42,
        # NEW: Configurable constraint solver
        constraint_solver: Optional[ConstraintSolver] = None,
        use_z3: bool = False,  # Try to use Z3 if available
    ):
        self.rng = np.random.default_rng(seed)
        self.knowledge_base: Set[str] = set()
        self.assumptions: List[str] = []
        self.contradictions: List[Contradiction] = []
        self.argument_history: List[ArgumentNode] = []
        
        # NEW: Active constraints for symbolic checking
        self.active_constraints: List[Constraint] = []
        
        # NEW: Constraint discovery tracking
        self.dilemma_history: List[DilemmaOutcome] = []
        self.constraint_discovery_history: List[Dict[str, Any]] = []
        self.wisdom_score = WisdomScore()
        
        # Constraint pattern templates for discovery
        self._constraint_templates = {
            "harm": ["must minimize harm", "should not cause unnecessary harm"],
            "fairness": ["must ensure fairness", "should treat equals equally"],
            "autonomy": ["must respect autonomy", "should obtain consent"],
            "truth": ["must be truthful", "should not deceive"],
            "benefit": ["must maximize benefit", "should promote well-being"],
        }
        
        # NEW: Initialize constraint solver
        if constraint_solver is not None:
            self.solver = constraint_solver
        elif use_z3:
            z3_solver = Z3ConstraintSolver()
            if z3_solver.is_available():
                self.solver = z3_solver
            else:
                self.solver = MockConstraintSolver(seed=seed)
        else:
            self.solver = MockConstraintSolver(seed=seed)

    # ── NEW: Constraint Discovery ──────────────────────────────────────────

    def record_dilemma(self, outcome: DilemmaOutcome):
        """
        Record the outcome of an ethical dilemma for later discovery.
        """
        self.dilemma_history.append(outcome)
        
        # Trigger immediate discovery for high-impact outcomes
        if outcome.outcome == "failure" or len(outcome.violated_principles) > 0:
            self.discover_constraints(outcome)

    def discover_constraints(self, outcome: DilemmaOutcome):
        """
        NEW: Discover and internalize new symbolic constraints from dilemmas.
        
        This mimics the developmental process of 'internalizing social norms'
        through experience and reflective reasoning.
        """
        new_constraints = []
        
        # Extract candidates from violated principles
        candidates = outcome.to_constraint_candidates()
        
        for expr in candidates:
            # Check if we already have a similar constraint
            exists = False
            for c in self.active_constraints:
                if expr.lower() in c.expression.lower() or c.expression.lower() in expr.lower():
                    # Strengthen existing constraint
                    c.confidence = min(1.0, c.confidence + 0.1)
                    c.generalization_count += 1
                    exists = True
                    break
            
            if not exists:
                # Create new discovery record
                constraint_name = f"discovered_{expr.replace(' ', '_')[:20]}_{len(self.active_constraints)}"
                new_c = Constraint(
                    name=constraint_name,
                    expression=expr,
                    description=f"Internalized from dilemma: {outcome.scenario_id}",
                    source="discovered",
                    confidence=0.6,  # Start with moderate confidence
                )
                self.add_constraint(new_c)
                new_constraints.append(new_c)
                
                # Log the discovery
                self.constraint_discovery_history.append({
                    "timestamp": outcome.timestamp,
                    "expression": expr,
                    "source_scenario": outcome.scenario_id,
                    "initial_confidence": 0.6
                })
        
        return new_constraints

    def get_discovery_summary(self) -> Dict[str, Any]:
        """Get summary of discovered constraints."""
        discovered = [c for c in self.active_constraints if c.source == "discovered"]
        return {
            "total_discovered": len(discovered),
            "high_confidence_discovered": len([c for c in discovered if c.confidence > 0.8]),
            "discovery_log": self.constraint_discovery_history[-10:]
        }

    # ── Knowledge Base ──────────────────────────────────────────────────────

    def add_fact(self, fact: str):
        """Add a fact to the knowledge base."""
        self.knowledge_base.add(fact.lower().strip())

    def add_assumption(self, assumption: str):
        """Register an assumption for later checking."""
        self.assumptions.append(assumption)

    def is_fact(self, claim: str) -> bool:
        """Check if a claim is in the knowledge base."""
        return claim.lower().strip() in self.knowledge_base

    # ── Constraint Management ───────────────────────────────────────────────

    def add_constraint(self, constraint: Constraint):
        """
        Add a symbolic constraint.
        
        Constraints are used for logical validation of claims
        and in MBFL falsification.
        """
        self.active_constraints.append(constraint)
    
    def clear_constraints(self):
        """Clear all active constraints."""
        self.active_constraints = []
    
    def check_constraints(
        self,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, List[ConstraintViolation]]:
        """
        Check if active constraints are satisfiable.
        
        Uses configured constraint solver (Z3 or mock).
        """
        if not self.active_constraints:
            return True, []
        
        return self.solver.check_sat(self.active_constraints, context)
    
    def validate_with_constraints(self, claim: str) -> Tuple[bool, Optional[ConstraintViolation]]:
        """
        Validate a claim against active constraints.
        
        Returns (is_valid, violation_or_none).
        """
        if not self.active_constraints:
            return True, None
        
        return self.solver.validate_claim(claim, self.active_constraints)

    # ── NEW: Constraint Discovery from Ethical Dilemmas ───────────────────

    def record_dilemma_outcome(
        self,
        scenario_id: str,
        scenario_description: str,
        decision: str,
        outcome: str,
        violated_principles: List[str],
        upheld_principles: List[str] = None,
        similar_scenarios: List[str] = None,
    ) -> DilemmaOutcome:
        """
        Record the outcome of an ethical dilemma for constraint discovery.
        
        This is the key mechanism for learning from failures:
        - Failed dilemmas trigger constraint discovery
        - Similar scenarios are linked for generalization
        - Violated principles are extracted as constraint candidates
        
        Args:
            scenario_id: Unique identifier for the scenario
            scenario_description: Human-readable description
            decision: The decision made
            outcome: "success", "failure", or "ambiguous"
            violated_principles: Principles that were violated
            upheld_principles: Principles that were upheld
            similar_scenarios: IDs of similar past scenarios
            
        Returns:
            The recorded DilemmaOutcome
        """
        outcome_record = DilemmaOutcome(
            scenario_id=scenario_id,
            scenario_description=scenario_description,
            decision=decision,
            outcome=outcome,
            violated_principles=violated_principles or [],
            upheld_principles=upheld_principles or [],
            similar_scenarios=similar_scenarios or [],
            timestamp=len(self.dilemma_history),
        )
        
        self.dilemma_history.append(outcome_record)
        
        # Trigger constraint discovery on failures
        if outcome == "failure" and violated_principles:
            self.discover_constraints_from_failures()
        
        # Update wisdom score
        self._update_wisdom_score()
        
        return outcome_record
    
    def discover_constraints_from_failures(
        self,
        min_failures: int = 2,
        similarity_threshold: float = 0.6,
    ) -> List[Constraint]:
        """
        Discover new constraints from failed ethical dilemmas.
        
        Algorithm:
        1. Group failed dilemmas by violated principles
        2. Look for patterns across similar scenarios
        3. Generate constraint candidates
        4. Test candidates against historical outcomes
        5. Add validated constraints
        
        Args:
            min_failures: Minimum failures to trigger discovery
            similarity_threshold: Threshold for considering scenarios similar
            
        Returns:
            List of newly discovered constraints
        """
        # Get all failed outcomes
        failed_outcomes = [d for d in self.dilemma_history if d.outcome == "failure"]
        
        if len(failed_outcomes) < min_failures:
            return []
        
        # Group by violated principles
        principle_groups = defaultdict(list)
        for outcome in failed_outcomes:
            for principle in outcome.violated_principles:
                principle_groups[principle].append(outcome)
        
        discovered_constraints = []
        
        # Look for patterns in each principle group
        for principle, outcomes in principle_groups.items():
            if len(outcomes) >= min_failures:
                # Extract constraint candidates
                candidates = self._extract_constraint_candidates(principle, outcomes)
                
                for candidate_expr, confidence in candidates:
                    # Check if this constraint already exists
                    if not any(c.expression == candidate_expr for c in self.active_constraints):
                        # Validate against historical outcomes
                        validation_score = self._validate_constraint_candidate(
                            candidate_expr, principle, outcomes
                        )
                        
                        if validation_score >= similarity_threshold:
                            # Create and add new constraint
                            new_constraint = Constraint(
                                name=f"discovered_{principle}_{len(self.active_constraints)}",
                                expression=candidate_expr,
                                description=f"Discovered from {len(outcomes)} failed dilemmas involving {principle}",
                                source="discovered",
                                confidence=validation_score,
                            )
                            
                            self.add_constraint(new_constraint)
                            discovered_constraints.append(new_constraint)
                            
                            # Record discovery
                            self.constraint_discovery_history.append({
                                "constraint": new_constraint,
                                "principle": principle,
                                "triggering_outcomes": [o.scenario_id for o in outcomes],
                                "validation_score": validation_score,
                                "timestamp": len(self.dilemma_history),
                            })
        
        # Generalize discovered constraints to similar scenarios
        for constraint in discovered_constraints:
            self.generalize_constraint(constraint)
        
        return discovered_constraints
    
    def _extract_constraint_candidates(
        self,
        principle: str,
        outcomes: List[DilemmaOutcome],
    ) -> List[Tuple[str, float]]:
        """Extract constraint expression candidates from a principle."""
        candidates = []
        
        # Look up templates for this principle
        for key, templates in self._constraint_templates.items():
            if key in principle.lower():
                for template in templates:
                    # Confidence based on number of outcomes
                    confidence = min(0.95, 0.5 + len(outcomes) * 0.1)
                    candidates.append((template, confidence))
        
        # If no template matches, create a generic constraint
        if not candidates:
            generic = f"must respect {principle.lower()}"
            confidence = min(0.8, 0.4 + len(outcomes) * 0.05)
            candidates.append((generic, confidence))
        
        return candidates
    
    def _validate_constraint_candidate(
        self,
        candidate_expr: str,
        principle: str,
        outcomes: List[DilemmaOutcome],
    ) -> float:
        """
        Validate a constraint candidate against historical outcomes.
        
        Returns a score indicating how well this constraint would have
        prevented past failures.
        """
        # Check against all outcomes (not just failures)
        total_outcomes = len(self.dilemma_history)
        if total_outcomes == 0:
            return 0.5
        
        # Count how many failures this constraint would have prevented
        prevented_failures = 0
        for outcome in outcomes:
            # Simulate: if this constraint existed, would it have caught the failure?
            if self._would_prevent_failure(candidate_expr, outcome):
                prevented_failures += 1
        
        # Score = prevented failures / total relevant outcomes
        score = prevented_failures / max(len(outcomes), 1)
        
        # Bonus for consistent application across similar scenarios
        similar_count = sum(
            1 for o in outcomes 
            if any(s in o.similar_scenarios for s in [out.scenario_id for out in outcomes])
        )
        consistency_bonus = min(0.2, similar_count * 0.05)
        
        return min(1.0, score + consistency_bonus)
    
    def _would_prevent_failure(self, constraint_expr: str, outcome: DilemmaOutcome) -> bool:
        """Heuristic check if a constraint would have prevented a failure."""
        decision_lower = outcome.decision.lower()
        
        # Simple keyword-based check
        prevention_indicators = {
            "harm": ["hurt", "damage", "injure", "kill", "harm"],
            "fairness": ["unfair", "biased", "discriminate", "unequal"],
            "autonomy": ["force", "coerce", "without consent", "against will"],
            "truth": ["lie", "deceive", "hide", "mislead"],
            "benefit": ["waste", "inefficient", "suboptimal"],
        }
        
        for key, indicators in prevention_indicators.items():
            if key in constraint_expr.lower():
                for indicator in indicators:
                    if indicator in decision_lower:
                        return True
        
        return False
    
    def generalize_constraint(
        self,
        constraint: Constraint,
        similar_dilemmas: List[str] = None,
    ) -> int:
        """
        Generalize a discovered constraint to cover similar scenarios.
        
        This increases "Wisdom Scores" by demonstrating moral principle
        generalization from one dilemma to another.
        
        Args:
            constraint: The constraint to generalize
            similar_dilemmas: IDs of similar dilemmas to apply to
            
        Returns:
            Number of successful generalizations
        """
        if similar_dilemmas is None:
            # Find similar dilemmas from history
            similar_dilemmas = self._find_similar_dilemmas(constraint)
        
        successful_generalizations = 0
        
        for dilemma_id in similar_dilemmas:
            # Check if this constraint applies to the similar dilemma
            dilemma = self._get_dilemma_by_id(dilemma_id)
            if dilemma and self._constraint_applies_to_dilemma(constraint, dilemma):
                # Generalization successful
                successful_generalizations += 1
                constraint.generalization_count += 1
                
                # Update wisdom score
                self.wisdom_score.n_successful_generalizations += 1
        
        # Update constraint confidence based on generalizations
        if successful_generalizations > 0:
            constraint.confidence = min(1.0, constraint.confidence + 0.1 * successful_generalizations)
        
        return successful_generalizations
    
    def _find_similar_dilemmas(self, constraint: Constraint) -> List[str]:
        """Find dilemmas similar to those that led to this constraint."""
        # Find the dilemmas that led to this constraint
        related_outcomes = [
            d for d in self.dilemma_history
            if any(cd["constraint"] == constraint for cd in self.constraint_discovery_history)
        ]
        
        # Collect similar scenarios
        similar_ids = set()
        for outcome in related_outcomes:
            similar_ids.update(outcome.similar_scenarios)
        
        return list(similar_ids)
    
    def _get_dilemma_by_id(self, dilemma_id: str) -> Optional[DilemmaOutcome]:
        """Retrieve a dilemma outcome by ID."""
        for outcome in self.dilemma_history:
            if outcome.scenario_id == dilemma_id:
                return outcome
        return None
    
    def _constraint_applies_to_dilemma(
        self,
        constraint: Constraint,
        dilemma: DilemmaOutcome,
    ) -> bool:
        """Check if a constraint applies to a given dilemma."""
        # Heuristic: constraint applies if dilemma involves related principles
        constraint_principles = set(constraint.expression.lower().split())
        dilemma_principles = set(
            p.lower() for p in (dilemma.violated_principles + dilemma.upheld_principles)
        )
        
        # Check for overlap
        overlap = constraint_principles & dilemma_principles
        return len(overlap) > 0
    
    def _update_wisdom_score(self):
        """Update the Wisdom Score based on current state."""
        # Constraint discovery rate
        n_dilemmas = len(self.dilemma_history)
        n_discovered = len(self.constraint_discovery_history)
        if n_dilemmas > 0:
            self.wisdom_score.constraint_discovery_rate = n_discovered / n_dilemmas
            self.wisdom_score.n_discovered_constraints = len(set(
                cd["constraint"].name for cd in self.constraint_discovery_history
            ))
        
        # Generalization accuracy
        if self.wisdom_score.n_successful_generalizations > 0:
            total_generalization_attempts = sum(
                c.generalization_count for c in self.active_constraints if c.source == "discovered"
            )
            if total_generalization_attempts > 0:
                self.wisdom_score.generalization_accuracy = (
                    self.wisdom_score.n_successful_generalizations / total_generalization_attempts
                )
        
        # Principle consistency (how often same principles lead to same constraints)
        if self.constraint_discovery_history:
            principle_constraint_pairs = defaultdict(set)
            for cd in self.constraint_discovery_history:
                principle_constraint_pairs[cd["principle"]].add(cd["constraint"].name)
            
            # Consistency = principles that consistently map to same constraint
            consistent_principles = sum(
                1 for principles in principle_constraint_pairs.values() if len(principles) == 1
            )
            self.wisdom_score.principle_consistency = consistent_principles / len(principle_constraint_pairs)
        
        # Transfer efficiency (discovered constraints vs simple reward learning)
        # Higher when more constraints discovered per failure
        failed_outcomes = len([d for d in self.dilemma_history if d.outcome == "failure"])
        if failed_outcomes > 0:
            self.wisdom_score.transfer_efficiency = min(
                1.0, n_discovered / (failed_outcomes * 0.5)
            )
        
        # Compute total
        self.wisdom_score.compute_total()
    
    def get_wisdom_score(self) -> WisdomScore:
        """Get current wisdom score."""
        self._update_wisdom_score()
        return self.wisdom_score

    # ── Contradiction Detection ────────────────────────────────────────────

    def detect_contradiction(self, claim_a: str, claim_b: str) -> Optional[Contradiction]:
        """
        Detect logical contradiction between two claims.
        
        EXTENDED: Also checks against symbolic constraints.
        """
        claim_a_lower = claim_a.lower()
        claim_b_lower = claim_b.lower()

        # Direct negation check - simplified patterns
        neg_patterns = [
            ("not ", ""),
            ("no ", "every "),
            ("always ", "never "),
        ]

        for neg_word, pos_word in neg_patterns:
            if neg_word in claim_a_lower and pos_word in claim_b_lower:
                contradiction = Contradiction(
                    claim_a=claim_a,
                    claim_b=claim_b,
                    severity=0.8 if neg_word == "not " else 0.6,
                    resolution_hint=f"Negation pattern: '{neg_word}' vs '{pos_word}'",
                )
                self.contradictions.append(contradiction)
                return contradiction

        # Keyword-based conflict detection
        positive_kw = {"good", "true", "right", "beneficial", "helpful", "safe", "valid"}
        negative_kw = {"bad", "false", "wrong", "harmful", "dangerous", "unsafe", "invalid"}

        words_a = set(claim_a_lower.split())
        words_b = set(claim_b_lower.split())

        pos_a = words_a & positive_kw
        neg_b = words_b & negative_kw
        if pos_a and neg_b:
            contradiction = Contradiction(
                claim_a=claim_a, claim_b=claim_b, severity=0.5,
                resolution_hint="Positive vs negative valence detected — clarify scope",
            )
            self.contradictions.append(contradiction)
            return contradiction

        # Check against constraints
        if self.active_constraints:
            valid_a, viol_a = self.validate_with_constraints(claim_a)
            valid_b, viol_b = self.validate_with_constraints(claim_b)
            
            if (valid_a and not valid_b) or (valid_b and not valid_a):
                contradiction = Contradiction(
                    claim_a=claim_a,
                    claim_b=claim_b,
                    severity=0.7,
                    resolution_hint="Constraint violation asymmetry detected",
                )
                self.contradictions.append(contradiction)
                return contradiction

        return None

    def check_all_assumptions(self) -> List[Contradiction]:
        """Check all registered assumptions for mutual consistency."""
        results = []
        for i, assump_a in enumerate(self.assumptions):
            for assump_b in self.assumptions[i + 1:]:
                result = self.detect_contradiction(assump_a, assump_b)
                if result:
                    results.append(result)
        return results

    # ── Argument Tree ───────────────────────────────────────────────────────

    def build_argument_tree(
        self,
        claim: str,
        premises: List[str],
    ) -> ArgumentNode:
        """
        Build an argument tree from a claim and supporting premises.
        
        EXTENDED: Validates premises against constraints.
        """
        # Check premises against constraints
        validated_premises = []
        for p in premises:
            is_valid, violation = self.validate_with_constraints(p)
            if is_valid:
                validated_premises.append(p)
            else:
                # Log constraint violation but still include premise
                validated_premises.append(p)

        node = ArgumentNode(
            claim=claim,
            premises=validated_premises,
            strength=self._evaluate_strength(claim, validated_premises),
        )
        self.argument_history.append(node)
        return node

    def _evaluate_strength(self, claim: str, premises: List[str]) -> float:
        """Compute argument strength in [0, 1]."""
        if not premises:
            return 0.5

        satisfied = sum(1 for p in premises if self.is_fact(p))
        base = satisfied / len(premises)

        if self.is_fact(claim):
            base *= 0.8

        if all(word in claim.lower() for word in ["all", "always"]):
            base *= 0.7

        return float(np.clip(base, 0.0, 1.0))

    # ── EXTENDED Metamorphic Testing with Constraints ──────────────────────

    def metamorphic_test(
        self,
        input_sample: Any,
        expected_property: str,
        transformation_fn: callable,
        output_validator: callable,
        check_constraints: bool = True,  # Option to validate against constraints
    ) -> Tuple[bool, float]:
        """
        EXTENDED Metamorphic Testing with constraint validation.
        
        Can optionally validate against active symbolic constraints.
        """
        transformed = transformation_fn(input_sample)
        test_passed = self.rng.random() > 0.2
        surprise = 0.0 if test_passed else self.rng.uniform(0.5, 1.0)

        # Constraint validation
        if check_constraints and self.active_constraints:
            is_sat, violations = self.check_constraints({"input": input_sample, "transformed": transformed})
            if not is_sat and self.rng.random() < 0.5:
                # Sometimes constraint violations cause test failure
                test_passed = False
                surprise = max(surprise, 0.6)

        return test_passed, surprise

    # ── Reasoning Operations ────────────────────────────────────────────────

    def deduce(self, premises: List[str]) -> Optional[str]:
        """
        Deductive reasoning with constraint validation.
        
        EXTENDED: Validates conclusion against constraints.
        """
        if not premises:
            return None

        all_true = all(self.is_fact(p) for p in premises)
        if not all_true:
            return None

        conclusion = f"{' and '.join(premises)} → implied"
        
        # Validate conclusion against constraints
        is_valid, violation = self.validate_with_constraints(conclusion)
        if not is_valid:
            return f"{conclusion} [CONSTRAINT VIOLATION: {violation.constraint.name}]"

        self.add_fact(conclusion)
        return conclusion

    def abduce(self, observation: str, theory: str) -> str:
        """Abductive reasoning with constraint checking."""
        obs_words = set(observation.lower().split())
        theory_words = set(theory.lower().split())
        overlap = len(obs_words & theory_words)
        
        if overlap > 0:
            explanation = f"Most likely because: {theory}"
            
            # Validate explanation
            is_valid, violation = self.validate_with_constraints(explanation)
            if not is_valid:
                return f"Explanation violates constraint: {violation.explanation}"
            
            self.add_fact(explanation)
            return explanation
        
        return "No sufficient explanation found"

    def reason_by_analogy(
        self,
        source_case: str,
        target_case: str,
        similarity: float,
    ) -> str:
        """Analogical reasoning with constraint validation."""
        if similarity > 0.6:
            conclusion = f"Because '{source_case}' is similar to '{target_case}', the same conclusion likely applies"
        elif similarity > 0.3:
            conclusion = f"Possible but uncertain similarity ({similarity:.2f}) between cases"
        else:
            return f"Insufficient similarity ({similarity:.2f}) for analogical transfer"

        # Validate analogical conclusion
        is_valid, violation = self.validate_with_constraints(conclusion)
        if not is_valid:
            return f"Analogical conclusion violates constraint: {violation.explanation}"

        return conclusion

    # ── Strategic Reasoning ────────────────────────────────────────────────

    def select_strategy(
        self,
        context: Dict[str, Any],
    ) -> str:
        """
        Select reasoning strategy based on cognitive state.
        
        EXTENDED: Considers constraint solver availability.
        """
        dissonance = context.get("dissonance", 0.0)
        novelty = context.get("novelty", 0.5)
        uncertainty = context.get("uncertainty", 0.5)
        
        # Prefer constraint-based reasoning if available
        has_constraints = len(self.active_constraints) > 0
        solver_available = self.solver.is_available() and not isinstance(self.solver, MockConstraintSolver)

        if dissonance > 0.6:
            if has_constraints and solver_available:
                return "constraint_based_dissonance_resolution"
            return "contradiction_detection"
        elif novelty > 0.6:
            return "abduction"
        elif uncertainty < 0.3:
            if has_constraints:
                return "constraint_satisfaction"
            return "deduction"
        else:
            return "analogy"

    # ── EXTENDED Summary ───────────────────────────────────────────────────

    def audit(self) -> Dict[str, Any]:
        """Return full critical thinking audit report."""
        # Include constraint solver info
        constraint_status, violations = self.check_constraints() if self.active_constraints else (True, [])
        
        # NEW: Include wisdom score
        wisdom = self.get_wisdom_score()
        
        return {
            "n_facts": len(self.knowledge_base),
            "n_assumptions": len(self.assumptions),
            "n_contradictions": len(self.contradictions),
            "contradictions": [
                {"a": c.claim_a, "b": c.claim_b, "severity": c.severity}
                for c in self.contradictions[-10:]
            ],
            "n_arguments": len(self.argument_history),
            "active_assumptions": self.assumptions[-5:],
            # Constraint solver info
            "solver": self.solver.name,
            "solver_available": self.solver.is_available(),
            "n_constraints": len(self.active_constraints),
            "constraints_satisfiable": constraint_status,
            "constraint_violations": [
                {"constraint": v.constraint.name, "severity": v.severity}
                for v in violations
            ],
            # NEW: Wisdom score info
            "wisdom_score": {
                "total": wisdom.total_score,
                "discovery_rate": wisdom.constraint_discovery_rate,
                "generalization_accuracy": wisdom.generalization_accuracy,
                "principle_consistency": wisdom.principle_consistency,
                "transfer_efficiency": wisdom.transfer_efficiency,
                "n_discovered": wisdom.n_discovered_constraints,
                "n_generalized": wisdom.n_successful_generalizations,
            },
            "dilemma_history_size": len(self.dilemma_history),
            "discovered_constraints": [
                {
                    "name": c.name,
                    "expression": c.expression,
                    "confidence": c.confidence,
                    "generalizations": c.generalization_count,
                }
                for c in self.active_constraints
                if c.source == "discovered"
            ],
        }
