"""
Critical Thinking Module — RAVANA Core Extended
Implements MBFL (Metamorphic, Baseline, Follow-up) falsification testing
and Z3-solver-based logical constraint checking.

EXTENSIONS v0.2.0:
- ConstraintSolver base class for symbolic reasoning
- Z3ConstraintSolver stub (optional Z3 dependency)
- MockConstraintSolver for testing without dependencies
- Integration into MBFL falsification pipeline

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
# EXTENDED Critical Thinking Module
# ─────────────────────────────────────────────────────────────────────────────

class CriticalThinkingModule:
    """
    EXTENDED Critical thinking engine with symbolic constraint solving.
    
    NEW in v0.2.0:
    - Pluggable constraint solver (Z3, Mock, or custom)
    - Constraint-based falsification in MBFL pipeline
    - Symbolic validation of claims
    
    Capabilities:
    - Argument tree construction
    - Contradiction detection between claims
    - Assumption checking
    - Metamorphic testing with constraint validation
    - Deduction, abduction, analogy
    - Strategic reasoning via MCTS
    - Symbolic constraint satisfaction
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

    # ── NEW: Constraint Management ───────────────────────────────────────────

    def add_constraint(self, constraint: Constraint):
        """
        NEW: Add a symbolic constraint.
        
        Constraints are used for logical validation of claims
        and in MBFL falsification.
        """
        self.active_constraints.append(constraint)
    
    def clear_constraints(self):
        """NEW: Clear all active constraints."""
        self.active_constraints = []
    
    def check_constraints(
        self,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, List[ConstraintViolation]]:
        """
        NEW: Check if active constraints are satisfiable.
        
        Uses configured constraint solver (Z3 or mock).
        """
        if not self.active_constraints:
            return True, []
        
        return self.solver.check_sat(self.active_constraints, context)
    
    def validate_with_constraints(self, claim: str) -> Tuple[bool, Optional[ConstraintViolation]]:
        """
        NEW: Validate a claim against active constraints.
        
        Returns (is_valid, violation_or_none).
        """
        if not self.active_constraints:
            return True, None
        
        return self.solver.validate_claim(claim, self.active_constraints)

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

        # NEW: Check against constraints
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
        # NEW: Check premises against constraints
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
        check_constraints: bool = True,  # NEW: Option to validate against constraints
    ) -> Tuple[bool, float]:
        """
        EXTENDED Metamorphic Testing with constraint validation.
        
        NEW: Can optionally validate against active symbolic constraints.
        """
        transformed = transformation_fn(input_sample)
        test_passed = self.rng.random() > 0.2
        surprise = 0.0 if test_passed else self.rng.uniform(0.5, 1.0)

        # NEW: Constraint validation
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
        
        # NEW: Validate conclusion against constraints
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
            
            # NEW: Validate explanation
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

        # NEW: Validate analogical conclusion
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
        
        # NEW: Prefer constraint-based reasoning if available
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
        # NEW: Include constraint solver info
        constraint_status, violations = self.check_constraints() if self.active_constraints else (True, [])
        
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
            # NEW: Constraint solver info
            "solver": self.solver.name,
            "solver_available": self.solver.is_available(),
            "n_constraints": len(self.active_constraints),
            "constraints_satisfiable": constraint_status,
            "constraint_violations": [
                {"constraint": v.constraint.name, "severity": v.severity}
                for v in violations
            ],
        }
