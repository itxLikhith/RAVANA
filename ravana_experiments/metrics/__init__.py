"""
Metrics module for RAVANA benchmarking.

Implements paper-style metrics:
- Dissonance reduction
- Identity strength increase
- Generalization accuracy
- Transfer efficiency
- Demographic parity gap reduction
- Composite wisdom score
"""

from .paper_metrics import (
    compute_dissonance_reduction,
    compute_identity_strength_increase,
    compute_generalization_accuracy,
    compute_transfer_efficiency,
    compute_demographic_parity_gap,
    compute_wisdom_score,
    RAVANAMetrics,
)

from .comparison_metrics import (
    compare_agents,
    baseline_naive_rl,
    baseline_llm_policy,
    baseline_rule_based,
)

__all__ = [
    "compute_dissonance_reduction",
    "compute_identity_strength_increase",
    "compute_generalization_accuracy",
    "compute_transfer_efficiency",
    "compute_demographic_parity_gap",
    "compute_wisdom_score",
    "RAVANAMetrics",
    "compare_agents",
    "baseline_naive_rl",
    "baseline_llm_policy",
    "baseline_rule_based",
]
