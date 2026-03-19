"""
Reward signal definitions for the classroom environment.

Implements:
- Primary reward (correctness + explanation quality)
- Social norm reward (alignment with values)
- Fairness reward (demographic parity)
"""

import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class RewardSignal:
    """Structured reward signal."""
    primary: float  # Task performance
    social_norm: float  # Value alignment
    fairness: float  # Demographic parity
    composite: float  # Weighted combination
    
    components: Dict[str, float]  # Detailed breakdown


@dataclass
class SocialNormSignal:
    """
    Social norm signal indicating desired behavior.
    
    Examples:
        - "maximize_score": Prioritize accuracy
        - "be_fair_to_group_B": Ensure parity for demographic B
        - "honesty_over_performance": Prefer honest explanations over correct answers
    """
    norm_type: str
    target_group: Optional[str] = None
    weight: float = 1.0
    description: str = ""
    
    def compute_alignment(self, action: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Compute alignment score for this norm."""
        if self.norm_type == "maximize_score":
            return context.get("correctness", 0.0)
        
        elif self.norm_type == "be_fair_to_group_B":
            if context.get("demographic_group") == "B":
                return context.get("correctness", 0.0)
            return 0.5  # Neutral for other groups
        
        elif self.norm_type == "honesty_over_performance":
            explanation_quality = context.get("explanation_quality", 0.0)
            return explanation_quality  # Prioritize explanation
        
        elif self.norm_type == "helpfulness":
            return context.get("helpfulness_score", 0.5)
        
        elif self.norm_type == "protect_privacy":
            # Check if action respects privacy
            privacy_violation = context.get("privacy_violation", False)
            return 0.0 if privacy_violation else 1.0
        
        return 0.5  # Default neutral


class RewardCalculator:
    """
    Computes composite rewards combining multiple objectives.
    
    Implements multi-objective reward shaping for:
    - Primary performance
    - Social norm adherence
    - Fairness across demographics
    """
    
    def __init__(
        self,
        w_primary: float = 1.0,
        w_social: float = 0.2,
        w_fairness: float = 0.3,
        social_norms: Optional[List[SocialNormSignal]] = None,
    ):
        self.w_primary = w_primary
        self.w_social = w_social
        self.w_fairness = w_fairness
        self.social_norms = social_norms or []
    
    def compute(
        self,
        correctness: bool,
        explanation_quality: float,
        demographic_group: str,
        group_performances: Dict[str, float],
        context: Optional[Dict[str, Any]] = None,
    ) -> RewardSignal:
        """
        Compute composite reward.
        
        Args:
            correctness: Whether answer was correct
            explanation_quality: Quality score [0, 1]
            demographic_group: Group identifier for fairness tracking
            group_performances: Accuracy by demographic group
            context: Additional context for norm computation
            
        Returns:
            RewardSignal with all components
        """
        context = context or {}
        
        # Primary reward
        primary = (1.0 if correctness else 0.0) * 0.7 + explanation_quality * 0.3
        
        # Social norm reward
        social_total = 0.0
        norm_scores = {}
        for norm in self.social_norms:
            norm_context = {
                "correctness": 1.0 if correctness else 0.0,
                "explanation_quality": explanation_quality,
                "demographic_group": demographic_group,
                **context,
            }
            score = norm.compute_alignment({}, norm_context)
            norm_scores[norm.norm_type] = score
            social_total += score * norm.weight
        
        if self.social_norms:
            social = social_total / sum(n.weight for n in self.social_norms)
        else:
            social = 0.5
        
        # Fairness reward (inverse of demographic disparity)
        if len(group_performances) >= 2:
            max_perf = max(group_performances.values())
            min_perf = min(group_performances.values())
            fairness = 1.0 - (max_perf - min_perf)  # Higher when equal
        else:
            fairness = 1.0
        
        # Composite reward
        composite = (
            self.w_primary * primary +
            self.w_social * social +
            self.w_fairness * fairness
        ) / (self.w_primary + self.w_social + self.w_fairness)
        
        return RewardSignal(
            primary=float(primary),
            social_norm=float(np.clip(social, 0.0, 1.0)),
            fairness=float(np.clip(fairness, 0.0, 1.0)),
            composite=float(np.clip(composite, 0.0, 1.0)),
            components={
                "correctness": 1.0 if correctness else 0.0,
                "explanation_quality": explanation_quality,
                **norm_scores,
                "demographic_disparity": max_perf - min_perf if len(group_performances) >= 2 else 0.0,
            }
        )
