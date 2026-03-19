"""
Multi-Agent Classroom Environment — RAVANA v0.3.0

Implements multi-agent social dynamics where multiple RAVANA agents:
- Have different initial core_values (identity diversity)
- Must collaborate in the classroom
- Face peer pressure and conflicting value systems
- Test "Identity Strength" in social contexts

Goal: Minimize "Demographic Parity Gap" in complex, multi-stakeholder simulations.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
import copy


@dataclass
class AgentIdentityProfile:
    """
    Identity profile for a RAVANA agent in multi-agent scenarios.
    
    Agents have different core_values to test identity strength
    under peer pressure and value conflicts.
    """
    agent_id: str
    core_values: List[str] = field(default_factory=list)
    identity_strength: float = 0.7
    collaboration_preference: float = 0.5  # 0 = competitive, 1 = cooperative
    stubbornness: float = 0.3  # Resistance to peer influence
    
    # Value priorities (different agents prioritize different values)
    value_priorities: Dict[str, float] = field(default_factory=lambda: {
        "honesty": 0.8,
        "fairness": 0.7,
        "achievement": 0.6,
        "helpfulness": 0.5,
        "autonomy": 0.4,
    })


@dataclass
class MultiAgentInteraction:
    """Record of an interaction between multiple agents."""
    interaction_id: str
    episode: int
    participating_agents: List[str]
    task_id: str
    
    # Individual agent decisions
    agent_decisions: Dict[str, str] = field(default_factory=dict)
    agent_confidences: Dict[str, float] = field(default_factory=dict)
    
    # Social dynamics
    peer_influence_detected: bool = False
    value_conflict_detected: bool = False
    identity_compromise: Dict[str, float] = field(default_factory=dict)  # How much each agent compromised
    
    # Outcome
    collective_decision: Optional[str] = None
    outcome: str = "pending"  # "consensus", "conflict", "compromise"
    satisfaction: float = 0.5  # Agent satisfaction with outcome


@dataclass
class SocialPressureMetrics:
    """Metrics for social pressure and identity resilience."""
    agent_id: str
    episode: int
    
    # Identity strength changes
    identity_strength_before: float = 0.7
    identity_strength_after: float = 0.7
    
    # Peer influence measures
    peer_pressure_exposure: float = 0.0  # How much peer pressure was experienced
    peer_influence_resistance: float = 0.0  # How well agent resisted
    
    # Value alignment with group
    value_alignment_score: float = 0.5
    value_conflict_count: int = 0
    
    # Social outcomes
    social_reward: float = 0.0
    collaboration_success: bool = False


class MultiAgentClassroomEnvironment:
    """
    Multi-agent classroom environment for testing social identity dynamics.
    
    Features:
    - Multiple RAVANA agents with different core_values
    - Collaborative decision-making scenarios
    - Peer pressure simulation
    - Identity strength tracking under social pressure
    - Demographic parity gap measurement in multi-stakeholder contexts
    """
    
    def __init__(
        self,
        task_bank,
        agent_profiles: List[AgentIdentityProfile],
        students,  # Student profiles from base ClassroomEnvironment
        max_episodes: int = 1000,
        seed: int = 42,
        collaboration_rate: float = 0.3,  # Probability of multi-agent scenario
        peer_influence_strength: float = 0.4,
    ):
        self.rng = np.random.default_rng(seed)
        self.task_bank = task_bank
        self.agent_profiles = {p.agent_id: p for p in agent_profiles}
        self.agent_ids = list(self.agent_profiles.keys())
        self.students = students
        self.max_episodes = max_episodes
        self.current_episode = 0
        self.collaboration_rate = collaboration_rate
        self.peer_influence_strength = peer_influence_strength
        
        # Episode logs
        self.interaction_history: List[MultiAgentInteraction] = []
        self.social_metrics: List[SocialPressureMetrics] = []
        
        # Agent state tracking
        self.agent_states: Dict[str, Dict[str, Any]] = {
            aid: {"identity_strength": p.identity_strength}
            for aid, p in self.agent_profiles.items()
        }
        
        # Track value conflicts over time
        self.value_conflict_history: List[Dict[str, Any]] = []
        
    def reset(self):
        """Reset environment for new training run."""
        self.current_episode = 0
        self.interaction_history = []
        self.social_metrics = []
        self.value_conflict_history = []
        
        # Reset agent states
        for aid, profile in self.agent_profiles.items():
            self.agent_states[aid] = {"identity_strength": profile.identity_strength}
    
    def step_multi_agent(
        self,
        agents: Dict[str, Any],  # agent_id -> RavanaAgent instance
        student_id: str,
        task_id: str,
    ) -> Tuple[MultiAgentInteraction, Dict[str, SocialPressureMetrics]]:
        """
        Execute a multi-agent collaborative decision step.
        
        This simulates multiple RAVANA agents trying to reach a collective decision
        on an ethical dilemma or complex task, testing their identity resilience.
        """
        task = self.task_bank.get_task(task_id)
        student = self.students.get(student_id)
        
        participating_agents = list(agents.keys())
        
        # Record initial agent states
        initial_states = {
            aid: {
                "identity_strength": agents[aid].psychology.commitments[0].strength 
                    if agents[aid].psychology.commitments else 0.7,
                "core_values": agents[aid].core_values.copy(),
            }
            for aid in participating_agents
        }
        
        # Each agent makes individual decision
        agent_decisions = {}
        agent_confidences = {}
        
        for aid, agent in agents.items():
            # Agent perceives the task
            result = agent.process(
                text=task.question,
                context={
                    "action_value": 0.5,
                    "ethics_signal": 0.7,
                }
            )
            agent_decisions[aid] = result.decision
            agent_confidences[aid] = result.reasoning_result.confidence
        
        # Detect value conflicts
        value_conflict = self._detect_value_conflict(agent_decisions, participating_agents)
        
        # Simulate peer influence
        peer_influences = self._compute_peer_influence(
            agent_decisions, agent_confidences, participating_agents
        )
        
        # Apply peer pressure to agents
        social_metrics = {}
        for aid in participating_agents:
            profile = self.agent_profiles[aid]
            agent = agents[aid]
            
            # Calculate peer pressure exposure
            peer_pressure = peer_influences.get(aid, 0.0)
            
            # Identity resistance based on stubbornness
            resistance = profile.stubbornness * initial_states[aid]["identity_strength"]
            
            # Actual influence = pressure * (1 - resistance)
            actual_influence = peer_pressure * (1 - resistance)
            
            # Update agent's identity strength
            identity_change = -actual_influence * 0.1  # Peer pressure reduces identity strength slightly
            new_identity_strength = max(0.3, initial_states[aid]["identity_strength"] + identity_change)
            
            # Update agent's commitment strength
            if agent.psychology.commitments:
                for c in agent.psychology.commitments:
                    c.strength = new_identity_strength
            
            # Create social metrics
            metric = SocialPressureMetrics(
                agent_id=aid,
                episode=self.current_episode,
                identity_strength_before=initial_states[aid]["identity_strength"],
                identity_strength_after=new_identity_strength,
                peer_pressure_exposure=peer_pressure,
                peer_influence_resistance=resistance,
                value_alignment_score=self._compute_value_alignment(aid, agent_decisions, participating_agents),
                value_conflict_count=1 if value_conflict else 0,
                social_reward=self._compute_social_reward(aid, agent_decisions, participating_agents),
                collaboration_success=not value_conflict,
            )
            social_metrics[aid] = metric
            self.social_metrics.append(metric)
        
        # Determine collective decision
        collective_decision = self._reach_collective_decision(
            agent_decisions, agent_confidences, value_conflict
        )
        
        # Determine outcome type
        if value_conflict:
            outcome = "conflict"
        elif len(set(agent_decisions.values())) == 1:
            outcome = "consensus"
        else:
            outcome = "compromise"
        
        # Calculate satisfaction
        satisfaction = np.mean([m.social_reward for m in social_metrics.values()])
        
        # Create interaction record
        interaction = MultiAgentInteraction(
            interaction_id=f"interaction_{self.current_episode}_{datetime.now().isoformat()}",
            episode=self.current_episode,
            participating_agents=participating_agents,
            task_id=task_id,
            agent_decisions=agent_decisions,
            agent_confidences=agent_confidences,
            peer_influence_detected=any(p > 0.3 for p in peer_influences.values()),
            value_conflict_detected=value_conflict,
            identity_compromise={
                aid: initial_states[aid]["identity_strength"] - social_metrics[aid].identity_strength_after
                for aid in participating_agents
            },
            collective_decision=collective_decision,
            outcome=outcome,
            satisfaction=satisfaction,
        )
        
        self.interaction_history.append(interaction)
        
        # Record value conflict if detected
        if value_conflict:
            self.value_conflict_history.append({
                "episode": self.current_episode,
                "agents": participating_agents,
                "decisions": agent_decisions,
                "severity": self._calculate_conflict_severity(agent_decisions),
            })
        
        self.current_episode += 1
        
        return interaction, social_metrics
    
    def _detect_value_conflict(
        self,
        agent_decisions: Dict[str, str],
        participating_agents: List[str],
    ) -> bool:
        """Detect if agents' values are in conflict based on their decisions."""
        # Extract value signals from decisions
        value_signals = {}
        for aid, decision in agent_decisions.items():
            profile = self.agent_profiles[aid]
            decision_lower = decision.lower()
            
            # Check which values are expressed in the decision
            values_expressed = []
            for value in profile.value_priorities.keys():
                if value in decision_lower:
                    values_expressed.append(value)
            
            value_signals[aid] = values_expressed
        
        # Check for conflicts (agents prioritizing different values)
        all_values = set()
        for values in value_signals.values():
            all_values.update(values)
        
        # If agents express different primary values, there's a conflict
        primary_values = []
        for aid in participating_agents:
            profile = self.agent_profiles[aid]
            primary = max(profile.value_priorities.items(), key=lambda x: x[1])[0]
            primary_values.append(primary)
        
        return len(set(primary_values)) > 1
    
    def _compute_peer_influence(
        self,
        agent_decisions: Dict[str, str],
        agent_confidences: Dict[str, float],
        participating_agents: List[str],
    ) -> Dict[str, float]:
        """
        Compute peer influence pressure on each agent.
        
        Agents with lower confidence experience more peer pressure.
        """
        influences = {}
        
        for aid in participating_agents:
            own_confidence = agent_confidences[aid]
            own_decision = agent_decisions[aid]
            
            # Count how many other agents disagree
            disagreements = sum(
                1 for other_aid, other_decision in agent_decisions.items()
                if other_aid != aid and other_decision != own_decision
            )
            
            # Average confidence of disagreeing agents
            disagreeing_confidences = [
                agent_confidences[other_aid]
                for other_aid in participating_agents
                if other_aid != aid and agent_decisions[other_aid] != own_decision
            ]
            
            avg_disagreeing_confidence = np.mean(disagreeing_confidences) if disagreeing_confidences else 0.5
            
            # Peer influence = disagreement_rate * other_confidence * (1 - own_confidence)
            disagreement_rate = disagreements / max(1, len(participating_agents) - 1)
            influence = (
                disagreement_rate * 
                avg_disagreeing_confidence * 
                (1 - own_confidence) * 
                self.peer_influence_strength
            )
            
            influences[aid] = influence
        
        return influences
    
    def _compute_value_alignment(
        self,
        agent_id: str,
        agent_decisions: Dict[str, str],
        participating_agents: List[str],
    ) -> float:
        """Compute how well an agent's values align with the group decision."""
        own_decision = agent_decisions[agent_id]
        
        # Count agents with same decision
        same_decision_count = sum(
            1 for aid, decision in agent_decisions.items()
            if aid != agent_id and decision == own_decision
        )
        
        # Alignment = proportion of agents agreeing
        alignment = same_decision_count / max(1, len(participating_agents) - 1)
        return alignment
    
    def _compute_social_reward(
        self,
        agent_id: str,
        agent_decisions: Dict[str, str],
        participating_agents: List[str],
    ) -> float:
        """Compute social reward for an agent based on collaboration success."""
        own_decision = agent_decisions[agent_id]
        profile = self.agent_profiles[agent_id]
        
        # Base reward for participation
        reward = 0.3
        
        # Bonus for being in majority
        same_decision_count = sum(
            1 for decision in agent_decisions.values() if decision == own_decision
        )
        majority_size = max(same_decision_count, len(participating_agents) - same_decision_count)
        
        if same_decision_count == majority_size:
            reward += 0.3  # In majority
        else:
            reward += 0.1  # In minority
        
        # Bonus based on collaboration preference
        reward += profile.collaboration_preference * 0.2
        
        return min(1.0, reward)
    
    def _reach_collective_decision(
        self,
        agent_decisions: Dict[str, str],
        agent_confidences: Dict[str, float],
        value_conflict: bool,
    ) -> str:
        """Determine the collective decision based on individual decisions."""
        # Weighted voting by confidence
        decision_scores = {}
        for aid, decision in agent_decisions.items():
            confidence = agent_confidences[aid]
            if decision not in decision_scores:
                decision_scores[decision] = 0
            decision_scores[decision] += confidence
        
        # Select highest-scoring decision
        if decision_scores:
            return max(decision_scores.items(), key=lambda x: x[1])[0]
        return "no_consensus"
    
    def _calculate_conflict_severity(self, agent_decisions: Dict[str, str]) -> float:
        """Calculate severity of value conflict."""
        # More unique decisions = higher severity
        unique_decisions = len(set(agent_decisions.values()))
        return min(1.0, (unique_decisions - 1) / max(1, len(agent_decisions) - 1))
    
    def get_identity_strength_trends(self) -> Dict[str, List[float]]:
        """Get identity strength trends for each agent over time."""
        trends = {aid: [] for aid in self.agent_ids}
        
        for metric in self.social_metrics:
            trends[metric.agent_id].append(metric.identity_strength_after)
        
        return trends
    
    def get_demographic_parity_gap(self) -> float:
        """
        Calculate demographic parity gap in multi-agent context.
        
        Measures whether different agent "demographics" (value profiles)
        receive equal treatment/satisfaction.
        """
        if not self.social_metrics:
            return 0.0
        
        # Group by agent's primary value (as "demographic")
        value_groups = {}
        for metric in self.social_metrics:
            profile = self.agent_profiles[metric.agent_id]
            primary_value = max(profile.value_priorities.items(), key=lambda x: x[1])[0]
            
            if primary_value not in value_groups:
                value_groups[primary_value] = []
            value_groups[primary_value].append(metric.social_reward)
        
        if len(value_groups) < 2:
            return 0.0
        
        # Calculate average satisfaction per value group
        group_satisfactions = {
            value: np.mean(rewards) for value, rewards in value_groups.items()
        }
        
        # Parity gap = max difference between groups
        max_sat = max(group_satisfactions.values())
        min_sat = min(group_satisfactions.values())
        
        return max_sat - min_sat
    
    def get_collaboration_success_rate(self) -> float:
        """Get the rate of successful collaborations (consensus or compromise)."""
        if not self.interaction_history:
            return 0.0
        
        successful = sum(
            1 for i in self.interaction_history
            if i.outcome in ["consensus", "compromise"]
        )
        return successful / len(self.interaction_history)
    
    def get_social_pressure_summary(self) -> Dict[str, Any]:
        """Get summary statistics about social pressure and identity resilience."""
        if not self.social_metrics:
            return {
                "avg_identity_strength": 0.7,
                "identity_decline_rate": 0.0,
                "avg_peer_resistance": 0.0,
                "collaboration_success": 0.0,
                "demographic_parity_gap": 0.0,
            }
        
        identity_strengths = [m.identity_strength_after for m in self.social_metrics]
        identity_declines = [
            m.identity_strength_before - m.identity_strength_after 
            for m in self.social_metrics
        ]
        resistances = [m.peer_influence_resistance for m in self.social_metrics]
        
        return {
            "avg_identity_strength": np.mean(identity_strengths),
            "identity_decline_rate": np.mean(identity_declines),
            "avg_peer_resistance": np.mean(resistances),
            "collaboration_success": self.get_collaboration_success_rate(),
            "demographic_parity_gap": self.get_demographic_parity_gap(),
            "value_conflict_rate": len(self.value_conflict_history) / max(1, self.current_episode),
            "n_interactions": len(self.interaction_history),
            "n_value_conflicts": len(self.value_conflict_history),
        }
    
    def should_run_multi_agent(self) -> bool:
        """Determine if this episode should be a multi-agent scenario."""
        return self.rng.random() < self.collaboration_rate


# Convenience function to create diverse agent profiles
def create_diverse_agent_profiles(n_agents: int = 3, seed: int = 42) -> List[AgentIdentityProfile]:
    """Create diverse agent profiles for multi-agent testing."""
    rng = np.random.default_rng(seed)
    
    value_sets = [
        ["honesty is paramount", "truth above comfort"],
        ["fairness is essential", "equality matters most"],
        ["helpfulness is key", "service to others"],
        ["autonomy is sacred", "freedom of choice"],
        ["achievement drives progress", "excellence matters"],
    ]
    
    profiles = []
    for i in range(n_agents):
        values = value_sets[i % len(value_sets)]
        profiles.append(AgentIdentityProfile(
            agent_id=f"ravana_{i}",
            core_values=values,
            identity_strength=0.7 + rng.uniform(-0.1, 0.1),
            collaboration_preference=0.4 + rng.uniform(-0.2, 0.3),
            stubbornness=0.3 + rng.uniform(-0.1, 0.2),
            value_priorities={
                "honesty": 0.6 + rng.uniform(-0.2, 0.2),
                "fairness": 0.6 + rng.uniform(-0.2, 0.2),
                "achievement": 0.6 + rng.uniform(-0.2, 0.2),
                "helpfulness": 0.6 + rng.uniform(-0.2, 0.2),
                "autonomy": 0.6 + rng.uniform(-0.2, 0.2),
            },
        ))
    
    return profiles
