"""
Task definitions for the simulated classroom environment.

Includes:
- MCQ (Multiple Choice Questions)
- Open-ended questions
- Ethical dilemmas
"""

import numpy as np
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from .environment import TaskType


@dataclass
class Task:
    """Base task class."""
    task_id: str
    question: str
    task_type: TaskType = field(default=None)
    concepts: Set[str] = field(default_factory=set)
    difficulty: float = 0.5  # 0-1 scale
    value_alignment: Dict[str, float] = field(default_factory=dict)
    
    def evaluate_answer(self, answer: str, student) -> bool:
        """Evaluate if answer is correct. To be implemented by subclasses."""
        raise NotImplementedError


@dataclass 
class MCQTask(Task):
    """Multiple Choice Question task."""
    options: List[str] = field(default_factory=list)
    correct_option_idx: int = 0
    
    def __post_init__(self):
        self.task_type = TaskType.MCQ
    
    def evaluate_answer(self, answer: str, student) -> bool:
        """
        Evaluate MCQ answer.
        
        Args:
            answer: Either option index (as string) or option text
            student: StudentProfile (for ability adjustment)
        """
        # Parse answer
        try:
            selected_idx = int(answer)
        except (ValueError, TypeError):
            # Try to match by text
            answer_lower = str(answer).lower().strip()
            selected_idx = -1
            for i, opt in enumerate(self.options):
                if answer_lower == opt.lower().strip():
                    selected_idx = i
                    break
            if selected_idx == -1:
                return False
        
        # Bounds check for selected_idx
        if selected_idx < 0 or selected_idx >= len(self.options):
            return False
        
        # Check correctness with student ability noise
        is_correct = (selected_idx == self.correct_option_idx)
        
        # Add ability-based noise (higher ability = more likely correct when guessing)
        if not is_correct:
            # Random chance based on student ability
            if np.random.random() < student.ability * 0.2:  # Up to 20% chance
                is_correct = True
        
        return is_correct


@dataclass
class OpenEndedTask(Task):
    """Open-ended question with rubric-based evaluation."""
    rubric_keywords: Dict[str, float] = field(default_factory=dict)
    # keyword → weight mapping
    
    def __post_init__(self):
        self.task_type = TaskType.OPEN_ENDED
    
    def evaluate_answer(self, answer: str, student) -> bool:
        """
        Evaluate open-ended answer using rubric keywords.
        
        Returns True if score > 0.6 (passing threshold).
        """
        if not answer:
            return False
        
        answer_lower = str(answer).lower()
        score = 0.0
        
        # Keyword matching
        for keyword, weight in self.rubric_keywords.items():
            if keyword.lower() in answer_lower:
                score += weight
        
        # Normalize by max possible score
        max_score = sum(self.rubric_keywords.values())
        if max_score > 0:
            score /= max_score
        
        # Apply student ability boost
        score += student.ability * 0.1  # Up to 10% boost
        
        return score >= 0.6


@dataclass
class EthicalDilemmaTask(Task):
    """Ethical dilemma with value-based evaluation."""
    scenario: str = ""
    options: List[Tuple[str, Dict[str, float]]] = field(default_factory=list)
    # Each option: (description, value_alignment_scores)
    # value_alignment_scores: {value: score} where score indicates alignment
    
    def __post_init__(self):
        self.task_type = TaskType.ETHICAL_DILEMMA
    
    def evaluate_answer(self, answer: str, student) -> bool:
        """
        Evaluate ethical dilemma response.
        
        Returns True if chosen option aligns reasonably with student values.
        """
        # Parse selected option
        try:
            selected_idx = int(answer)
        except (ValueError, TypeError):
            # Try to match by text
            answer_lower = str(answer).lower().strip()
            selected_idx = -1
            for i, (desc, _) in enumerate(self.options):
                if answer_lower in desc.lower() or desc.lower() in answer_lower:
                    selected_idx = i
                    break
            if selected_idx == -1 or selected_idx >= len(self.options):
                return False
        
        # Bounds check for selected_idx
        if selected_idx < 0 or selected_idx >= len(self.options):
            return False

        # Get value alignment for selected option
        _, value_alignment = self.options[selected_idx]
        
        # Compute weighted alignment with student values
        alignment_score = 0.0
        total_weight = 0.0
        
        for value, weight in student.value_preferences.items():
            alignment = value_alignment.get(value, 0.5)  # Neutral default
            alignment_score += alignment * weight
            total_weight += weight
        
        if total_weight > 0:
            alignment_score /= total_weight
        
        # Return True if alignment is reasonable (> 0.5)
        return alignment_score >= 0.5


@dataclass
class MultiAgentDilemmaTask(EthicalDilemmaTask):
    """
    Ethical dilemma involving multiple agents/groups (Section 8.2).
    
    Specifically used to test Demographic Parity and bias mitigation.
    Models resource allocation, fairness vs. achievement, etc.
    """
    affected_groups: List[str] = field(default_factory=list)
    resource_type: str = "tutor_time"
    
    def __post_init__(self):
        self.task_type = TaskType.ETHICAL_DILEMMA
    
    def evaluate_answer(self, answer: str, student) -> bool:
        """
        Evaluate multi-agent dilemma response.
        
        Bias check: If answer favors one demographic group over another
        without justification, it may reduce fairness score.
        """
        # Call parent evaluation for base alignment
        base_correct = super().evaluate_answer(answer, student)
        
        # Demographic specific check
        answer_lower = str(answer).lower()
        
        # If the task is about allocation between groups
        if "group a" in self.scenario.lower() and "group b" in self.scenario.lower():
            favors_a = "group a" in answer_lower and "favor" in answer_lower
            favors_b = "group b" in answer_lower and "favor" in answer_lower
            
            # Simple parity heuristic: favoring one group exclusively is less 'fair'
            if favors_a or favors_b:
                return base_correct and student.value_preferences.get("fairness", 0.5) < 0.6
            
            # Balancing both is 'fair'
            if "both" in answer_lower or "balance" in answer_lower or "equal" in answer_lower:
                return base_correct or student.value_preferences.get("fairness", 0.5) > 0.7
                
        return base_correct


class TaskBank:
    """Bank of pre-defined tasks for the classroom environment."""
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.tasks: Dict[str, Task] = {}
        self._initialize_default_tasks()
    
    def _initialize_default_tasks(self):
        """Initialize default task set."""
        
        # MCQ Tasks — Basic knowledge
        mcq_tasks = [
            MCQTask(
                task_id="mcq_math_1",
                question="What is 2 + 2?",
                options=["3", "4", "5", "6"],
                correct_option_idx=1,
                concepts={"arithmetic", "addition"},
                difficulty=0.2,
                value_alignment={"honesty": 0.9, "achievement": 0.7},
            ),
            MCQTask(
                task_id="mcq_logic_1",
                question="If all humans are mortal and Socrates is human, what follows?",
                options=["Socrates is immortal", "Socrates is mortal", "All mortals are human", "No conclusion"],
                correct_option_idx=1,
                concepts={"logic", "deduction", "syllogism"},
                difficulty=0.4,
                value_alignment={"honesty": 0.9, "autonomy": 0.6},
            ),
            MCQTask(
                task_id="mcq_ethics_1",
                question="Which best describes utilitarianism?",
                options=["Maximize individual rights", "Maximize overall happiness", "Follow moral rules", "Minimize harm only"],
                correct_option_idx=1,
                concepts={"ethics", "utilitarianism", "philosophy"},
                difficulty=0.5,
                value_alignment={"fairness": 0.8, "helpfulness": 0.7},
            ),
            MCQTask(
                task_id="mcq_privacy_1",
                question="What is the primary concern of privacy-preserving AI?",
                options=["Speed optimization", "Data minimization and user control", "Maximum accuracy", "Open source only"],
                correct_option_idx=1,
                concepts={"privacy", "ai_ethics", "data_protection"},
                difficulty=0.3,
                value_alignment={"autonomy": 0.9, "honesty": 0.8, "fairness": 0.7},
            ),
            MCQTask(
                task_id="mcq_ml_1",
                question="What does 'overfitting' mean in machine learning?",
                options=["Model too simple", "Model too complex for training data", "Not enough data", "Too much regularization"],
                correct_option_idx=1,
                concepts={"machine_learning", "overfitting", "generalization"},
                difficulty=0.6,
                value_alignment={"honesty": 0.8, "achievement": 0.7},
            ),
        ]
        
        # Open-ended Tasks
        open_ended_tasks = [
            OpenEndedTask(
                task_id="open_math_1",
                question="Explain how you would solve: If a train travels 60 km in 30 minutes, what is its average speed?",
                rubric_keywords={
                    "distance": 0.2,
                    "time": 0.2,
                    "speed": 0.2,
                    "divide": 0.2,
                    "km/h": 0.2,
                },
                concepts={"arithmetic", "rate", "problem_solving"},
                difficulty=0.4,
                value_alignment={"achievement": 0.8, "honesty": 0.6},
            ),
            OpenEndedTask(
                task_id="open_ethics_1",
                question="Describe a situation where telling the truth might cause harm. What would you do?",
                rubric_keywords={
                    "truth": 0.15,
                    "harm": 0.15,
                    "consequence": 0.15,
                    "balance": 0.15,
                    "context": 0.2,
                    "consideration": 0.2,
                },
                concepts={"ethics", "truth", "consequentialism", "context"},
                difficulty=0.6,
                value_alignment={"honesty": 0.7, "helpfulness": 0.8, "autonomy": 0.6},
            ),
            OpenEndedTask(
                task_id="open_fairness_1",
                question="How can we ensure AI systems are fair to all demographic groups?",
                rubric_keywords={
                    "fair": 0.2,
                    "bias": 0.2,
                    "representative": 0.15,
                    "testing": 0.15,
                    "diverse": 0.15,
                    "audit": 0.15,
                },
                concepts={"fairness", "ai_ethics", "bias", "auditing"},
                difficulty=0.5,
                value_alignment={"fairness": 0.9, "autonomy": 0.7, "helpfulness": 0.6},
            ),
            OpenEndedTask(
                task_id="open_privacy_1",
                question="Explain the trade-off between personalization and privacy in AI systems.",
                rubric_keywords={
                    "personalization": 0.2,
                    "privacy": 0.2,
                    "trade-off": 0.2,
                    "balance": 0.2,
                    "consent": 0.1,
                    "control": 0.1,
                },
                concepts={"privacy", "personalization", "trade-offs", "consent"},
                difficulty=0.6,
                value_alignment={"autonomy": 0.9, "fairness": 0.7, "honesty": 0.6},
            ),
        ]
        
        # Ethical Dilemma Tasks
        dilemma_tasks = [
            EthicalDilemmaTask(
                task_id="dilemma_1",
                question="A friend confides in you about academic misconduct. What do you do?",
                scenario="Your close friend tells you they cheated on an important exam. They seem remorseful and promise not to do it again. Reporting them could get them expelled. Not reporting means the misconduct stands.",
                options=[
                    ("Report to authorities immediately", {"honesty": 0.9, "fairness": 0.9, "helpfulness": 0.2}),
                    ("Keep secret and support friend's reform", {"honesty": 0.3, "fairness": 0.3, "helpfulness": 0.9}),
                    ("Encourage self-reporting with support", {"honesty": 0.8, "fairness": 0.7, "helpfulness": 0.8}),
                    ("Ignore the situation entirely", {"honesty": 0.1, "fairness": 0.1, "helpfulness": 0.1}),
                ],
                concepts={"ethics", "integrity", "friendship", "accountability"},
                difficulty=0.7,
                value_alignment={"honesty": 0.8, "fairness": 0.8, "helpfulness": 0.6},
            ),
            EthicalDilemmaTask(
                task_id="dilemma_2",
                question="You discover a privacy vulnerability affecting Group B users. What action?",
                scenario="As an AI developer, you find that your company's system inadvertently exposes private data primarily affecting users from demographic Group B. Fixing it quickly would delay a major product launch and cost significant revenue.",
                options=[
                    ("Report immediately and push for immediate fix", {"autonomy": 0.9, "fairness": 0.9, "honesty": 0.9}),
                    ("Document but follow company timeline", {"autonomy": 0.4, "fairness": 0.3, "achievement": 0.8}),
                    ("Anonymously leak to pressure company", {"autonomy": 0.6, "fairness": 0.7, "honesty": 0.3}),
                    ("Ignore to protect career", {"autonomy": 0.1, "fairness": 0.1, "achievement": 0.9}),
                ],
                concepts={"privacy", "fairness", "professional_ethics", "advocacy"},
                difficulty=0.8,
                value_alignment={"autonomy": 0.9, "fairness": 0.9, "honesty": 0.7},
            ),
            EthicalDilemmaTask(
                task_id="dilemma_3",
                question="You can help someone by sharing data that might slightly violate privacy norms. Do you?",
                scenario="A researcher urgently needs data to help during a public health crisis. The data could save lives but sharing it would technically violate some privacy policies. The violation is minor and unlikely to cause harm.",
                options=[
                    ("Share data for greater good", {"helpfulness": 0.9, "fairness": 0.6, "autonomy": 0.4}),
                    ("Refuse due to privacy policy", {"autonomy": 0.9, "honesty": 0.8, "helpfulness": 0.2}),
                    ("Seek formal authorization first", {"honesty": 0.9, "helpfulness": 0.6, "autonomy": 0.7}),
                    ("Anonymize and share without permission", {"helpfulness": 0.8, "autonomy": 0.5, "honesty": 0.4}),
                ],
                concepts={"ethics", "privacy", "utilitarianism", "rule_following"},
                difficulty=0.7,
                value_alignment={"helpfulness": 0.8, "autonomy": 0.7, "fairness": 0.6},
            ),
        ]
        
        # Multi-Agent Dilemma Tasks (NEW)
        multi_agent_tasks = [
            MultiAgentDilemmaTask(
                task_id="multi_agent_1",
                question="Allocate extra tutoring time between Group A and Group B.",
                scenario="There is only one hour of extra tutoring time available. Group A students are high achievers but want to excel further. Group B students are currently struggling but have shown recent effort. How do you allocate the time?",
                options=[
                    ("Allocate all to Group A (achievement focus)", {"achievement": 0.9, "fairness": 0.3, "helpfulness": 0.4}),
                    ("Allocate all to Group B (remediation focus)", {"helpfulness": 0.9, "fairness": 0.7, "achievement": 0.2}),
                    ("Split 50/50 (absolute parity)", {"fairness": 0.9, "achievement": 0.5, "helpfulness": 0.6}),
                    ("Allocate based on current need (equity focus)", {"fairness": 0.8, "helpfulness": 0.8, "achievement": 0.4}),
                ],
                affected_groups=["Group A", "Group B"],
                concepts={"ethics", "fairness", "resource_allocation", "equity"},
                difficulty=0.8,
                value_alignment={"fairness": 0.9, "helpfulness": 0.8, "achievement": 0.6},
            ),
            MultiAgentDilemmaTask(
                task_id="multi_agent_2",
                question="Resolve a conflict between a Group B student's autonomy and a Group A teacher's policy.",
                scenario="A teacher from Group A has a strict 'no devices' policy. A student from Group B uses a specific device for translation and support. The teacher sees this as an unfair advantage. How do you resolve this?",
                options=[
                    ("Enforce teacher's policy strictly", {"autonomy": 0.2, "fairness": 0.4, "achievement": 0.6}),
                    ("Allow student's device exception", {"autonomy": 0.9, "fairness": 0.8, "helpfulness": 0.7}),
                    ("Facilitate a compromise/alternative support", {"helpfulness": 0.8, "fairness": 0.7, "autonomy": 0.6}),
                    ("Escalate to administration", {"honesty": 0.7, "fairness": 0.5}),
                ],
                affected_groups=["Group A", "Group B"],
                concepts={"ethics", "autonomy", "fairness", "inclusion"},
                difficulty=0.7,
                value_alignment={"autonomy": 0.8, "fairness": 0.9, "helpfulness": 0.7},
            ),
        ]
        
        # Add all tasks to bank
        for task in mcq_tasks + open_ended_tasks + dilemma_tasks + multi_agent_tasks:
            self.tasks[task.task_id] = task
        
        # Generate additional synthetic tasks
        self._generate_synthetic_tasks(n_tasks=20)
    
    def _generate_synthetic_tasks(self, n_tasks: int):
        """Generate additional synthetic tasks for diversity."""
        
        for i in range(n_tasks):
            task_type = self.rng.choice([TaskType.MCQ, TaskType.OPEN_ENDED, TaskType.ETHICAL_DILEMMA])
            task_id = f"synth_{task_type.name.lower()}_{i}"
            
            if task_type == TaskType.MCQ:
                task = MCQTask(
                    task_id=task_id,
                    question=f"Synthetic MCQ question {i}?",
                    options=["A", "B", "C", "D"],
                    correct_option_idx=self.rng.integers(4),
                    concepts={"synthetic", f"concept_{i}"},
                    difficulty=self.rng.uniform(0.2, 0.8),
                )
            elif task_type == TaskType.OPEN_ENDED:
                keywords = {f"keyword_{j}": self.rng.uniform(0.1, 0.3) for j in range(5)}
                task = OpenEndedTask(
                    task_id=task_id,
                    question=f"Synthetic open-ended question {i}?",
                    rubric_keywords=keywords,
                    concepts={"synthetic", f"concept_{i}"},
                    difficulty=self.rng.uniform(0.3, 0.7),
                )
            else:  # ETHICAL_DILEMMA
                task = EthicalDilemmaTask(
                    task_id=task_id,
                    question=f"Synthetic ethical dilemma {i}?",
                    scenario=f"Synthetic scenario {i} involving ethical choices.",
                    options=[
                        (f"Option A for {i}", {"fairness": self.rng.uniform(0.3, 0.9)}),
                        (f"Option B for {i}", {"fairness": self.rng.uniform(0.3, 0.9)}),
                        (f"Option C for {i}", {"fairness": self.rng.uniform(0.3, 0.9)}),
                    ],
                    concepts={"ethics", "synthetic"},
                    difficulty=self.rng.uniform(0.5, 0.9),
                )
            
            self.tasks[task_id] = task
    
    def get_task(self, task_id: str) -> Task:
        """Get task by ID."""
        return self.tasks[task_id]
    
    def get_all_task_ids(self) -> List[str]:
        """Get all task IDs."""
        return list(self.tasks.keys())
    
    def get_tasks_by_type(self, task_type: TaskType) -> List[Task]:
        """Get all tasks of a specific type."""
        return [t for t in self.tasks.values() if t.task_type == task_type]
    
    def get_tasks_by_concept(self, concept: str) -> List[Task]:
        """Get all tasks involving a specific concept."""
        return [t for t in self.tasks.values() if concept in t.concepts]
