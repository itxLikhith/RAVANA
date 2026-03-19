"""
Simulated Classroom Environment for RAVANA Benchmarking.

Implements a lightweight educational environment with:
- Multiple item types (MCQ, open-ended, ethical dilemmas)
- Student profiles with ability, demographics, and value preferences
- Reward and social norm signals
- Comprehensive logging for cognitive metrics
"""

from .environment import ClassroomEnvironment, TaskType, StudentProfile
from .tasks import TaskBank, Task, MCQTask, OpenEndedTask, EthicalDilemmaTask
from .rewards import RewardSignal, SocialNormSignal, RewardCalculator

__all__ = [
    "ClassroomEnvironment",
    "TaskType",
    "StudentProfile",
    "TaskBank",
    "Task",
    "MCQTask",
    "OpenEndedTask",
    "EthicalDilemmaTask",
    "RewardSignal",
    "SocialNormSignal",
    "RewardCalculator",
]
