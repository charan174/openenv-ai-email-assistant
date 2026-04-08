"""
models.py — Pydantic models for the AI Email Assistant OpenEnv environment.

Defines the core data structures exchanged between the agent and the
environment: Observation (what the agent sees), Action (what the agent does),
and Reward (feedback on the action).
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Literal, Any, Dict


class Observation(BaseModel):
    """
    Represents the current state visible to the agent.
    Contains the email currently being processed plus inbox metadata.
    """
    email_id: str = Field(..., description="Unique identifier of the current email")
    subject: str = Field("", description="Subject line of the email")
    email_text: str = Field(..., description="Full body text of the email")
    sender: str = Field(..., description="Sender email address")
    timestamp: str = Field("", description="When the email was received")
    priority: str = Field("unknown", description="Current priority label (unknown if unset)")
    current_time: str = Field(..., description="Simulated current wall-clock time")
    unread_count: int = Field(0, description="Number of unread emails remaining in inbox")
    task_name: str = Field("", description="Name of the active task")
    task_description: str = Field("", description="Human-readable description of the task objective")


class Action(BaseModel):
    """
    Represents an agent action on the current email.
    Only the fields relevant to `action_type` need to be populated.
    """
    action_type: Literal[
        "classify_email",
        "reply_email",
        "mark_priority",
        "schedule_meeting",
        "skip",
    ] = Field(..., description="Type of action to perform")
    email_id: str = Field(..., description="ID of the email this action targets")
    classification: Optional[Literal["spam", "important", "normal"]] = Field(
        None, description="Classification label (required for classify_email)"
    )
    reply_text: Optional[str] = Field(
        None, description="Reply body text (required for reply_email)"
    )
    priority_level: Optional[Literal["low", "medium", "high"]] = Field(
        None, description="Priority level (required for mark_priority)"
    )
    schedule_time: Optional[str] = Field(
        None, description="Meeting time string (required for schedule_meeting)"
    )


class Reward(BaseModel):
    """
    Feedback signal returned after every step.
    Contains both a numeric score and a human-readable explanation.
    """
    value: float = Field(..., description="Numeric reward between -1.0 and 1.0")
    reason: str = Field("", description="Human-readable explanation of the reward")


class EnvironmentState(BaseModel):
    """
    Internal bookkeeping state of the environment, exposed via GET /state.
    """
    inbox: List[Dict[str, Any]] = Field(default_factory=list)
    processed: List[Dict[str, Any]] = Field(default_factory=list)
    current_index: int = 0
    task_name: str = ""
    max_steps: int = 0
    current_step: int = 0
    cumulative_reward: float = 0.0
