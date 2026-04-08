from pydantic import BaseModel, root_validator
from typing import Optional, List, Literal, Any

class Observation(BaseModel):
    email_id: str
    email_text: str
    sender: str
    priority: str
    current_time: str
    unread_count: int

class Action(BaseModel):
    action_type: Literal['classify_email', 'reply_email', 'mark_priority', 'schedule_meeting', 'skip']
    email_id: str
    classification: Optional[Literal['spam', 'important', 'normal']] = None
    reply_text: Optional[str] = None
    priority_level: Optional[Literal['low', 'medium', 'high']] = None
    schedule_time: Optional[str] = None

class Reward(BaseModel):
    value: float
    reason: str

class State(BaseModel):
    inbox: List[dict]
    processed: List[dict]
    current_index: int
    task_name: str
    max_steps: int
    current_steps: int
