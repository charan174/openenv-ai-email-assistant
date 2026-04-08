"""
app.py — FastAPI server for the AI Email Assistant OpenEnv environment.

Exposes the OpenEnv-compliant HTTP API:
  POST /reset   → reset environment for a given task
  POST /step    → submit an action, get observation + reward + done + info
  GET  /state   → read current observation without side effects

Also serves a minimal health-check root endpoint for Hugging Face Spaces.
"""

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, Optional

from models import Observation, Action, Reward
from environment import EmailAssistantEnv

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AI Email Assistant — OpenEnv",
    description=(
        "An OpenEnv-compliant environment simulating real-world email management. "
        "Tasks include classification, reply generation, and priority/scheduling."
    ),
    version="1.0.0",
)

# Enable CORS for browser-based agents
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global environment instance (re-created on each /reset)
# ---------------------------------------------------------------------------
env: Optional[EmailAssistantEnv] = None


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_name: str = "easy"


class ResetResponse(BaseModel):
    observation: Observation


class StepRequest(BaseModel):
    action: Action


class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any] = {}


class StateResponse(BaseModel):
    observation: Observation


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    """Health-check / landing page for Hugging Face Spaces."""
    return {
        "service": "AI Email Assistant — OpenEnv",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": ["/reset", "/step", "/state"],
    }


@app.post("/reset", response_model=ResetResponse)
def reset(request: ResetRequest):
    """
    Reset the environment to a fresh state for the specified task.

    Task names: 'easy', 'medium', 'hard'
    """
    global env
    env = EmailAssistantEnv(task_name=request.task_name)
    obs = env.reset()
    return ResetResponse(observation=obs)


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest):
    """
    Submit an agent action and receive the next observation, reward,
    done flag, and optional info dict.
    """
    if env is None:
        # Auto-reset to easy if not initialised
        return _auto_reset_and_step(request)

    obs, reward, done, info = env.step(request.action)
    return StepResponse(
        observation=obs,
        reward=reward.value,
        done=done,
        info=info,
    )


@app.get("/state", response_model=StateResponse)
def state():
    """
    Return the current observation without modifying the environment.
    """
    if env is None:
        # Return a placeholder observation if not yet initialised
        placeholder = Observation(
            email_id="none",
            subject="",
            email_text="Environment not initialised. Call POST /reset first.",
            sender="system",
            timestamp="",
            priority="none",
            current_time="",
            unread_count=0,
            task_name="",
            task_description="",
        )
        return StateResponse(observation=placeholder)

    return StateResponse(observation=env.get_state())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _auto_reset_and_step(request: StepRequest) -> StepResponse:
    """Convenience: auto-reset to 'easy' and then step."""
    global env
    env = EmailAssistantEnv(task_name="easy")
    env.reset()
    obs, reward, done, info = env.step(request.action)
    return StepResponse(
        observation=obs,
        reward=reward.value,
        done=done,
        info=info,
    )
