from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict, Any

app = FastAPI()

# ---------------- MODELS ----------------

class Observation(BaseModel):
    email: str
    sender: str
    priority: str

class Action(BaseModel):
    classify_email: Optional[str] = None
    reply_email: Optional[str] = None
    mark_priority: Optional[str] = None

# ---------------- STATE ----------------

current_task = {
    "email": "Win a free iPhone!",
    "sender": "spam@fake.com",
    "priority": "low"
}

# ---------------- RESET ----------------

@app.post("/reset")
def reset():
    return {
        "observation": current_task   # ✅ REQUIRED FORMAT
    }

# ---------------- STEP ----------------

@app.post("/step")
def step(action: Action):
    score = 0.0

    # Classification
    if action.classify_email == "spam":
        score += 0.5

    # Reply
    if action.reply_email:
        score += 0.3

    # Priority
    if action.mark_priority == "high":
        score += 0.2

    return {
        "observation": current_task,
        "reward": float(score),   # ✅ ensure float
        "done": True,
        "info": {}
    }

# ---------------- STATE ----------------

@app.get("/state")
def state():
    return {
        "observation": current_task
    }