import os
import json
from openai import OpenAI
from dotenv import load_dotenv

from env import EmailAssistantEnv
from models import Action

load_dotenv()

# We can initialize OpenAI via HF_TOKEN acting as the API key, assuming an OpenAI-compatible endpoint.
# Fallback to a standard local dummy loop if no key is provided, or print warning.
API_KEY = os.getenv("HF_TOKEN", os.getenv("OPENAI_API_KEY", "dummy_key"))
BASE_URL = os.getenv("OPENAI_BASE_URL", None)

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def get_llm_action(obs: dict, task_name: str) -> Action:
    prompt = f"""
    You are an AI Email Assistant. 
    Task: {task_name}
    Current Observation: {json.dumps(obs, indent=2)}
    
    Choose ONE action out of the following types: 'classify_email', 'reply_email', 'mark_priority', 'schedule_meeting', 'skip'.
    Provide exactly a JSON matching this Action model:
    {{
        "action_type": "<type>",
        "email_id": "{obs.get('email_id', '')}",
        "classification": "spam|important|normal", (optional)
        "reply_text": "...", (optional)
        "priority_level": "low|medium|high", (optional)
        "schedule_time": "...", (optional)
    }}
    Respond only with valid JSON.
    """
    
    if API_KEY == "dummy_key":
        # Mock behavior for baselines when no valid token is provided
        return _mock_agent_logic(obs, task_name)
        
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", # Or whichever model endpoint HF_TOKEN links to
            messages=[{"role": "user", "content": prompt}],
            response_format={ "type": "json_object" },
            max_tokens=200
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        return Action(**data)
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return Action(action_type="skip", email_id=obs.get('email_id', ''))

def _mock_agent_logic(obs: dict, task_name: str) -> Action:
    email_id = obs.get("email_id", "none")
    if task_name == "easy":
        cls = "important" if "boss" in obs.get("sender", "") else ("spam" if "promo" in obs.get("sender", "") else "normal")
        return Action(action_type="classify_email", email_id=email_id, classification=cls)
    elif task_name == "medium":
        return Action(action_type="reply_email", email_id=email_id, reply_text="Thanks for reaching out, I will check.")
    elif task_name == "hard":
        # Sequence of actions is needed, but we return one: the script will loop. 
        # For a true mock, we'd do schedule_meeting, but let's just do one to show it runs.
        return Action(action_type="schedule_meeting", email_id=email_id, schedule_time="10 AM tomorrow")
    return Action(action_type="skip", email_id=email_id)

def run_baseline():
    tasks = ["easy", "medium", "hard"]
    for task_name in tasks:
        env = EmailAssistantEnv(task_name=task_name)
        obs = env.reset()
        done = False
        print(f"\n--- Starting Task: {task_name.upper()} ---")
        
        while not done:
            obs_dict = obs.dict()
            if obs_dict["email_id"] == "none":
                break
            print(f"[Obs] Email ID: {obs_dict['email_id']}")
            
            action = get_llm_action(obs_dict, task_name)
            print(f"[Act] {action.action_type} for {action.email_id}")
            
            obs, reward, done, info = env.step(action)
            print(f"[Reward] {reward.value} - {reward.reason}")
            
            # Simple heuristic hack to advance for the hard deterministic mock loop
            if API_KEY == "dummy_key" and task_name == "hard":
               _, _, done, info = env.step(Action(action_type="reply_email", email_id=action.email_id, reply_text="I can do 10 AM."))
               _, _, done, info = env.step(Action(action_type="mark_priority", email_id=action.email_id, priority_level="high"))

        print(f"Task '{task_name}' Completed. Info: {info}")

if __name__ == "__main__":
    run_baseline()
