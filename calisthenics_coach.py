from vision_agents.core import Agent, AgentLauncher, User, Runner
from vision_agents.plugins import getstream, gemini, ultralytics
from dotenv import load_dotenv
import os
import asyncio

# Import WebSocket server
from websocket_server import send_to_frontend

load_dotenv()

SELECTED_EXERCISE = "pushups"

EXERCISE_FILES = {
    "pushups": "instructions/pushups.md",
    "squats": "instructions/squats.md",
    "pullups": "instructions/pullups.md",
    "planks": "instructions/planks.md",
    "dips": "instructions/dips.md",
    "burpees": "instructions/burpees.md",
}

def load_instructions(exercise_name: str) -> str:
    file_path = EXERCISE_FILES.get(exercise_name)
    if not file_path or not os.path.exists(file_path):
        raise ValueError(f"Instruction file not found: {file_path}")
    with open(file_path, 'r') as f:
        return f.read()

# Global state for tracking reps and scores
current_reps = 0
last_score = 0

async def on_yolo_keypoints(keypoints):
    """Called when YOLO detects keypoints"""
    # Send keypoints to frontend for skeleton overlay
    await send_to_frontend("keypoints", {"keypoints": keypoints})

async def on_rep_completed(score):
    """Called when agent detects a completed rep"""
    global current_reps, last_score
    current_reps += 1
    last_score = score
    
    # Send rep update to frontend
    await send_to_frontend("rep_update", {
        "reps": current_reps,
        "score": score
    })

async def on_agent_feedback(text):
    """Called when agent gives verbal feedback"""
    # Send feedback to frontend
    await send_to_frontend("feedback", {"text": text})

async def create_agent(**kwargs) -> Agent:
    instructions = load_instructions(SELECTED_EXERCISE)
    
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name=f"Calisthenics Coach ({SELECTED_EXERCISE.title()})", id="agent"),
        instructions=instructions,
        llm=gemini.Realtime(fps=10),
        processors=[ultralytics.YOLOPoseProcessor(model_path="yolo11n-pose.pt")],
    )
    
    # Wire up callbacks (THIS IS PSEUDOCODE — check Vision Agents docs for actual API)
    # agent.on('keypoints_detected', on_yolo_keypoints)
    # agent.on('rep_completed', on_rep_completed)
    # agent.on('feedback_given', on_agent_feedback)
    
    return agent

async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    call = await agent.create_call(call_type, call_id)
    async with agent.join(call):
        greeting = f"Hey! I'm your calisthenics coach. Let's work on your {SELECTED_EXERCISE}. Show me your form!"
        await agent.simple_response(greeting)
        await agent.finish()

if __name__ == "__main__":
    # Run WebSocket server in background
    import subprocess
    subprocess.Popen(["python", "-m", "uvicorn", "websocket_server:app", "--port", "8000"])
    
    # Run agent
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()