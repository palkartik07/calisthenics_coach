from vision_agents.core import Agent, AgentLauncher, User, Runner
from vision_agents.plugins import getstream, openai, ultralytics  # Using OpenAI instead of Gemini
from dotenv import load_dotenv

load_dotenv()

# TODO: Later, you'll add logic here to switch prompts based on exercise selection
# For now, just hardcode push-ups

async def create_agent(**kwargs) -> Agent:
    return Agent(
        edge=getstream.Edge(),
        agent_user=User(name="Calisthenics Coach", id="agent"),
        
        # 🔴 CHANGE THIS: Your push-up coaching prompt
        instructions="""
        You are a calisthenics coach watching the user do push-ups via webcam.
        You will receive YOLO pose keypoints showing body positions.
        
        Check these form cues:
        1. Elbow angle at bottom (should be ~90 degrees or less)
        2. Back alignment (shoulders, hips, ankles in a straight line)
        3. Depth (chest comes close to floor)
        4. Hand position (roughly shoulder-width)
        
        After each rep, give brief spoken feedback on form.
        Score each rep 1-10.
        Keep feedback concise - one sentence max.
        """,
        
        # 🔴 You can switch between openai.Realtime() or gemini.Realtime()
        llm=openai.Realtime(fps=10),  # 10 frames per second
        
        # 🟢 This stays the same — YOLO pose works for all exercises
        processors=[ultralytics.YOLOPoseProcessor(model_path="yolo11n-pose.pt")],
    )

async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    call = await agent.create_call(call_type, call_id)
    async with agent.join(call):
        # 🔴 CHANGE THIS: Your greeting message
        await agent.simple_response("Hey! I'm your calisthenics coach. Let's work on those push-ups. Show me your form!")
        await agent.finish()

if __name__ == "__main__":
    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()