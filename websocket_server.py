from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json

app = FastAPI()

# Allow frontend to connect from localhost:3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active WebSocket connections
active_connections = []

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        while True:
            # Keep connection alive
            await asyncio.sleep(1)
    except:
        active_connections.remove(websocket)

async def send_to_frontend(data_type, payload):
    """Send data to all connected frontend clients"""
    message = json.dumps({"type": data_type, **payload})
    for connection in active_connections:
        try:
            await connection.send_text(message)
        except:
            active_connections.remove(connection)

# Run this server alongside your agent
# python -m uvicorn websocket_server:app --port 8000