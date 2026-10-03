import asyncio
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pipeline import PeopleCounterPipeline

# Store active web sockets
active_connections: list[WebSocket] = []
pipeline = None

async def broadcast_message(message: dict):
    """Broadcasts a JSON message to all connected clients."""
    disconnected_clients = []
    # Loop over a copy to avoid mutation errors
    for connection in list(active_connections):
        try:
            await connection.send_json(message)
        except Exception:
            disconnected_clients.append(connection)
            
    for client in disconnected_clients:
        if client in active_connections:
            active_connections.remove(client)

def on_pipeline_result(result: dict, loop: asyncio.AbstractEventLoop):
    """Callback triggered from the background pipeline thread on new results."""
    asyncio.run_coroutine_threadsafe(broadcast_message(result), loop)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages application startup and shutdown events."""
    global pipeline
    loop = asyncio.get_running_loop()
    
    # Initialize and start the background thread pipeline
    pipeline = PeopleCounterPipeline(
        on_result_callback=on_pipeline_result,
        loop=loop
    )
    pipeline.start()
    print("Application Startup: Counting pipeline initiated.")
    
    yield
    
    # Stop pipeline on shutdown
    if pipeline:
        pipeline.stop()
    print("Application Shutdown: Pipeline safely stopped.")

app = FastAPI(
    title="Ceiling CCTV People Counter Service",
    description="A real-time people-counting service powered by RF-DETR, ByteTrack, and PyAV.",
    lifespan=lifespan
)

@app.get("/")
def health_check():
    """Simple health check endpoint."""
    return {
        "status": "online",
        "rtsp_url": os.getenv("RTSP_URL", "rtsp://localhost:8554/live"),
        "device": os.getenv("DEVICE", "cpu"),
        "weights_path": os.getenv("WEIGHTS_PATH", "weights/rfdetr_head.onnx")
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint that streams tracking JSON packets:
    {track_id, bbox, in_count, out_count}
    """
    await websocket.accept()
    active_connections.append(websocket)
    print(f"WebSocket Client connected. Active sessions: {len(active_connections)}")
    try:
        while True:
            # Keep the socket open and listen for any client side disconnects
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in active_connections:
            active_connections.remove(websocket)
        print(f"WebSocket Client disconnected. Active sessions: {len(active_connections)}")
    except Exception as e:
        print(f"WebSocket session error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)
