from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import asyncio
import json
from typing import List
import sys
import os
import traceback
import logging
import threading
from pathlib import Path
import time
import uvicorn
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('server.log')
    ]
)

# Add the project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.eeg.data_acquisition import EEGStream
    from src.eeg.signal_processing import SignalProcessor
    from src.ml.emotion_classifier import EmotionClassifier
    from src.chatbot.therapeutic_bot import TherapeuticBot
    logging.info("Successfully imported required modules")
except ImportError as e:
    logging.error(f"Failed to import required modules: {e}")
    logging.error(f"Python path: {sys.path}")
    logging.error(traceback.format_exc())
    sys.exit(1)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logging.info("FastAPI server starting up...")
    try:
        eeg_stream = EEGStream()
        signal_processor = SignalProcessor()
        emotion_classifier = EmotionClassifier('models/emotion_classifier.keras')
        chatbot = TherapeuticBot()
        
        # Create emotion state
        current_emotion = "neutral"
        current_confidence = 0.5
        
        if not eeg_stream.connect():
            logging.warning("EEG hardware not detected. Running in simulation mode.")
        else:
            logging.info("EEG system initialized successfully!")

    except Exception as e:
        logging.error(f"Error initializing EEG system components: {e}")
        logging.error(traceback.format_exc())
        logging.info("Running in simulation mode.")
        eeg_stream = None
        current_emotion = "Simulation Mode"
        current_confidence = 0.0
    yield
    # Shutdown
    logging.info("FastAPI server shutting down...")

# Create FastAPI app with custom error handlers
app = FastAPI(
    title="EEG Analysis Chatbot API",
    description="API for EEG-based emotion analysis and chatbot interaction",
    version="1.0.0",
    lifespan=lifespan
)

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logging.error(f"Unhandled exception: {exc}")
    logging.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error occurred"}
    )

# Add CORS middleware with more specific configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Store WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._lock = threading.Lock()

    async def connect(self, websocket: WebSocket):
        try:
            await websocket.accept()
            with self._lock:
                self.active_connections.append(websocket)
            logging.info(f"New WebSocket connection. Total connections: {len(self.active_connections)}")
        except Exception as e:
            logging.error(f"Error accepting WebSocket connection: {e}")
            raise

    def disconnect(self, websocket: WebSocket):
        with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
                logging.info(f"WebSocket disconnected. Remaining connections: {len(self.active_connections)}")

    async def broadcast(self, message: str):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logging.error(f"Error broadcasting to websocket: {e}")
                disconnected.append(connection)

        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

manager = ConnectionManager()

class ChatMessage(BaseModel):
    message: str

@app.websocket("/ws/emotions")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            try:
                # Simulate emotions when EEG is not available
                if not eeg_stream or not hasattr(eeg_stream, 'current_emotion'):
                    emotion = "Simulation Mode"
                    confidence = 0.0
                else:
                    emotion = eeg_stream.current_emotion or "Analyzing..."
                    confidence = eeg_stream.current_confidence or 0.0
                
                # Send emotion update
                await websocket.send_json({
                    "emotion": emotion,
                    "confidence": confidence,
                    "relaxed_confidence": 1 - confidence if emotion == "stressed" else confidence,
                    "stressed_confidence": confidence if emotion == "stressed" else 1 - confidence,
                    "timestamp": time.time()
                })
                
                # Wait before sending next update
                await asyncio.sleep(5)
            except Exception as e:
                logging.error(f"Error in websocket loop: {e}")
                break
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logging.error(f"Unexpected websocket error: {e}")
        manager.disconnect(websocket)

@app.post("/api/chat")
async def chat_endpoint(message: ChatMessage):
    try:
        if not message.message.strip():
            return JSONResponse(
                status_code=400,
                content={"error": "Message cannot be empty"}
            )

        if not eeg_stream:
            return JSONResponse(
                status_code=200,
                content={
                    "response": "I'm running in simulation mode without EEG. I'll still try to help you! What's on your mind?"
                }
            )

        # Generate response using the chatbot
        response = chatbot.generate_response(
            message.message,
            current_emotion,
            current_confidence
        )
        return JSONResponse(
            status_code=200,
            content={"response": response}
        )
    except Exception as e:
        logging.error(f"Error in chat endpoint: {e}")
        logging.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error occurred. Please try again."}
        )

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    try:
        status = {
            "status": "healthy",
            "mode": "simulation" if not eeg_stream else "normal",
            "connections": len(manager.active_connections),
            "timestamp": time.time()
        }
        logging.info(f"Health check: {status}")
        return JSONResponse(status_code=200, content=status)
    except Exception as e:
        error_msg = f"Error in health check: {e}"
        logging.error(error_msg)
        logging.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": error_msg}
        )

@app.get("/api/emotion")
async def get_emotion():
    try:
        if not eeg_stream or not hasattr(eeg_stream, 'current_emotion'):
            return JSONResponse(
                status_code=200,
                content={
                    "emotion": "Simulation Mode",
                    "confidence": 0.0,
                    "timestamp": time.time()
                }
            )
        emotion = eeg_stream.current_emotion or "Analyzing..."
        confidence = eeg_stream.current_confidence or 0.0
        return JSONResponse(
            status_code=200,
            content={
                "emotion": emotion,
                "confidence": confidence,
                "timestamp": time.time()
            }
        )
    except Exception as e:
        logging.error(f"Error in /api/emotion endpoint: {e}")
        logging.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={
                "emotion": "error",
                "confidence": 0.0,
                "timestamp": time.time(),
                "error": str(e)
            }
        )

@app.post("/api/user-info")
async def submit_user_info(request: Request):
    try:
        data = await request.json()
        # Log the received user info for debugging
        logging.info(f"Received user info: {data}")
        # You can add logic here to store/process user info as needed
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "User information saved successfully"
            }
        )
    except Exception as e:
        logging.error(f"Error in /api/user-info endpoint: {e}")
        logging.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Failed to save user information: {str(e)}"
            }
        )

if __name__ == "__main__":
    logging.info("Starting FastAPI server...")
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=5000,
        log_level="info",
        access_log=True
    ) 