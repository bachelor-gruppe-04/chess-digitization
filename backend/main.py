import asyncio
from fastapi import FastAPI
from logic.api.routes import admin_routes, video_routes, websocket_routes
from logic.api.entity.ml_simulator import fake_ml_moves, simulate_multiple_fake_ml_moves
from logic.api.services.board_service import start_detector

app = FastAPI()

app.include_router(video_routes.router)
app.include_router(websocket_routes.router)
app.include_router(admin_routes.router)

@app.on_event("startup")
async def start_simulator():
  #asyncio.create_task(simulate_multiple_fake_ml_moves())
  asyncio.create_task(start_detector())
  pass
