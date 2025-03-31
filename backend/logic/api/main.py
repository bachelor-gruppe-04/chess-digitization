import asyncio
from fastapi import FastAPI
from routes import admin_routes, video_routes, websocket_routes
from entity.ml_simulator import fake_ml_moves, simulate_multiple_fake_ml_moves

app = FastAPI()

app.include_router(video_routes.router)
app.include_router(websocket_routes.router)
app.include_router(admin_routes.router)

@app.on_event("startup")
async def start_simulator():
  asyncio.create_task(simulate_multiple_fake_ml_moves())
