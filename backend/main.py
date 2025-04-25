import asyncio
import threading
from fastapi import FastAPI
from logic.api.routes import admin_routes, video_routes, websocket_routes
from logic.api.entity.ml_simulator import fake_ml_moves, simulate_multiple_fake_ml_moves
from logic.api.services.board_service import start_detector
from logic.view.app_view import App
from logic.api.services.board_service import reset_game, reset_all_games
import logic.view.state as state

app = FastAPI()

app.include_router(video_routes.router)
app.include_router(websocket_routes.router)
app.include_router(admin_routes.router)

def start_gui():
  window = App(reset_game_func=reset_game, reset_all_games_func=reset_all_games)
  window.mainloop()

@app.on_event("startup")
async def main():
  asyncio.create_task(simulate_multiple_fake_ml_moves())
  # asyncio.create_task(start_detector())
  
  state.event_loop = asyncio.get_event_loop()
  gui_thread = threading.Thread(target=start_gui, daemon=True)
  gui_thread.start()
  
@app.get("/")
async def root():
  return {"message": "FastAPI + Tkinter running"}