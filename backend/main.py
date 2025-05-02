import asyncio
import threading
from fastapi import FastAPI
from logic.api.routes import admin_routes, video_routes, websocket_routes
from logic.api.routes.admin_routes import reset_board, reset_all_boards
from logic.api.entity.ml_simulator import fake_ml_moves, simulate_multiple_fake_ml_moves
from logic.view.app_view import App
import logic.view.state as state

app = FastAPI()

app.include_router(video_routes.router)
app.include_router(websocket_routes.router)
app.include_router(admin_routes.router)

def start_gui():
  window = App(reset_board_function=reset_board, reset_all_boards_function=reset_all_boards)
  window.mainloop()

@app.on_event("startup")
async def main():
  asyncio.create_task(simulate_multiple_fake_ml_moves())
  state.event_loop = asyncio.get_event_loop()
  gui_thread = threading.Thread(target=start_gui, daemon=True)
  gui_thread.start()