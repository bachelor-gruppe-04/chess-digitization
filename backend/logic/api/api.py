import asyncio

from fastapi import FastAPI, WebSocket, Path
from fastapi.responses import StreamingResponse
from typing import List, Dict
from board import Board
from camera import Camera

app = FastAPI()

clients = []
move_history: List[str] = []
# camera_sources: Dict[int, Camera] = {i: Camera(i) for i in range(3)}
boards: Dict[int, Board] = {i: Board(i) for i in range(3)}

@app.get("/video/{id}")
def video_feed(id: int = Path(..., ge=0, le=(len(boards) - 1))) -> StreamingResponse:
  """Dynamic video stream from multiple webcams. """
  if id in boards:
    return StreamingResponse(
      boards[id].get_camera().generate_frames(), 
      media_type="multipart/x-mixed-replace; boundary=frame"
    )
  return {"error": "Invalid camera ID"}

@app.websocket("/moves/0")
async def websocket_endpoint(websocket: WebSocket) -> None:
  """ Sends chess moves and history.
  
  Args:
    websocket (WebSocket): WebSocket connection
  """
  await websocket.accept()
  clients.append(websocket)
  try:
    for move in move_history:
      await websocket.send_text(move)
    while True:
      await websocket.receive_text()
  except:
    clients.remove(websocket)

async def send_move(move: str) -> None:
  """ Send a chess move to all clients.

  Args:
    move (str): Chess move
  """
  move_history.append(move)
  for client in clients:
    await client.send_text(move)

async def reset_game() -> None:
  """ Reset the chess game. """
  global move_history
  move_history = []
  for client in clients:
    await client.send_text("RESET")
    
async def fake_ml_moves() -> None:
  """ Simulate a chess game using hardcoded moves. """
  moves = ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"]
  for move in moves:
    await asyncio.sleep(3)
    print(f"Sending move: {move}")
    await send_move(move)
  await asyncio.sleep(5)
  print("Resetting game...")
  await reset_game()
  moves = ["a3", "g6", "c4", "Bg7", "d4", "Nf6"]
  for move in moves:
    await asyncio.sleep(3)
    print(f"Sending move: {move}")
    await send_move(move)
    
@app.on_event("startup")
async def start_fake_moves() -> None:
  """ Start the fake ML moves. """
  asyncio.create_task(fake_ml_moves())
