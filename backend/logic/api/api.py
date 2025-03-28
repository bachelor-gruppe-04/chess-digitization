import asyncio
from fastapi import FastAPI, WebSocket, Path
from fastapi.responses import StreamingResponse
from typing import List, Dict
from board import Board
from camera import Camera

app = FastAPI()

clients = []
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

@app.websocket("/moves/{board_id}")
async def websocket_endpoint(websocket: WebSocket, board_id: int) -> None:
  """ Sends chess moves and history.
  
  Args:
    websocket (WebSocket): WebSocket connection
  """
  await websocket.accept()
  if board_id not in boards:
    await websocket.close()
    return
    
  boards[board_id].clients.append(websocket)
  try:
    for move in boards[board_id].move_history:
      await websocket.send_text(move)
    while True:
      await websocket.receive_text()
  except:
    boards[board_id].clients.remove(websocket)

async def send_move(board_id: int, move: str) -> None:
  """ Send a chess move to all clients.

  Args:
    move (str): Chess move
  """
  boards[board_id].move_piece(move)
  for client in boards[board_id].clients:
    await client.send_text(move)

async def reset_game(board_id: int) -> None:
  """ Reset the chess game. """
  boards[board_id].move_history = []
  for client in boards[board_id].clients:
    await client.send_text("RESET")
    
async def fake_ml_moves() -> None:
  """ Simulate a chess game using hardcoded moves. """
  moves = ["e4", "e5", "Rs4", "Nc6", "Bb5", "a6"]
  for board_id in boards:
    for move in moves:
      await asyncio.sleep(3)
      print(f"Sending move: {move} to board {board_id}")
      await send_move(board_id, move)
    
  # await asyncio.sleep(5)
  # print("Resetting game...")
  # await reset_game(board_id)
  # moves = ["a3", "g6", "c4", "Bg7", "d4", "Nf6"]
  
  # for board_id in boards:
  #   for move in moves:
  #     await asyncio.sleep(3)
  #     print(f"Sending move: {move} to board {board_id}")
  #     await send_move(board_id, move)
    
@app.on_event("startup")
async def start_fake_moves() -> None:
  """ Start the fake ML moves. """
  asyncio.create_task(fake_ml_moves())
