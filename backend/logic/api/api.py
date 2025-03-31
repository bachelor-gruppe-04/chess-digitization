import asyncio
from fastapi import FastAPI, WebSocket, Path
from fastapi.responses import StreamingResponse
from typing import List, Dict
from board import Board
from camera import Camera

app = FastAPI()

clients = []
boards: Dict[int, Board] = {i: Board(i) for i in range(1)}

@app.get("/video/{id}")
def video_feed(id: int = Path(..., ge=0, le=(len(boards) - 1))) -> StreamingResponse:
  """Dynamic video stream from multiple webcams. """
  if id in boards:
    return StreamingResponse(
      boards[id].camera.generate_frames(), 
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
  checked_move, move_status = boards[board_id].process_move(move)
  if move_status:
    for client in boards[board_id].clients:
      # print(f"Sending move: {move} to board {board_id}")
      await client.send_text(checked_move)
      
      

async def reset_game(board_id: int) -> None:
  """ Reset the chess game of a board. """
  board: Board = boards[board_id]
  
  # board.move_history = []
  for client in board.clients:
    await client.send_text(board.reset_board())
    
    
    
async def reset_all_games() -> None:
  """ Reset the chess game to all boards. """
  for board_id in boards.keys():
    await reset_game(board_id)
    
    
    
async def simulate_multiple_fake_ml_moves() -> None:
  """ Simulate multiple boards playing at once (concurrent moves). """
  await asyncio.sleep(15)
  
  games = {
      0: ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7", "Re1", "b5", "Bb3", "d6", "c3", "O-O"],
      1: ["d4", "d5", "c4", "e6", "Nc3", "Nf6", "Bg5", "Be7", "e3", "O-O", "Nf3", "h6", "Bh4", "b6", "cxd5", "Nxd5"],
      2: ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "c3", "Nf6", "d3", "d6", "O-O", "O-O", "Nbd2", "a6", "Bb3", "Ba7"]
  }
  
  async def send_moves(board_id, moves):
    for move in moves:
      await asyncio.sleep(1)
      # print(f"Sending {move} to board {board_id}")
      await send_move(board_id, move)
      
  tasks = [send_moves(board_id, moves) for board_id, moves in games.items()]
  await asyncio.gather(*tasks)
  # print("All games finished.")

        
    
async def simulate_single_fake_ml_moves() -> None:
  """ Simulate a chess game using hardcoded moves. """
  moves = ["e4", "  d5 ", "   exd5", "Nc6  ", "Bb5", "a6"] # Legal moves: e4, d5, exd5, Nc6, Bb5, a6
  for move in moves:
    # await asyncio.sleep(3)
    await send_move(0, move)
    
  # await asyncio.sleep(5)
    
  # moves = ["b4", "e6", "Nf3", "c5", "c3", "cxb4"]  
  # for move in moves:
  #   # await asyncio.sleep(3)
  #   await send_move(1, move)
    
  # # await asyncio.sleep(5)
    
  # moves = ["a4", "a5", "b3", "b6", "Ra2", "Ra7"]  
  # for move in moves:
  #   # await asyncio.sleep(3)
  #   await send_move(2, move)
    
  await asyncio.sleep(10)
  
  await reset_all_games()
  
  await asyncio.sleep(10)
  
  moves = ["e4", "  d5 ", "   exd5", "Nc6  ", "Bb5", "a6"]  
  for move in moves:
    # await asyncio.sleep(3)
    await send_move(0, move)
  
  # moves = ["e4", "e5", "Rs4", "Nc6", "Bb5", "a6"]
  # for board_id in boards:
  #   for move in moves:
  #     await asyncio.sleep(3)
  #     await send_move(board_id, move)
    
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
  asyncio.create_task(simulate_single_fake_ml_moves())
