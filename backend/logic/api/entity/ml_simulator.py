import asyncio
from services.board_service import send_move, reset_all_games

async def fake_ml_moves() -> None:
  """ Simulate a chess game using hardcoded moves. """
  moves = ["e4", "d5", "exd5", "Nc6", "Bb5", "a6"]
  for move in moves:
    await send_move(0, move)
    await asyncio.sleep(2)
      
  await asyncio.sleep(5)
  await reset_all_games()
    
async def simulate_multiple_fake_ml_moves() -> None:
  """ Simulate multiple boards playing at once (concurrent moves). """
  await asyncio.sleep(15)
  
  games = {
    1: ["d4", "d7", "c4", "e6", "Nc3", "Nf6", "Bg5", "Be7", "e3", "O-O", "Nf3", "h6", "Bh4", "b6", "cxd5", "Nxd5"], # d4, d5, c4
    2: ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "c3", "Nf6", "d3", "d6", "O-O", "O-O", "Nbd2", "a6", "Bb3", "Ba7"]
  }
  
  async def send_moves(board_id, moves):
    for move in moves:
      await asyncio.sleep(1)
      await send_move(board_id, move)
      
  tasks = [send_moves(board_id, moves) for board_id, moves in games.items()]
  await asyncio.gather(*tasks)
