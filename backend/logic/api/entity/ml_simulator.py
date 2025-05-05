import asyncio
from logic.api.services.board_service import BoardService

board_service = BoardService()

async def fake_ml_moves() -> None:
  """ Simulate a chess game using hardcoded moves. """
  moves = ["e4", "d5", "exd5", "Nc6", "Bb5", "a6"]
  for move in moves:
    await board_service.send_move(1, move)
    await asyncio.sleep(2)
    
async def simulate_multiple_fake_ml_moves() -> None:
  """ Simulate multiple boards playing at once (concurrent moves). """
  
  games = {
    # 2: ["d4", "d7", "c4", "e6", "Nc3", "Nf6", "Bg5", "Be7", "e3", "O-O", "Nf3", "h6", "Bh4", "b6", "cxd5", "Nxd5"], # d4, d5, c4
    1: ["e4", "e5", "Qh5", "Nf6", "d3", "Nxh5"],
    2: ["a4", "d5", "Ra2", "Bh3", "a5", "b6", "gxh3", "bxa5", "Rxa5", "Nc6", "Bg2", "Nxa5", "Bxd5", "Qxd5", "Nf3", "Nc4", "Na3", "Rd8"]
  }
  
  async def send_moves(board_id, moves):
    for move in moves:
      await asyncio.sleep(3)
      await board_service.send_move(board_id, move)
      
  tasks = [send_moves(board_id, moves) for board_id, moves in games.items()]
  await asyncio.gather(*tasks)
