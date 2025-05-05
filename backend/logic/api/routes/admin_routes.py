from fastapi import APIRouter
from logic.api.services.board_service import BoardService
import logic.api.services.board_storage as storage

router = APIRouter()

board_service = BoardService()

@router.post("/reset/{board_id}")
async def reset_board(board_id: int):
  """ Reset a specific board. 
  
  Args:
    board_id (int): Board ID
  """
  await board_service.reset_game(board_id)
  return {"status": "reset", "board": board_id}

@router.post("/reset_all")
async def reset_all_boards():
  """ Reset all boards. """
  await board_service.reset_all_games()
  return {"status": "all boards reset"}

@router.get("/boards")
async def list_boards():
  """ List all boards. """
  ids = list(storage.boards.keys())
  return {"board_count": len(ids), "boards": ids}