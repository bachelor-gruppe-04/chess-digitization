from fastapi import APIRouter
from services.board_service import reset_game, reset_all_games

router = APIRouter()

@router.post("/reset/{board_id}")
async def reset_board(board_id: int):
  await reset_game(board_id)
  return {"status": "reset", "board": board_id}

@router.post("/reset_all")
async def reset_all():
  await reset_all_games()
  return {"status": "all boards reset"}
