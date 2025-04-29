from fastapi import APIRouter

from logic.api.services.board_service import BoardService
# from logic.api.services.board_service import reset_game, reset_all_games

router = APIRouter()

board_service = BoardService()

@router.post("/reset/{board_id}")
async def reset_board(board_id: int):
  await board_service.reset_game(board_id)
  return {"status": "reset", "board": board_id}

@router.post("/reset_all")
async def reset_all_boards():
  await board_service.reset_all_games()
  return {"status": "all boards reset"}