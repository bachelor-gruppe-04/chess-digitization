from fastapi import APIRouter, WebSocket
from services.board_service import boards

router = APIRouter()

@router.websocket("/moves/{board_id}")
async def websocket_endpoint(websocket: WebSocket, board_id: int) -> None:
  """ Sends chess moves and history.
  
  Args:
    websocket (WebSocket): WebSocket connection
    board_id (int): Board ID
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
