from fastapi import APIRouter, Path, HTTPException
from fastapi.responses import StreamingResponse
import logic.api.services.board_storage as storage

router = APIRouter()

@router.get("/video/{id}")
def video_feed(id: int = Path(..., ge=1)) -> StreamingResponse:
  """Dynamic video stream from multiple webcams. 
  
  Args:
    id (int): Board ID
  """
  if id not in storage.boards:
    raise HTTPException(404, f"Board {id} not found.")
  
  return StreamingResponse(
    storage.boards[id].camera.generate_frames(),
    media_type="multipart/x-mixed-replace; boundary=frame"
  )
