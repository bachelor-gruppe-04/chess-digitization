from fastapi import APIRouter, Path, HTTPException
from fastapi.responses import StreamingResponse
from logic.api.services.board_storage import boards

router = APIRouter()

@router.get("/video/{id}")
def video_feed(id: int = Path(..., ge=1)) -> StreamingResponse:
  """Dynamic video stream from multiple webcams. """
  if id not in boards:
    raise HTTPException(404, f"No such board: {id}")
  
  return StreamingResponse(
    boards[id].camera.generate_frames(),
    media_type="multipart/x-mixed-replace; boundary=frame"
  )
