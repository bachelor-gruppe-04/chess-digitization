import cv2
from typing import Generator

class Camera:
  """ Camera class to handle webcam. """
  
  def __init__(self, id: int) -> None:
    """ Initialize the camera object.

    Args:
      cam_id (int): Camera ID
    """
    self.cam_id = id
    self.camera = cv2.VideoCapture(id)
  


  def generate_frames(self) -> Generator[bytes, None, None]:
    """ Generate frames from the laptop webcam.
  
    Yields:
      Generator[bytes, None, None]: Image frames
    """
    while True:
      success, frame = self.camera.read()
      if not success:
        break
      _, buffer = cv2.imencode(".jpg", frame)
      frame_bytes = buffer.tobytes()
      yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")