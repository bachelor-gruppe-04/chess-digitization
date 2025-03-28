import cv2
from typing import Generator

class Camera:
  """ Camera class to handle webcam. """
  
  def __init__(self, cam_id: int) -> None:
    """ Initialize the camera object.

    Args:
      cam_id (int): Camera ID
    """
    self.set_cam_id(cam_id)
    self.camera = cv2.VideoCapture(self.cam_id)
    
    
    
  def set_cam_id(self, cam_id: int) -> TypeError | None:
    if not isinstance(cam_id, int):
      raise TypeError(f"cam_id must be an integar, got {type(cam_id).__name__}")
    if cam_id < 0:
      raise ValueError(f"cam_id must be a positive number, got {cam_id}")
    
    self.cam_id = cam_id
    
    
    
  def get_cam_id(self) -> int:
    """ Get the camera ID. """
    return self.cam_id
  
  

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