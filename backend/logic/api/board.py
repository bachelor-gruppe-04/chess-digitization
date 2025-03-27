from camera import Camera
from typing import List

class Board:
    def __init__(self, id: int):
        self.id = id
        self.camera = Camera(id)
        self.move_history: List[str] = []

    def get_camera(self) -> Camera:
        return self.camera

    def get_move_history(self) -> List[str]:
        return self.move_history