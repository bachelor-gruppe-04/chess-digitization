from typing import List, Optional

class MovesData:
    def __init__(self, sans: List[str], from_positions: List[int], to_positions: List[int], targets: List[int]):
        self.sans = sans
        self.from_positions = from_positions
        self.to_positions = to_positions
        self.targets = targets


class MovesPair:
    def __init__(self, move1: MovesData, move2: Optional[MovesData] = None, moves: Optional[MovesData] = None):
        self.move1 = move1
        self.move2 = move2
        self.moves = moves
