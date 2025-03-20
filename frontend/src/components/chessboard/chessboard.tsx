import Tile from "../tile/tile";
import "./chessboard.css";

/**
 * Chessboard Component
 * 
 * This component renders a simple chessboard with pieces. The board consists of 
 * 64 alternating black and white tiles, dynamically generated along with chess pieces.
 */

interface Piece {
  image: string;
  x: number;
  y: number;
}

const pieces: Piece[] = [];

/**
 * Initialize pieces for both black and white.
 * The loop runs twice: once for black (p = 0) and once for white (p = 1).
 * The pieces are placed according to standard chess rules.
 */

for(let p = 0; p < 2; p++) {
  const type = (p === 0) ? "b" : "w"; // Set piece color (black or white)
  const y = (p === 0) ? 7 : 0; // Define row position for each side
  
  pieces.push({image: `assets/images/rook_${type}.png`, x: 0, y: y})
  pieces.push({image: `assets/images/rook_${type}.png`, x: 7, y: y});
  pieces.push({image: `assets/images/knight_${type}.png`, x: 1, y: y});
  pieces.push({image: `assets/images/knight_${type}.png`, x: 6, y: y}); 
  pieces.push({image: `assets/images/bishop_${type}.png`, x: 2, y: y});
  pieces.push({image: `assets/images/bishop_${type}.png`, x: 5, y: y});
  pieces.push({image: `assets/images/queen_${type}.png`, x: 3, y: y});
  pieces.push({image: `assets/images/king_${type}.png`, x: 4, y: y});
}

/**
 * Initialize black and white pawns.
 * Each pawn is placed in its respective row (black on row 6, white on row 1).
 */

for(let i = 0; i < 8; i++) {
  pieces.push({image: "assets/images/pawn_b.png", x: i, y: 6});
}
for(let i = 0; i < 8; i++) {
  pieces.push({image: "assets/images/pawn_w.png", x: i, y: 1});
}

function Chessboard() {
  const verticalAxis = ["1", "2", "3", "4", "5", "6", "7", "8"];
  const horizontalAxis = ["a", "b", "c", "d", "e", "f", "g", "h"];
  

  let board = [];

  /**
   * Generate the chessboard by iterating over rows (vertical axis) and columns (horizontal axis).
   * The outer loop starts from the highest row (8) down to 1 to maintain traditional chess notation.
   * The inner loop iterates through each column to create tiles, and checks if a piece exists at the position.
   */

  for (let j = verticalAxis.length - 1; j >= 0; j--) {
    for (let i = 0; i < horizontalAxis.length; i++) {
      const number = j + i + 2;
      let image = undefined;

      pieces.forEach((p) => {
        if (p.x === i && p.y === j) {
          image = p.image;
        }
      });

      board.push(<Tile image={image}  number={number} />)
    }
  }

  return (
    <>
      <div id="chessboard">{board}</div>
    </>
  );
}

export default Chessboard