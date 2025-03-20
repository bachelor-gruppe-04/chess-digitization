import Tile from "../tile/tile";
import "./chessboard.css";

/**
 * Chessboard Component
 * 
 * This component renders a simple chessboard. The board consists of 64 alternating 
 * black and white tiles, generated dynamically.
 */

interface Piece {
  image: string;
  x: number;
  y: number;
}

const pieces: Piece[] = [];

// Black pieces
pieces.push({image: "assets/images/rook_b.png", x: 0, y: 7}); // Rook
pieces.push({image: "assets/images/rook_b.png", x: 7, y: 7}); // ROok
pieces.push({image: "assets/images/knight_b.png", x: 1, y: 7}); // knight
pieces.push({image: "assets/images/knight_b.png", x: 6, y: 7}); // knight
pieces.push({image: "assets/images/bishop_b.png", x: 2, y: 7}); // Bishop
pieces.push({image: "assets/images/bishop_b.png", x: 5, y: 7}); // Bishop
pieces.push({image: "assets/images/queen_b.png", x: 3, y: 7}); // Queen
pieces.push({image: "assets/images/king_b.png", x: 4, y: 7}); // King

// Black pawns
for(let i = 0; i < 8; i++) {
  pieces.push({image: "assets/images/pawn_b.png", x: i, y: 6});
}

// White pieces
pieces.push({image: "assets/images/rook_w.png", x: 0, y: 0}); // Rook
pieces.push({image: "assets/images/rook_w.png", x: 7, y: 0}); // Rook
pieces.push({image: "assets/images/knight_w.png", x: 1, y: 0}); // knight
pieces.push({image: "assets/images/knight_w.png", x: 6, y: 0}); // knight
pieces.push({image: "assets/images/bishop_w.png", x: 2, y: 0}); // Bishop
pieces.push({image: "assets/images/bishop_w.png", x: 5, y: 0}); // Bishop
pieces.push({image: "assets/images/queen_w.png", x: 3, y: 0}); // Queen
pieces.push({image: "assets/images/king_w.png", x: 4, y: 0}); // King

// White pawns
for(let i = 0; i < 8; i++) {
  pieces.push({image: "assets/images/pawn_w.png", x: i, y: 1});
}

function Chessboard() {
  const verticalAxis = ["1", "2", "3", "4", "5", "6", "7", "8"];
  const horizontalAxis = ["a", "b", "c", "d", "e", "f", "g", "h"];
  

  let board = [];

  /**
   * Generate the chessboard by iterating over rows (vertical axis) and columns 
   * (horizontal axis). The outer loop starts from the highest row (8) down to 
   * 1 to maintain traditional chess notation. The inner loop iterates through 
   * each column to create tiles. The tile color is determined based on the sum 
   * of row and column indices.
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