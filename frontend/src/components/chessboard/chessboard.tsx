import { useEffect, useState } from "react";
import { Chess } from "chess.ts";
import Tile from "../tile/tile";
import "./chessboard.css";

/**
 * Chessboard Component
 * 
 * This component renders a simple chessboard with pieces using the `chess.ts` library.
 * It supports piece initialization from FEN, move execution, and basic drag-and-drop.
 */

interface Piece {
  image: string;
  x: number;
  y: number;
}

let activePiece: HTMLElement | null = null;

/**
 * Handles grabbing a chess piece when the user clicks on it.
 * Sets the piece to absolute positioning and updates its location to follow the mouse.
 * 
 * Is not neccessary. To be removed.
 */

function grabPiece(e: React.MouseEvent) {
  const element = e.target as HTMLElement;

  if (element.classList.contains("chess-piece")) {
    const x = e.clientX - 50;
    const y = e.clientY - 50;
    element.style.position = "absolute";
    element.style.left = `${x}px`;
    element.style.top = `${y}px`;

    activePiece = element;
  }
}

/**
 * Moves the currently active chess piece based on mouse movement.
 * Continually updates the piece's position as the mouse moves.
 * 
 * Is not neccessary. To be removed.
 */

function movePiece(e: React.MouseEvent) {
  if (activePiece) {
    const x = e.clientX - 50;
    const y = e.clientY - 50;
    activePiece.style.position = "absolute";
    activePiece.style.left = `${x}px`;
    activePiece.style.top = `${y}px`;
  }
}

/**
 * Drops the currently dragged piece.
 * Resets the active piece to null.
 * 
 * Is not neccessary. To be removed.
 */

function dropPiece(_e: React.MouseEvent) {
  if (activePiece) {
    activePiece = null;
  }
}

/**
 * Converts a FEN string into a list of pieces with their image, x, and y coordinates.
 * Used to populate the board with the correct position from a game state.
 */

function generatePositionFromFen(fen: string): Piece[] {
  const board = fen.split(" ")[0];
  const rows = board.split("/");
  const pieceMap: { [key: string]: string } = {
    p: "pawn_b", P: "pawn_w",
    r: "rook_b", R: "rook_w",
    n: "knight_b", N: "knight_w",
    b: "bishop_b", B: "bishop_w",
    q: "queen_b", Q: "queen_w",
    k: "king_b", K: "king_w",
  };

  const pieces: Piece[] = [];

  // Parse each row from top (8) to bottom (1)
  for (let y = 0; y < rows.length; y++) {
    let x = 0;
    for (const char of rows[y]) {
      if (isNaN(Number(char))) {
        const image = `assets/images/${pieceMap[char]}.png`;
        pieces.push({ image, x, y: 7 - y }); // Flip y to match board orientation
        x++;
      } else {
        x += parseInt(char); // Empty tiles
      }
    }
  }

  return pieces;
}

function Chessboard() {
  const [pieces, setPieces] = useState<Piece[]>([]);
  const chess = new Chess();

  /**
   * On component mount, set initial board state using FEN.
   * Also expose a `makeMove` function on the `window` for manual move execution in console.
   */

  useEffect(() => {
    setPieces(generatePositionFromFen(chess.fen()));

    // Expose move handler for console
    (window as any).makeMove = (notation: string) => {
      if (chess.move(notation)) {
        setPieces(generatePositionFromFen(chess.fen()));
      } else {
        console.warn("Illegal move:", notation);
      }
    };
  }, []);

  const verticalAxis = ["1", "2", "3", "4", "5", "6", "7", "8"];
  const horizontalAxis = ["a", "b", "c", "d", "e", "f", "g", "h"];

  let board = [];

  /**
   * Generate the chessboard tiles and place pieces on them.
   * The outer loop iterates through each row from 8 to 1 (top to bottom).
   * The inner loop creates each tile and adds a piece if one exists at that coordinate.
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

      board.push(<Tile key={`${j},${i}`} image={image} number={number} />);
    }
  }

  return (
    <div
      onMouseMove={e => movePiece(e)}
      onMouseDown={e => grabPiece(e)}
      onMouseUp={e => dropPiece(e)}
      id="chessboard"
    >
      {board}
    </div>
  );
}

export default Chessboard;
