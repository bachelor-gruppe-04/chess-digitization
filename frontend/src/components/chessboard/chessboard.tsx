import { useEffect, useState } from "react";
import { Chess } from "chess.ts";
import Tile from "../tile/tile";
import "./chessboard.css";
import { useWebSocket } from "../../hooks/useWebSocket";


/**
 * Chessboard Component
 *
 * This component renders a chessboard UI and manages the game state using the `chess.ts` library.
 * It listens to a WebSocket stream for move updates and reflects them visually on the board.
 * The board is initialized using a FEN string and updated as new valid moves arrive.
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
 * NOTE: This feature is unused and marked for removal.
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
 * NOTE: This feature is unused and marked for removal.
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
 * NOTE: This feature is unused and marked for removal.
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

interface ChessboardProps {
  // Function to update the list of chess moves in the parent component
  setMoves: React.Dispatch<React.SetStateAction<string[]>>;
}

function Chessboard({ setMoves }: ChessboardProps) {
  const [pieces, setPieces] = useState<Piece[]>([]);
  const chess = new Chess();
  const moves = useWebSocket(`ws://localhost:8000/moves`);


  /**
   * Initializes the board on first render using the current FEN from `chess.ts`.
   * Also exposes a helper `makeMove` function to the browser console for debugging.
   */

  useEffect(() => {
    setPieces(generatePositionFromFen(chess.fen()));

    // Expose move handler for console
    (window as any).makeMove = (notation: string) => {
      const move = chess.move(notation);
      if (move) {
        setPieces(generatePositionFromFen(chess.fen()));
        setMoves?.((prev) => [...prev, move.san]); // Add move in Standard Algebraic Notation
      } else {
        console.warn("Illegal move:", notation);
      }
    };
  }, []);

  /**
   * Applies incoming moves from the WebSocket to the board.
   * Skips illegal or repeated moves, and syncs the piece state accordingly.
   */

  useEffect(() => {
    // Guard clause: skip if no moves to process
    if (!moves || moves.length === 0) return;
  
    // Clone move array to avoid mutation
    const newMoves = moves.slice();
    console.log(newMoves);
  
    // Attempt to apply each move to the chess instance
    newMoves.forEach((notation) => {
      const move = chess.move(notation);
      if (move) {
        setMoves((prev) => {
          // Prevent duplicate SANs
          if (prev.includes(move.san)) return prev;
          return [...prev, move.san];
        });
      } else {
        console.warn("Illegal or duplicate move from WebSocket:", notation);
      }
    });
  
     // Update board pieces after moves have been applied
    if (newMoves.length > 0) {
      setPieces(generatePositionFromFen(chess.fen()));
    }
  }, [moves]);

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
