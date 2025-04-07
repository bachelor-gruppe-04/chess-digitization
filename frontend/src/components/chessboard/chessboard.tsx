import { forwardRef, useImperativeHandle, useEffect, useState } from "react";
import { Chess } from "chess.ts";
import Tile from "../tile/tile";
import "./chessboard.css";
import { useWebSocket } from "../../hooks/useWebSocket";


/**
 * Chessboard Component
 *
 * This component renders an interactive chessboard and manages game state using the `chess.ts` library.
 * It consumes a WebSocket stream to receive and apply moves, and exposes its internal move list to parent components.
 */

interface Piece {
  image: string;
  x: number;
  y: number;
}

/**
 * Converts a FEN string into an array of piece objects.
 * Each object includes image path and board coordinates.
 * Used to render pieces from the current game state.
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
        const image = `/assets/images/${pieceMap[char]}.png`;
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
  id: number;
}

/**
 * Unique ID to identify the board
 */
export interface ChessboardHandle {
  getMoves: () => string[];
}

const Chessboard = forwardRef<ChessboardHandle, ChessboardProps>(({ id }, ref) => {
  const [pieces, setPieces] = useState<Piece[]>([]); // Current piece layout
  const chess = new Chess(); // Chess game instance
  const [moveList, setMoveList] = useState<string[]>([]); // Local move history
  const moves = useWebSocket(`ws://localhost:8000/moves/${id}`); // WebSocket listener for this board

  /**
   * Expose the list of SAN moves to parent components via ref.
   */
  useImperativeHandle(ref, () => ({
    getMoves: () => moveList,
  }));

  /**
   * Initializes the board on first render using the current FEN from `chess.ts`.
   * Also exposes a helper `makeMove` function to the browser console for debugging.
   * 
   * NOTE: Used for testing the board manually through console input.
   */
  useEffect(() => {
    setPieces(generatePositionFromFen(chess.fen()));

    // Expose move handler for console
    (window as any).makeMove = (notation: string) => {
      const move = chess.move(notation);
      if (move) {
        setPieces(generatePositionFromFen(chess.fen()));
        setMoveList((prev) => [...prev, move.san]);
      } else {
        console.warn("Illegal move:", notation);
      }
    };
  }, []);

  /**
   * Listens for new moves from the WebSocket and applies them sequentially.
   * Filters out illegal moves and syncs the internal state (move list and pieces).
   */
  useEffect(() => {
    if (!moves || moves.length === 0) return;

    chess.reset();
    const validSanMoves: string[] = [];
  
    // Attempt to apply each move to the chess instance
    moves.forEach((notation) => {
      const move = chess.move(notation);
      if (move) {
        validSanMoves.push(move.san);
      } else {
        console.warn("Illegal move from WebSocket:", notation);
      }
    });
  
    setMoveList(validSanMoves);

    if (validSanMoves.length > 0) {
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
    
    <div id="chessboard">
      {board}
    </div>
  );
});

export default Chessboard;
