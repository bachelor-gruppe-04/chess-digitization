import { forwardRef, useImperativeHandle, useEffect, useState } from "react";
import { Chess } from "chess.ts";
import Tile from "../tile/tile";
import "./chessboard.css";
import { useWebSocket } from "../../hooks/useWebSocket";


/**
 * Chessboard Component
 *
 * This component renders a full chessboard UI and manages game state using `chess.ts`.
 * It receives move updates via WebSocket, updates board position accordingly,
 * highlights the most recent move, and exposes its move list to parent components via ref.
 */

// Represents a single piece's position and image
interface Piece {
  image: string;
  x: number;
  y: number;
}

/**
 * Converts a FEN string into an array of Piece objects.
 * Maps chess notation characters to piece image paths and coordinates.
 * The y-coordinate is flipped to align with visual board orientation.
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

  // Parse the FEN rows (from top to bottom)
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


/**
 * Props for Chessboard
 * - `id`: Unique ID for the board, used in WebSocket connection
 */
interface ChessboardProps {
  id: number;
}

/**
 * Ref interface exposed by the Chessboard component.
 * Allows parent components to retrieve the current move list.
 */
export interface ChessboardHandle {
  getMoves: () => string[];
}

const Chessboard = forwardRef<ChessboardHandle, ChessboardProps>(({ id }, ref) => {
  const [pieces, setPieces] = useState<Piece[]>([]); // Current piece layout
  const chess = new Chess(); // Chess game instance
  const [moveList, setMoveList] = useState<string[]>([]); // Local move history
  const [lastMoveSquares, setLastMoveSquares] = useState<string[]>([]);  // Last move data
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

        let highlights: string[] = [];

        if (move.san === "O-O") {
          highlights = [move.from, move.color === "w" ? "h1" : "h8"];
        } else if (move.san === "O-O-O") {
          highlights = [move.from, move.color === "w" ? "a1" : "a8"];
        } else {
          highlights = [move.from, move.to];
        }

        setLastMoveSquares(highlights);
      } else {
        console.warn("Illegal move:", notation);
      }
    };
  }, []);

  /**
   * Apply all moves received from the WebSocket.
   * Valid moves are added to the move list and used to update the board.
   * The most recent move's squares are tracked for highlighting.
   */
  useEffect(() => {
    chess.reset();

    if (!moves || moves.length === 0){
      setMoveList([]);
      setLastMoveSquares([]);
      setPieces(generatePositionFromFen(chess.fen()));
      return;
    };

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
      const history = chess.history({ verbose: true });
      const move = history[history.length - 1];
      if (move) {
        let highlights: string[] = [];

        if (move.san === "O-O") {
          highlights = [move.from, move.color === "w" ? "h1" : "h8"];
        } else if (move.san === "O-O-O") {
          highlights = [move.from, move.color === "w" ? "a1" : "a8"];
        } else {
          highlights = [move.from, move.to];
        }

        setLastMoveSquares(highlights);
      }
      setPieces(generatePositionFromFen(chess.fen()));
    }
  }, [moves]);

  const verticalAxis = ["1", "2", "3", "4", "5", "6", "7", "8"];
  const horizontalAxis = ["a", "b", "c", "d", "e", "f", "g", "h"];

  let board = [];

   /**
   * Generates the full 8x8 chessboard as a grid of tiles.
   * Each tile receives a piece image if one exists at its (x, y) coordinate.
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

      const square = `${horizontalAxis[i]}${verticalAxis[j]}`;
      const isHighlighted = lastMoveSquares.includes(square);

      board.push(
        <Tile
          key={`${j},${i}`}
          image={image}
          number={number}
          highlight={isHighlighted}
        />
      );
    }
  }

  return (
    <div id="chessboard">
      {board}
    </div>
  );
});

export default Chessboard;
