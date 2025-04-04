import Chessboard, { ChessboardHandle } from '../../components/chessboard/chessboard';
import PGN from '../../components/pgn/pgn';
import Camera from '../../components/camera/camera';
import './boardview.css';
import { useRef, useEffect, useState } from "react";

/**
/**
 * BoardView Component
 *
 * This layout component arranges a single chessboard along with a PGN move list and camera feed.
 * It integrates child components and synchronizes the move history from the chessboard.
 */

/**
 * Unique ID to identify the board
 */
interface BoardViewProps {
  id: number;
}

function BoardView({ id }: BoardViewProps) {
  const boardRef = useRef<ChessboardHandle>(null); // Ref to access Chessboard's imperative handle (exposes getMoves method)
  const [moves, setMoves] = useState<string[]>([]); // State to hold the current list of moves in algebraic notation (SAN)

  /**
   * useEffect sets up a polling interval to fetch the latest moves
   * from the chessboard via the exposed `getMoves()` method.
   * This ensures the PGN view stays in sync with the board.
   */
  useEffect(() => {
    const interval = setInterval(() => {
      if (boardRef.current) {
        const newMoves = boardRef.current.getMoves();
        setMoves(newMoves);
      }
    }, 500); // Fetch moves every 500ms

    return () => clearInterval(interval); // Clear the interval when component unmounts
  }, []);

  return (
    <div className="table-view">
      <div className='left-wrapper'>
          <div className='camera-wrapper'>
            <Camera id={id} />
          </div>
          <div className="pgn-wrapper">
            <PGN moves={moves} />
          </div>
        </div>
        <div className="chessboard-wrapper">
          <Chessboard ref={boardRef} id={id} />
        </div>
    </div>
  );
}

export default BoardView;
