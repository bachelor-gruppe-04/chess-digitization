import './boardview.css';

import Camera from '../../components/camera/camera';
import Chessboard, { ChessboardHandle } from '../../components/chessboard/chessboard';
import PGN from '../../components/pgn/pgn';
import { useRef, useEffect, useState } from "react";
import { NavLink } from 'react-router-dom';

/**
 * BoardView Component
 *
 * This layout component arranges a single chessboard alongside:
 * - A PGN move list (reflecting the current state of the game),
 * - A camera feed (useful for physical board views or livestreams),
 * - A navigation link back to the tournament overview.
 *
 * The component uses a `ref` to access the move history from the Chessboard component
 * and updates the local `moves` state at regular intervals to keep the PGN list in sync.
 */

/**
 * Props for BoardView
 * - `id`: Unique identifier for the board, used to connect to the correct data stream (e.g., WebSocket or camera).
 */
interface BoardViewProps {
  id: number;
}

function BoardView({ id }: BoardViewProps) {
  const pgnRef = useRef<HTMLDivElement>(null); // Ref to scroll the PGN list container
  const boardRef = useRef<ChessboardHandle>(null); // Ref to access Chessboard's imperative handle (exposes getMoves method)
  const [moves, setMoves] = useState<string[]>([]); // State to hold the current list of moves in algebraic notation (SAN)

   /**
   * Sets up a polling interval to sync moves from the Chessboard component.
   * Fetches move history every 500ms and updates the PGN view accordingly.
   */
  useEffect(() => {
    const interval = setInterval(() => {
      if (boardRef.current) {
        const newMoves = boardRef.current.getMoves();
        setMoves(newMoves);
      }
    }, 500); // Fetch moves every 500ms

    return () => clearInterval(interval);  // Clean up on component unmount
  }, []);

  /**
   * Automatically scroll the PGN list to the bottom whenever new moves are added.
   * This ensures that the most recent moves are always visible.
   */
  useEffect(() => {
    if (pgnRef.current) {
      pgnRef.current.scrollTop = pgnRef.current.scrollHeight;
    }
  }, [moves]);

  return (
    <div className="table-view">
      <div className='left-wrapper'>
          <div className='back-button'>
            <NavLink
              to="/">
              ← Back to tournament overview
            </NavLink>
          </div>
          <div className='camera-wrapper'>
            <Camera id={id} />
          </div>
          <div className="pgn-wrapper" ref={pgnRef}>
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
