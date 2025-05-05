import './boardview.css';
import Camera from '../../components/camera/camera';
import Chessboard, { ChessboardHandle } from '../../components/chessboard/chessboard';
import PGN from '../../components/pgn/pgn';
import { useRef, useEffect, useState } from "react";
import { NavLink, useParams } from 'react-router-dom';
import EvalBar from '../../components/stockfish/evalbar';
import useEvaluation from '../../components/stockfish/stockfish';

/**
 * BoardView Component
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
  id: string;
}

function BoardView() {
  const pgnRef = useRef<HTMLDivElement>(null); // Ref to scroll the PGN list container
  const boardRef = useRef<ChessboardHandle>(null); // Ref to access Chessboard's imperative handle (exposes getMoves method)
  const [moves, setMoves] = useState<string[]>([]); // State to hold the current list of moves in algebraic notation (SAN)
  const [fen, setFen] = useState<string>(''); // State to hold the current FEN string
  const evaluation = useEvaluation(fen); // Fetch evaluation from Stockfish API based on the current FEN
  const { id } = useParams<{ id: string}>();

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

    const newFEN = boardRef.current?.getFEN();
    if (newFEN) {
      setFen(newFEN);
      console.log('New FEN:', newFEN);
    }
  }, [moves]);

  const mate = (() => {
    if (evaluation && typeof evaluation === 'string' && evaluation.startsWith('M')) {
      return evaluation; // Return the mate value directly, e.g., "M1", "-M2"
    }
    return null;  // No mate situation
  })();
  

  return (
    <div className="table-view">
      <div className='left-wrapper'>
        <div className='back-button'>
          <NavLink to="/">← Back to tournament overview</NavLink>
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
      <EvalBar 
        id={id} 
        evaluation={mate || evaluation}  // Pass the mate value if it exists, otherwise fallback to the regular evaluation
        mate={mate ? parseInt(mate.replace('M', '')) : null}  // Convert mate string to number for EvalBar
      />
    </div>
  );
}

export default BoardView;