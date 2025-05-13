import './boardview.css';

import Camera from '../../components/camera/camera';
import Chessboard, { ChessboardHandle } from '../../components/chessboard/chessboard';
import PGN from '../../components/pgn/pgn';
import { useRef, useEffect, useState } from "react";
import { NavLink, useParams } from 'react-router-dom';
import EvalBar from '../../components/evalbar/evalbar';
import useEvaluation from '../../hooks/useEvaluation';
import { useLocation } from 'react-router-dom';

/**
 * BoardView Component
 *
 * Renders a complete view for a single board in a tournament.
 * Includes:
 * - Live camera feed of the physical board
 * - Interactive digital chessboard with synced game state
 * - PGN move list
 * - Evaluation bar showing Stockfish advantage
 * - PGN download functionality
 *
 * This component uses hooks and references to keep track of FEN strings, moves, and evaluation.
 */

function BoardView() {
  const pgnRef = useRef<HTMLDivElement>(null); // Ref to scroll the PGN list container
  const boardRef = useRef<ChessboardHandle>(null); // Ref to access Chessboard's imperative handle (exposes getMoves method)
  const [moves, setMoves] = useState<string[]>([]); // State to hold the current list of moves in algebraic notation (SAN)
  const [fen, setFen] = useState<string>(''); // State to hold the current FEN string
  const evaluation = useEvaluation(fen); // Fetch evaluation from Stockfish API based on the current FEN
  const { id } = useParams<{ id: string}>(); // Unique identifier for the board, used to connect to the correct data stream (e.g., WebSocket or camera).
  const location = useLocation();
  const { whitePlayer, blackPlayer } = location.state || {}; // Optional player names passed from route state

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

    if (boardRef.current) {
      const newFEN = boardRef.current?.getFEN();
      if (newFEN) {
        setFen(newFEN);
      }
    }
  }, [moves]);

  /**
   * Extract mate information from evaluation string (e.g., "M1", "-M2").
   * If not a mate situation, returns null.
   */
  const mate = (() => {
    if (evaluation && typeof evaluation === 'string' && evaluation.startsWith('M')) {
      return evaluation;
    }
    return null;
  })();
  
  /**
   * Update the browser tab title dynamically based on the board ID
   */
  useEffect(() => {
    document.title = `Board ${id} - ChessCamera`;
  }, [id]);

  /**
   * Generates and downloads a PGN file using current game info and headers.
   *
   * The user should be able to enter the tournament name at the start of the tournament.
   * The result should reflect the game status and display one of:
   * "1-0" (white wins), "0-1" (black wins), "1/2-1/2" (draw), or "In Progress".
   */
  const downloadPGN = () => {
    if (!boardRef.current) return;
  
    const moveText = boardRef.current.getPGN?.() ?? "";
  
    const date = new Date().toISOString().split("T")[0];
  
    const pgnHeaders = [
      `[Event "Ålesund Grand Prix 2025"]`,
      `[Date "${date}"]`,
      `[White "${whitePlayer || "White"}"]`,
      `[Black "${blackPlayer || "Black"}"]`,
      `[Result ""]`,
    ];
  
    const fullPGN = pgnHeaders.join("\n") + (moveText ? `\n\n${moveText}` : "");
  
    const blob = new Blob([fullPGN], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `game-${date}-board-${id}.pgn`;
    document.body.appendChild(a);
    a.click();
  
    setTimeout(() => {
      URL.revokeObjectURL(url);
      document.body.removeChild(a);
    }, 100);
  };

  return (
    <div className="table-view">
      <div className='left-wrapper'>
        <div className="top-controls">
          <div className='back-button'>
            <NavLink to="/">← Back to tournament overview</NavLink>
          </div>
          <div className="pgn-download-button">
            <button onClick={downloadPGN}>
              Download PGN
            </button>
          </div>
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