import './tournamentview.css';
import { useState } from 'react';
import BoardView from '../boardview/boardview';

/**
 * TournamentView Component
 * 
 * This component serves as a container for rendering multiple chess boards
 * and manages the move history state. 
 */


function TournamentView() {
  // State to hold the list of chess moves (in algebraic notation)
  const [moves, setMoves] = useState<string[]>([]);

  return (
    <div>
      <BoardView moves={moves} setMoves={setMoves} id={1}/>
    </div>
  );
}

export default TournamentView;
