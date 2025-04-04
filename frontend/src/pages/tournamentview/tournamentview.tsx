import './tournamentview.css';
import BoardView from '../boardview/boardview';

/**
 * TournamentView Component
 * 
 * This component serves as a container for rendering multiple chess boards
 * and manages the move history state. 
 */

function TournamentView() {

  return (
    <div>
      <BoardView id={1}/>
    </div>
  );
}

export default TournamentView;
