import './tournamentview.css';
import { NavLink } from 'react-router-dom';

/**
 * TournamentView Component
 * 
 * This component serves as a container for rendering multiple chess boards
 * and manages the move history state. 
 */

function TournamentView() {
  return (
    <div>
      <h1>Tournament View</h1>
      <div className="board-links">
        <NavLink
          to="/board/1">
          Go to Board 1
        </NavLink>
        <br />
        <NavLink
          to="/board/2">
          Go to Board 2
        </NavLink>
      </div>
    </div>
  );
}

export default TournamentView;
