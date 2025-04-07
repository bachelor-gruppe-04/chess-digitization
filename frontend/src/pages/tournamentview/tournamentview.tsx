import './tournamentview.css';

import { NavLink } from 'react-router-dom';

/**
 * TournamentView Component
 * 
 * This component serves as a navigation hub for multiple chess boards in a tournament.
 * It renders links to individual board views using React Router's `NavLink` component.
 * 
 * Each link directs the user to a unique board route (e.g., `/board/1`, `/board/2`).
 * This is a scalable layout for managing and switching between boards in a tournament.
 */

function TournamentView() {
  return (
    <div>
      <h1>Tournament View</h1>

      {/* Navigation links to individual boards */}
      <div className="board-links">
        <NavLink to="/board/1">
          Go to Board 1
        </NavLink>
        <br />
        <NavLink to="/board/2">
          Go to Board 2
        </NavLink>
      </div>
    </div>
  );
}

export default TournamentView;
