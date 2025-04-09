import './tournamentview.css';

import TableRow from '../../components/tableRow/tableRow';

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
      <div className="heading">
        Tournament<span> View</span>
      </div>

      <table className="tournament-table">
        <thead>
          <tr>
            <th>Board</th>
            <th>White</th>
            <th>Black</th>
            <th>Game</th>
          </tr>
        </thead>
        <tbody>
          <TableRow boardNumber={1} whitePlayer="Player A" blackPlayer="Player B" />
          <TableRow boardNumber={2} whitePlayer="Player C" blackPlayer="Player D" />
        </tbody>
      </table>
    </div>
  );
}

export default TournamentView;
