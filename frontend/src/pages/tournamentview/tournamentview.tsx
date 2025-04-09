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
    <div className='tournament-view'>
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
          <TableRow boardNumber={3} whitePlayer="Player E" blackPlayer="Player F" />
          <TableRow boardNumber={4} whitePlayer="Player G" blackPlayer="Player H" />
          <TableRow boardNumber={5} whitePlayer="Player I" blackPlayer="Player J" />
          <TableRow boardNumber={6} whitePlayer="Player K" blackPlayer="Player L" />
          <TableRow boardNumber={7} whitePlayer="Player M" blackPlayer="Player N" />
          <TableRow boardNumber={8} whitePlayer="Player O" blackPlayer="Player P" />
        </tbody>
      </table>
    </div>
  );
}

export default TournamentView;
