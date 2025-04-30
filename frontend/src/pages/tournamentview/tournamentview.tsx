import './tournamentview.css';

import TableRow from '../../components/tableRow/tableRow';
import { boards } from '../../data/boards';

/**
 * TournamentView Component
 *
 * This component serves as the main hub for navigating between multiple live chess boards
 * during a tournament. It displays a table listing all active games.
 *
 * Each row shows:
 * - Board number
 * - White player (name and rating)
 * - Black player (name and rating)
 * - A link to the live game view
 *
 * Data is dynamically loaded from the `boards` array for scalability.
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
          {boards.map((board) => (
            <TableRow 
              key={board.id} 
              boardNumber={board.id} 
              whitePlayer={`${board.whitePlayer} (${board.whiteRating})`} 
              blackPlayer={`${board.blackPlayer} (${board.blackRating})`} 
            />
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default TournamentView;
