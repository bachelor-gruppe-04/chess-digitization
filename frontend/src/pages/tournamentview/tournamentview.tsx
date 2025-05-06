import './tournamentview.css';

import { useEffect, useState } from 'react';
import TableRow from '../../components/tableRow/tableRow';

/**
 * TournamentView Component
 *
 * Fetches a list of active board IDs from the backend and displays a tournament table.
 * Each row represents a chess game with mocked player names ("White #id" / "Black #id").
 *
 * If the data is loading, a skeleton loader is shown.
 * If an error occurs while fetching data, a user-friendly error message is displayed.
 */

function TournamentView() {
  const [boards, setBoards] = useState<number[]>([]); // Stores array of board IDs fetched from backend
  const [error, setError] = useState<string>(); // Tracks any fetch-related errors

    /**
   * Update the browser tab title
   */
    useEffect(() => {
      document.title = `ChessCamera`;
    });
  
  /**
   * On component mount, fetch the list of board IDs from the backend.
   * Endpoint: GET http://localhost:8000/boards
   *
   * Expected response: { boards: [1, 2, 3, ...] }
   * Handles malformed payloads and network errors.
   */
  useEffect(() => {
    fetch('http://localhost:8000/boards')
      .then(res => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then(data => {
        if (!Array.isArray(data.boards)) {
          throw new Error('Invalid payload: missing "boards" array');
        }
        setBoards(data.boards);
      })
      .catch(err => {
        console.error(err);
        setError(err.message);
      });
  }, []);

  /**
   * Render error state if fetch failed.
   */
  if (error) {
    return (
      <div className='tournament-view'>
        <div className="heading">Tournament<span> View</span></div>
        <div className="error-cell">Error loading boards: {error}</div>
      </div>
    );
  }

  return (
    <div className="tournament-view">
      <div className="heading">Tournament<span> View</span></div>

      {boards.length > 0 ? (
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
            {boards.map(id => (
              <TableRow
                key={id}
                boardNumber={id}
                whitePlayer={`White #${id}`}
                blackPlayer={`Black #${id}`}
              />
            ))}
          </tbody>
        </table>
      ) : (
        <div className="table-skeleton">
          {Array.from({ length: 8 * 4 }).map((_, i) => (
            <div key={i} className="skeleton-cell" />
          ))}
        </div>
      )}
    </div>
  );
}

export default TournamentView;