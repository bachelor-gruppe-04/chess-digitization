import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import TableRow from '../../components/tableRow/tableRow';
import './tournamentview.css';

export default function TournamentView() {
  const [boards, setBoards] = useState<number[]>([]);
  const [error, setError] = useState<string>();

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
