import './gamePreview.css';

import { NavLink } from "react-router-dom";
import Chessboard from '../../components/chessboard/chessboard';
import { boards } from '../../data/boards';

/**
 * GamePreview Component
 *
 * Provides a visual overview of all games in progress during a tournament.
 * Each board is displayed as a card with:
 * - Board (table) number
 * - White and black players with their ratings
 * - A live mini-chessboard preview
 *
 * Clicking on a card navigates the user to the full board view using React Router's `NavLink`.
 * The layout is grid-based and responsive for scalability.
 */

function GamePreview() {
  return (
    <div className="game-preview">
      <div className="heading">
        Game<span> Preview</span>
      </div>
      <div className="boards-grid">
        {boards.map((board) => (
          <NavLink key={board.id} to={`/board/${board.id}`} className="board-card">
            <div className="table-label">Table {board.id}</div>
            <div className="player-info">
              <span>{board.whitePlayer} | {board.whiteRating}</span>
            </div>
            <div className="board-preview">
              <div className="chessboard-preview-wrapper">
                <Chessboard id={board.id} />
              </div>
            </div>
            <div className="player-info">
              <span>{board.blackPlayer} | {board.blackRating}</span>
            </div>
          </NavLink>
        ))}
      </div>
    </div>
  );
}
  
export default GamePreview;
