import './tableRow.css';

import { NavLink } from 'react-router-dom';

/**
 * TableRow Component
 *
 * Renders a single row in a tournament table, displaying:
 * - Board number
 * - Player names
 * - A link to the live board view
 *
 * The "LIVE" link uses React Router's `NavLink` to navigate to the specific board's view.
 */

/**
 * Props for the TableRow component
 * - `boardNumber`: Unique board identifier (used for navigation)
 * - `whitePlayer`: Name of the player playing as white
 * - `blackPlayer`: Name of the player playing as black
 */
interface TableRowProps {
  boardNumber: number;
  whitePlayer: string;
  blackPlayer: string;
}

function TableRow({ boardNumber, whitePlayer, blackPlayer }: TableRowProps) {
  return (
    <tr className="table-row">
      <td>{boardNumber}</td>
      <td>{whitePlayer}</td>
      <td>{blackPlayer}</td>
      <td>
        <NavLink to={`/board/${boardNumber}`} className="live-button">
          LIVE
        </NavLink>
      </td>
    </tr>
  );
}

export default TableRow;