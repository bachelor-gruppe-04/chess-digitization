import Chessboard from '../components/chessboard/chessboard';
import PGN from '../components/pgn/pgn';
import './TableView.css';

/**
 * TableView Component
 *
 * This layout component arranges the chessboard and PGN move list side by side.
 * It passes move-related props to the corresponding child components:
 * - `Chessboard` handles the game state and piece interaction
 * - `PGN` displays the move history in a tabular format
 */

interface TableViewProps {
  moves: string[]; // Array of move strings (algebraic notation)
  setMoves: React.Dispatch<React.SetStateAction<string[]>>; // Function to update the list of moves
}

function TableView({ moves, setMoves }: TableViewProps) {
  return (
    <div className="table-view">
      <div className='left-wrapper'>
          <div className='camera-wrapper'>
            <p>Camera</p>
          </div>
          <div className="pgn-wrapper">
            <PGN moves={moves} />
          </div>
        </div>
        <div className="chessboard-wrapper">
          <Chessboard setMoves={setMoves} />
        </div>
    </div>
  );
}

export default TableView;
