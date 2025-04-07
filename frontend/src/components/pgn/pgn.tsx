import "./pgn.css";

/**
 * PGN Component
 *
 * Displays a list of chess moves in Portable Game Notation (PGN) format.
 * It takes an array of move strings and formats them into rows,
 * each representing a turn with a white and black move.
 */

/**
 * Props for the PGN component
 * - `moves`: Array of algebraic notation moves (e.g., ["e4", "e5", "Nf3", "Nc6"])
 */
interface PGNProps {
  moves: string[];
}

function PGN({ moves }: PGNProps) {
  const rows = [];

  /**
   * Convert the flat array of moves into rows with turn numbers.
   * Each row contains a `white` move and a `black` move.
   * If the number of moves is odd, the last row's `black` cell will be empty.
   */
  for (let i = 0; i < moves.length; i += 2) {
    rows.push({
      turn: Math.floor(i / 2) + 1,
      white: moves[i],
      black: moves[i + 1] ?? "",
    });
  }

  return (
    <div className="pgn-list">
      <div className="pgn-header">
        <span>#</span>
        <span>White</span>
        <span>Black</span>
      </div>
      {/* Render each turn row with appropriate highlighting for the latest move */}
      {rows.map((row, index) => (
        <div className="pgn-row" key={index}>
          <span>{row.turn}.</span>
          <span className={index * 2 === moves.length - 1 ? "highlight" : ""}>
            {row.white}
          </span>
          <span className={index * 2 + 1 === moves.length - 1 ? "highlight" : ""}>
            {row.black}
          </span>
        </div>
      ))}
    </div>
  );
}

export default PGN;