import "./pgn.css";

/**
 * PGN Component
 *
 * This component displays a list of chess moves in standard PGN (Portable Game Notation) format.
 * It accepts an array of move strings and formats them into rows with turn numbers,
 * alternating white and black moves.
 */

interface PGNProps {
  moves: string[]; // Array of algebraic move notation strings
}

function PGN({ moves }: PGNProps) {
  const rows = [];

  /**
   * Group the moves into rows of white and black turns.
   * Each pair of moves corresponds to one complete turn.
   * If there's an odd number of moves, the last black move will be empty.
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