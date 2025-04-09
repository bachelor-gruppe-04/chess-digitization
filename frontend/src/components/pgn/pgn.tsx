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
    <table className="pgn-table">
      <thead>
        <tr>
          <th>#</th>
          <th>White</th>
          <th>Black</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row, index) => (
          <tr key={index}>
            <td>{row.turn}.</td>
            <td className={index * 2 === moves.length - 1 ? "highlight" : ""}>
              {row.white}
            </td>
            <td className={index * 2 + 1 === moves.length - 1 ? "highlight" : ""}>
              {row.black}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default PGN;