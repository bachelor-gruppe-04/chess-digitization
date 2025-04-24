import "./pgn.css";

import { useEffect, useState } from "react";

/**
 * PGN Component
 *
 * Displays a list of chess moves using the PGN (Portable Game Notation) format.
 * Each turn consists of a pair of white and black moves. This component adapts to both
 * desktop (tabular view) and mobile (linear text view) for responsive display.
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

  const [isMobile, setIsMobile] = useState(false); // Tracks if the screen is mobile-sized

  /**
   * Handle responsive layout by listening to window resize events.
   * Sets `isMobile` to true when the screen width is 768px or less.
   */
  useEffect(() => {
    const handleResize = () => setIsMobile(window.innerWidth <= 768);
    handleResize();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  /**
   * If isMobile is true, the layout is set to inline text format
   */
  if (isMobile) {
    return (
      <div className="pgn-line">
        {rows.map((row, index) => (
          <span key={index}>
            {row.turn}.{" "}
            <span className={index * 2 === moves.length - 1 ? "highlight" : ""}>
              {row.white}
            </span>{" "}
            <span className={index * 2 + 1 === moves.length - 1 ? "highlight" : ""}>
              {row.black}
            </span>
          </span>
        ))}
      </div>
    );
  }

  /**
   * If isMobile is false, the layout is set to table format
   */
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
            <td className={index * 2 === moves.length - 1 ? "highlight" : ""}>{row.white}</td>
            <td className={index * 2 + 1 === moves.length - 1 ? "highlight" : ""}>{row.black}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}


export default PGN;