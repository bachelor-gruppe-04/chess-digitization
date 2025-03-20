import Tile from "../tile/tile";
import "./chessboard.css";

/**
 * Chessboard Component
 * 
 * This component renders a simple chessboard. The board consists of 64 alternating 
 * black and white tiles, generated dynamically.
 */

function Chessboard() {
  const verticalAxis = ["1", "2", "3", "4", "5", "6", "7", "8"];
  const horizontalAxis = ["a", "b", "c", "d", "e", "f", "g", "h"];

  let board = [];

  /**
   * Generate the chessboard by iterating over rows (vertical axis) and columns 
   * (horizontal axis). The outer loop starts from the highest row (8) down to 
   * 1 to maintain traditional chess notation. The inner loop iterates through 
   * each column to create tiles. The tile color is determined based on the sum 
   * of row and column indices.
   */
  for (let j = verticalAxis.length - 1; j >= 0; j--) {
    for (let i = 0; i < horizontalAxis.length; i++) {
      const number = j + i + 2;

      board.push(<Tile image="assets/images/pawn_b.png" number={number} />)
    }
  }

  return (
    <>
      <div id="chessboard">{board}</div>
    </>
  );
}

export default Chessboard