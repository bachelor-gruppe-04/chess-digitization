import "./Tile.css";

/**
 * Tile Component
 * 
 * This component represents an individual tile on a chessboard. The tile color is 
 * determined based on whether the given number is even or odd.
 */

interface Props {
  number: number;
}

function Tile({ number }: Props) {
  const tileColor = number % 2 === 0 ? "black-tile" : "white-tile";

  return <div className={`tile ${tileColor}`}></div>;
}

export default Tile;