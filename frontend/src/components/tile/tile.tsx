import "./Tile.css";

/**
 * Tile Component
 * 
 * This component represents an individual tile on a chessboard. The tile color is 
 * determined based on whether the given number is even or odd. An optional image 
 * (such as a chess piece) can be displayed on the tile.
 */

interface Props {
  number: number;
  image?: string;
}

function Tile({ number, image }: Props) {
  const tileColor = number % 2 === 0 ? "black-tile" : "white-tile";

  return (
    <div className={`tile ${tileColor}`}>
      {image && <img src={image} alt="Chess Piece" />}
    </div>
  );
}

export default Tile;