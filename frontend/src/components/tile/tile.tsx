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
  if (number % 2 === 0) {
    return (
      <div className="tile black-tile">
        {image && <div style={{ backgroundImage: `url(${image})` }} className="chess-piece"></div>}
      </div>
    );
  } else {
    return (
      <div className="tile white-tile">
        {image && <div style={{ backgroundImage: `url(${image})` }} className="chess-piece"></div>}
      </div>
    );
  }
}

export default Tile;