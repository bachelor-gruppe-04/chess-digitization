import "./Tile.css";

/**
 * Tile Component
 *
 * This component represents an individual square on a chessboard.
 * It dynamically determines the tile color (black or white) based on the provided `number`,
 * which encodes both the row and column indices. If an `image` prop is passed, it displays
 * a chess piece on the tile using a background image.
 */

interface Props {
  number: number; // Used to determine tile color
  image?: string; // Optional image URL for a chess piece
}

function Tile({ number, image }: Props) {
   // Determine the tile color: even = black, odd = white
  if (number % 2 === 0) {
    return (
      <div className="tile black-tile">
        {/* If an image URL is provided, render the piece using a div with background image */}
        {image && <div style={{ backgroundImage: `url(${image})` }} className="chess-piece"></div>}
      </div>
    );
  } else {
    return (
      <div className="tile white-tile">
        {/* If an image URL is provided, render the piece using a div with background image */}
        {image && <div style={{ backgroundImage: `url(${image})` }} className="chess-piece"></div>}
      </div>
    );
  }
}

export default Tile;