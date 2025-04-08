import "./tile.css";

/**
 * Tile Component
 *
 * Represents a single square on the chessboard.
 * The tile's color is determined by the `number` prop — even numbers create black tiles,
 * and odd numbers create white tiles. This logic ensures a checkered board pattern.
 *
 * If an `image` prop is provided (usually a chess piece), it is rendered on the tile
 * using a background image style.
 */

/**
 * Props for the Tile component
 * - `number`: Determines tile color based on parity (even = black, odd = white)
 * - `image`: Optional image URL for rendering a chess piece
 */
interface TileProps {
  number: number; 
  image?: string;
}

function Tile({ number, image }: TileProps) {
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