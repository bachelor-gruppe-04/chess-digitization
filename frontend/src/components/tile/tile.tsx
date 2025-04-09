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
 * 
 * If `highlight` is true, the tile is visually highlighted (e.g., for last move or selection).
 */

/**
 * Props for the Tile component
 * - `number`: Determines tile color based on parity (even = black, odd = white)
 * - `image`: Optional image URL for rendering a chess piece
* - `highlight`: (optional) Whether to visually highlight the tile
 */
interface TileProps {
  number: number; 
  image?: string;
  highlight?: boolean;
}

function Tile({ number, image, highlight  }: TileProps) {
  const baseClass = number % 2 === 0 ? "black-tile" : "white-tile"; // Determine tile color class based on parity
  const highlightClass = highlight ? "highlight" : ""; // Conditionally apply highlight class

  return (
    <div className={`tile ${baseClass} ${highlightClass}`}>
      {image && <div style={{ backgroundImage: `url(${image})` }} className="chess-piece"></div>}
    </div>
  );
}

export default Tile;