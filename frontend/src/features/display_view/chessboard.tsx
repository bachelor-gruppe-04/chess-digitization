import { Chess } from 'chess.ts'
import { useState } from 'react';
import './chessboard.css';


/**
 * This feature initializes a chess game using the `chess.ts` library.
 * It makes a few predefined moves and displays the board in ASCII format.
 * The board is styled using the `chessboard.css` file.
 */

function Chessboard() {
  const [chess] = useState(new Chess()); // Initialize chess game
  const [board, setBoard] = useState(chess.ascii()); // Store board state
  const [move, setMove] = useState(""); // Store user input

   // Function to handle user move
   const handleMove = () => {
    if (chess.move(move)) { // Check if move is valid
      setBoard(chess.ascii()); // Update board
    } else {
      alert("Invalid move! Try again.");
    }
    setMove(""); // Reset input field
  };

   return (
    <>
      <h1>CHESS</h1>
      <pre className="chessboard">{board}</pre>

      <input
        type="text"
        value={move}
        onChange={(e) => setMove(e.target.value)}
        placeholder="Enter move (e.g., e4)"
      />
      <button onClick={handleMove}>Make Move</button>
    </>
  );
}

export default Chessboard