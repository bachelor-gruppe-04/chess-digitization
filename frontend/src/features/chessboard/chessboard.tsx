import { Chess } from 'chess.ts'
import './chessboard.css'

/**
 * This feature initializes a chess game using the `chess.ts` library.
 * It makes a few predefined moves and displays the board in ASCII format.
 * The board is styled using the `chessboard.css` file.
 */

function Chessboard() {
  const chess = new Chess()

  // Make some initial moves to mock a gameplay
   chess.move('e4')
   chess.move('e5')
   chess.move('f4')
 
   const board = chess.ascii() // Generate ASCII representation of the board

  return (
    <>
      <h1>CHESS</h1>
      <div className='chessboard'>{board}</div>
    </>
  )
}

export default Chessboard
