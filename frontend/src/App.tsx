import './App.css'
import { Chess } from 'chess.ts'

function App() {

  const chess = new Chess()

   // make some moves
   chess.move('e4')
   chess.move('e5')
   chess.move('f4')
 
   const board = chess.ascii()

  return (
    <>
      <h1>CHESS</h1>
      <div className='chessboard'>{board}</div>
    </>
  )
}

export default App
