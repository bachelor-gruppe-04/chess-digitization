import './App.css'
import Chessboard from './features/display/chessboard'

/**
 * This is the main entry component of the application.
 */
function App() {

  return (
    <>
      <div className='chessboard'>
        <Chessboard />
      </div>
    </>
  )
}

export default App
