import './App.css';
import { useState } from 'react';
import TableView from './pages/boardview/boardview';

/**
 * App Component
 *
 * This is the main entry point of the React application.
 * It maintains the state of the chess moves and passes it
 * down to the `TableView` component.
 */

function App() {
  // State to hold the list of chess moves (in algebraic notation)
  const [moves, setMoves] = useState<string[]>([]);

  return (
    <div id="app">
      <TableView moves={moves} setMoves={setMoves} />
    </div>
  );
}

export default App;