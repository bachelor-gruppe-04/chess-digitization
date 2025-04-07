import './App.css';
import { BrowserRouter, Routes, Route } from "react-router-dom";
import BoardView from './pages/boardview/boardview';
import TournamentView from './pages/tournamentview/tournamentview';

/**
 * App Component
 *
 * This is the main entry point of the React application.
 */

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<TournamentView />} />
        <Route path="/board/1" element={<BoardView id={1}/>} />
        <Route path="/board/2" element={<BoardView id={2}/>} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
