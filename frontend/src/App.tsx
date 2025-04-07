import './App.css';

import BoardView from './pages/boardview/boardview';
import TournamentView from './pages/tournamentview/tournamentview';
import { BrowserRouter, Routes, Route } from "react-router-dom";

/**
 * App Component
 *
 * This is the root component of the React application.
 * It sets up client-side routing using React Router.
 *
 * Routes:
 * - `/`          → TournamentView (overview or menu for all boards)
 * - `/board/1`   → BoardView for Board 1
 * - `/board/2`   → BoardView for Board 2
 *
 * Additional board routes can be added here as needed.
 */

function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Main tournament view */}
        <Route path="/" element={<TournamentView />} />

        {/* Individual board views with IDs */}
        <Route path="/board/1" element={<BoardView id={1} />} />
        <Route path="/board/2" element={<BoardView id={2} />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
