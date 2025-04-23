import './App.css';

import BoardView from './pages/boardview/boardview';
import TournamentView from './pages/tournamentview/tournamentview';
import HowItWorks from './pages/howItWorks/howItWorks';
import AboutUs from './pages/aboutUs/aboutUs';
import Header from './layouts/header';
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
    <Header />
      <Routes>
        {/* Main tournament view */}
        <Route path="/" element={<TournamentView />} />
        {/* How it works view */}
        <Route path="/how-it-works" element={<HowItWorks />} />
        {/* About us view */}
        <Route path="/about" element={<AboutUs />} />

        {/* Individual board views with IDs */}
        <Route path="/board/1" element={<BoardView id={1} />} />
        <Route path="/board/2" element={<BoardView id={2} />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
