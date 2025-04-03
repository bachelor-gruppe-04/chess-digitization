import './App.css';
import TournamentView from './pages/tournamentview/tournamentview';

/**
 * App Component
 *
 * This is the main entry point of the React application.
 * It maintains the state of the chess moves and passes it
 * down to the `TableView` component.
 */

function App() {
  return (
    <div id="app">
      <TournamentView/>
    </div>
  );
}

export default App;