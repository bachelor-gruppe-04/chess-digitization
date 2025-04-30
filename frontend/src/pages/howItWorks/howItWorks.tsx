import './howItWorks.css';


/**
 * HowItWorks Component
 *
 * This page explains how the Chess Camera works for users.
 */


function HowItWorks() {
  return (
    <div className="how-it-works">
      <div className="heading">
        How it<span> Works</span>
      </div>

      <div className="how-it-works-content">
        <p>
          Welcome to the Chess Camera! This system allows chess games played on traditional boards to be digitized and streamed live through an automated setup.
        </p>

        <h2>Digitization Process</h2>
        <p>
          A standard USB-connected webcam is positioned above the chessboard to continuously capture images during gameplay. 
          These images are processed locally using image recognition models that detect the board layout, piece types, and move sequences in real time.
          The system validates moves according to chess rules, ensuring that the digital version remains synchronized with the physical game.
        </p>

        <h2>Live Boards</h2>
        <p>
          Each ongoing game is represented by a live digital board on the dashboard. 
          As players make moves in real life, the corresponding moves are instantly reflected online.
          Observers can watch the progress of games remotely without interfering with players at the venue.
        </p>

        <h2>Tournament View</h2>
        <p>
          The Tournament Table provides a comprehensive overview of all active games, including player names, ratings, and board numbers.
          Clicking on a match takes you to the detailed live board, where you can follow the game move by move.
        </p>

        <h2>Navigation</h2>
        <p>
          - Use the <strong>Game Preview</strong> page to see visual preview of each board.<br />
          - Use the <strong>Tournament View</strong> page for a structured table of matches.<br />
          - Click on a board to switch to live detailed viewing instantly.
        </p>

        <h2>Technology Behind Chess Camera</h2>
        <p>
          The system is built with a lightweight, scalable front-end powered by modern web technologies, including React and TypeScript.
          Real-time move updates are handled via WebSocket connections for minimal latency and smooth game tracking.
          Image recognition is performed locally using custom-trained LeYOLO models and Python-based processing, ensuring fast response times without requiring cloud services.
        </p>
      </div>
    </div>
  );
}

export default HowItWorks;
