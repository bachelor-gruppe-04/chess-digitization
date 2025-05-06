import './aboutUs.css';

import { useEffect } from 'react';

/**
 * AboutUs Component
 *
 * This page explains why the solution Chess Camera was developed.
 */

function AboutUs() {
  /**
   * Update the browser tab title
   */
  useEffect(() => {
    document.title = `About Us - ChessCamera`;
  });
  
  return (
    <div className="about-us">
      <div className="heading">
        About<span> Us</span>
      </div>

      <div className="about-us-content">
        <p>
          Welcome to Chess Camera — an solution designed to digitize and display chess gameplay in real time.
        </p>

        <h2>Project Purpose</h2>
        <p>
          This project was initiated for Aalesunds Schaklag with the goal of creating an automated system that captures live chess matches
          played on regular boards and transforms them into digital PGN files without manual input. 
          The digitized games are made accessible via an API or can be streamed as real-time move events through a message queue, 
          allowing spectators and systems to follow the action remotely.
        </p>

        <h2>System Overview</h2>
        <p>
          Chess Camera leverages advanced image recognition techniques to identify pieces and board states accurately.
          Using computer vision models, the system detects each move as it happens, while validating the legality of moves
          to ensure full compliance with chess rules. This enables seamless synchronization between the physical and digital boards.
        </p>

        <h2>Technical Design</h2>
        <p>
          Designed for portability and cost-efficiency, Chess Camera runs entirely on local hardware — typically a standard USB webcam
          paired with a machine running Windows or Ubuntu. By avoiding reliance on cloud computing, the system ensures faster processing, 
          reduced costs, and improved privacy for tournaments and chess clubs.
          Unlike expensive digital chessboards that require complex setup and infrastructure, Chess Camera offers a streamlined, 
          easy-to-deploy alternative for digitizing chess matches.
        </p>

        <h2>Development Process</h2>
        <p>
          The project was developed following agile methodologies, emphasizing iterative improvement, rapid prototyping, and adaptability
          to real-world requirements. Technologies such as LeYOLO for computer vision and Python for backend processing
          were used to build a robust and scalable system capable of operating reliably under tournament conditions.
        </p>
      </div>
    </div>
  );
}

export default AboutUs;
