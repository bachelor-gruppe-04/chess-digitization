import './camera.css';

import { useState  } from 'react';

/**
 * Camera Component
 *
 * Displays a live webcam feed for a given board using an image stream.
 * The video stream is served from a backend endpoint and updated automatically by the browser.
 *
 * The stream URL is constructed using the board ID: `http://localhost:8000/video/{id}`
 */

/**
 * Props for the Camera component
 * - `id`: Unique identifier used to fetch the correct webcam stream for a specific board
 */
interface CameraProps {
  id: number;
}

function Camera({ id }: CameraProps) {
  const [isFullscreen, setIsFullscreen] = useState(false); // Tracks whether the fullscreen view is active

  /**
   * Toggles fullscreen mode for the webcam feed.
   * When enabled, the webcam image is shown in an overlay with a close button.
   */
  const toggleFullscreen = () => {
    setIsFullscreen((prev) => !prev);
  };

  return (
    <>
      {/* Main webcam feed container */}
      <div className="webcam-container">
        <img
          src={`http://localhost:8000/video/${id}`}
          alt="Webcam Feed"
          className="webcam-feed"
        />
        <button className="fullscreen-button" onClick={toggleFullscreen}>
          ⛶
        </button>
      </div>

      {/* Fullscreen overlay (only rendered when fullscreen is active) */}
      {isFullscreen && (
        <div className="fullscreen-overlay" onClick={toggleFullscreen}>
          <div className="fullscreen-content" onClick={(e) => e.stopPropagation()}>
            <button className="close-button" onClick={toggleFullscreen}>
              ×
            </button>
            <img
              src={`http://localhost:8000/video/${id}`}
              alt="Webcam Fullscreen"
              className="webcam-fullscreen"
            />
          </div>
        </div>
      )}
    </>
  );
}

export default Camera;
