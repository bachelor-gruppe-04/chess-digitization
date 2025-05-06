import './camera.css';

import { createPortal } from 'react-dom';
import { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';

/**
 * Camera Component
 *
 * Displays a live webcam feed for a given chess board using an MJPEG stream (as an <img>).
 * Users can toggle fullscreen mode for a larger view of the board.
 *
 * Uses React Portal to render fullscreen overlay at the top-level DOM node (`#fullscreen-root`)
 * to avoid CSS stacking or layout issues.
 */

/**
 * Props for the Camera component
 * - `id`: Unique identifier used to fetch the correct webcam stream for a specific board
 */
interface CameraProps {
  id: string | undefined;
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

   /**
   * When fullscreen is active, prevent body scroll to avoid background scrolling.
   * Restore scroll on unmount or when fullscreen is disabled.
   */
  useEffect(() => {
    document.body.style.overflow = isFullscreen ? 'hidden' : 'auto';
    return () => {
      document.body.style.overflow = 'auto';
    };
  }, [isFullscreen]);

  /**
   * Fullscreen overlay element (rendered via portal)
   * - Click outside image closes fullscreen
   * - Click on the image itself does NOT close (uses event.stopPropagation)
   */
  const fullscreenContent = (
    <div className="fullscreen-overlay" >
      <div className="fullscreen-content" onClick={toggleFullscreen}>
        <div className="webcam-fullscreen-wrapper" onClick={(e) => e.stopPropagation()}>
          <button className="close-button" onClick={toggleFullscreen}>×</button>
          <img
            src={`http://localhost:8000/video/${id}`}
            alt="Webcam Fullscreen"
            className="webcam-fullscreen"
          />
        </div>
      </div>
    </div>
  );

  return (
    <>
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
      {isFullscreen && createPortal(fullscreenContent, document.getElementById('fullscreen-root')!)}
    </>
  );
}

export default Camera;
