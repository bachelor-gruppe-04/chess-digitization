import './camera.css';

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
  return (
    <div className="webcam-container">
      <img
        src={`http://localhost:8000/video/${id}`}
        alt="Webcam Feed"
        className="webcam-feed"
      />
    </div>
  );
}

export default Camera;
