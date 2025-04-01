import './camera.css';

interface CameraProps {
  id: number;
}

function Camera({ id }: CameraProps) {
  return (
    <div className="webcam-container">
      <h2>Live Webcam View</h2>
      <img
        src={`http://localhost:8000/video/${id}`}
        alt="Webcam Feed"
        className="webcam-feed"
      />
    </div>
  );
}

export default Camera;
