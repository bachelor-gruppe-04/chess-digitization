import React from "react";
import "./evalBar.css";

interface EvalBarProps {
  evaluation: number | null; // centipawn value or null
  id: number;                // board ID
}

const EvalBar: React.FC<EvalBarProps> = ({ evaluation, id }) => {
  const whitePercentage = (() => {
    if (evaluation === null) return 50;
    const clamped = Math.max(-10, Math.min(10, evaluation));  // Clamp evaluation to be between -10 and 10
    return 50 + clamped * 5;  // +ve for white, -ve for black
  })();

  console.log('Evaluation:', evaluation, 'ID:', id);  // Debug info

  return (
    <div className="eval-bar">
      <div className="white-bar" style={{ 
        [window.innerWidth < 768 ? 'width' : 'height']: `${whitePercentage}%` 
      }} />
      <div className="black-bar" style={{ 
        [window.innerWidth < 768 ? 'width' : 'height']: `${100 - whitePercentage}%` 
      }} />
    </div>
  );
};

export default EvalBar;
