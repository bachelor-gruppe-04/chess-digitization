import React from "react";
import "./evalBar.css";

interface EvalBarProps {
  evaluation: string | number | null;  // Accept both string (for mate) and number
  mate: number | null;
  id: number;
}
const EvalBar: React.FC<EvalBarProps> = ({ evaluation, mate, id }) => {
  // Determine if there's a mate, if mate is provided and is a valid number
  const isMate = mate !== null && mate !== 0; // Mate exists and is not 0

  // Determine the display value: if mate exists, show M1, M2, -M1, etc., else show evaluation or N/A
  const displayValue = isMate 
  ? `${mate! > 0 ? '' : '-'}M${Math.abs(mate!)}`
  : (evaluation !== null && typeof evaluation === 'number' ? evaluation.toString() : "");

  console.log(displayValue)

  const whitePercentage = (() => {
    if (isMate && Math.abs(mate!) === 1) {
      return mate! > 0 ? 100 : 0; // 100% white if White mates in 1, 0% if Black mates in 1
    }
  
    if (evaluation === null || typeof evaluation !== 'number') return 50;
  
    const clamped = Math.max(-10, Math.min(10, evaluation));
    return 50 + clamped * 5;
  })();
  

  const isPositiveEvaluation = typeof evaluation === 'number' && evaluation > 0.0;

  return (
    <div className="eval-bar">
      <div
        className="white-bar"
        style={{ height: `${whitePercentage}%` }}
      />
      <div
        className="black-bar"
        style={{ height: `${100 - whitePercentage}%` }}
      />
      <div
  className={`eval-text ${isPositiveEvaluation ? 'black-text' : 'white-text'}`}
>
  {displayValue}
</div>

    </div>
  );
};


export default EvalBar;
