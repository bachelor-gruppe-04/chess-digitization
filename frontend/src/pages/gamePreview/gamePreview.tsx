/**
 * GamePreview Component
 * 
 * This component serves as a navigation hub for multiple chess boards in a tournament.
 * It renders links to individual board views using React Router's `NavLink` component.
 * 
 * Each link directs the user to a unique board route (e.g., `/board/1`, `/board/2`).
 * This is a scalable layout for managing and switching between boards in a tournament.
 */

function GamePreview() {
  return (
    <div className='game-view'>
      <div className="heading">
        Game<span> Preview</span>
      </div>
      
    </div>
  );
}

export default GamePreview;
