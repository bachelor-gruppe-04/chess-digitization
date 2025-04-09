import "./Header.css";

import { Link } from "react-router-dom";

/**
 * Header Component
 *
 * This component renders the top navigation bar of the application.
 * It includes the app logo and navigation links to key pages.
 */
const Header = () => {
  return (
    <header className="header">
      <div className="header-logo">
        Chess<span>Cam</span>
      </div>
      <nav className="header-links">
        <Link to="/">Tournament View</Link>
        <Link to="/how-it-works">How it works</Link>
        <Link to="/about">About</Link>
      </nav>
    </header>
  );
};

export default Header;
