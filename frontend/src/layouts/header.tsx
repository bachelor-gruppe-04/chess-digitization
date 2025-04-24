import "./Header.css";

import { Link } from "react-router-dom";
import { useState } from "react";

/**
 * Header Component
 *
 * This component renders the top navigation bar of the application.
 * - Displays the logo (ChessCamera)
 * - Provides navigation links to key pages
 * - Includes a hamburger menu button for toggling visibility on smaller screens
 */
const Header = () => {
  const [menuOpen, setMenuOpen] = useState(false); // Controls visibility of the navigation menu

  return (
    <header className="header">
      <div className="header-logo">
        Chess<span>Camera</span>
      </div>

      <nav className={`header-links ${menuOpen ? "open" : ""}`}>
        <Link to="/" onClick={() => setMenuOpen(false)}>Tournament View</Link>
        <Link to="/how-it-works" onClick={() => setMenuOpen(false)}>How it works</Link>
        <Link to="/about" onClick={() => setMenuOpen(false)}>About</Link>
      </nav>

      <button
        className="hamburger"
        onClick={() => setMenuOpen((prev) => !prev)}
        aria-label="Toggle menu"
      >
        ☰
      </button>
    </header>
  );
};;

export default Header;
