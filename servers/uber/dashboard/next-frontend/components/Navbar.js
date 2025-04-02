import Link from 'next/link';
import '../styles/Navbar.css';

function Navbar() {
  return (
    <nav className="navbar">
      <Link href="/" passHref style={{ textDecoration: 'none' }}>
        <span className="navbar-brand">OpenMMLA Dashboard</span>
      </Link>
    </nav>
  );
}

export default Navbar;
