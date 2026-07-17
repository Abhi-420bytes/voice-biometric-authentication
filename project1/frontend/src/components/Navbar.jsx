import { NavLink } from 'react-router-dom'

const links = [
  { to: '/',            label: 'Dashboard' },
  { to: '/enroll',      label: 'Enroll' },
  { to: '/authenticate',label: 'Authenticate' },
  { to: '/spoof',       label: 'Spoof Detect' },
  { to: '/users',       label: 'Users' },
]

export default function Navbar() {
  return (
    <nav className="border-b border-border bg-card sticky top-0 z-50">
      <div className="max-w-5xl mx-auto px-4 flex items-center gap-6 h-14">
        <span className="text-accent font-bold text-sm tracking-widest mr-4">
          🎙 VOICE AUTH
        </span>
        {links.map(({ to, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              `text-xs font-medium transition-colors pb-0.5 border-b-2 ` +
              (isActive
                ? 'text-accent border-accent'
                : 'text-muted border-transparent hover:text-text')
            }
          >
            {label}
          </NavLink>
        ))}
      </div>
    </nav>
  )
}
