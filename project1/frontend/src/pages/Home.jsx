import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { getHealth, getUsers } from '../api'

function StatCard({ label, value, color = 'text-accent' }) {
  return (
    <div className="bg-card border border-border rounded-lg p-5">
      <p className="text-xs text-muted mb-1">{label}</p>
      <p className={`text-3xl font-bold ${color}`}>{value}</p>
    </div>
  )
}

function ActionCard({ to, icon, title, desc, color }) {
  return (
    <Link
      to={to}
      className="block bg-card border border-border rounded-lg p-5 hover:border-accent transition-colors group"
    >
      <div className={`text-2xl mb-2`}>{icon}</div>
      <p className={`font-bold text-sm mb-1 group-hover:text-accent transition-colors`}>{title}</p>
      <p className="text-xs text-muted">{desc}</p>
    </Link>
  )
}

export default function Home() {
  const [apiOk,  setApiOk]  = useState(null)
  const [users,  setUsers]  = useState([])

  useEffect(() => {
    getHealth()
      .then(() => setApiOk(true))
      .catch(() => setApiOk(false))
    getUsers()
      .then(d => setUsers(d.users ?? []))
      .catch(() => {})
  }, [])

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 fade-in">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-text mb-1">Voice Biometric Authentication</h1>
        <p className="text-muted text-sm">
          B.Tech Sem 6 — Speech Processing Project
        </p>
      </div>

      {/* API status banner */}
      <div className={`mb-6 px-4 py-2.5 rounded-lg border text-xs font-medium flex items-center gap-2 ${
        apiOk === null  ? 'border-border text-muted' :
        apiOk           ? 'border-green text-green bg-green/5' :
                          'border-red text-red bg-red/5'
      }`}>
        <span className={`w-2 h-2 rounded-full ${
          apiOk === null ? 'bg-muted' : apiOk ? 'bg-green' : 'bg-red'
        }`} />
        {apiOk === null ? 'Checking API...' :
         apiOk          ? 'API is running  ·  http://127.0.0.1:8000' :
                          'API is offline — run: python main.py'}
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-4 mb-8">
        <StatCard label="Enrolled Users"     value={users.length} />
        <StatCard label="Auth Modes"         value={2}   color="text-purple" />
        <StatCard label="Deepfake Detection" value="ON"  color="text-green"  />
      </div>

      {/* Quick actions */}
      <h2 className="text-sm font-bold text-muted uppercase tracking-widest mb-4">Quick Actions</h2>
      <div className="grid grid-cols-2 gap-4 mb-8">
        <ActionCard to="/enroll"       icon="🎤" title="Enroll New User"
          desc="Upload or record voice samples to register a speaker." />
        <ActionCard to="/authenticate" icon="🔐" title="Authenticate"
          desc="Verify a speaker's identity with text-independent or passphrase mode." />
        <ActionCard to="/spoof"        icon="🛡" title="Spoof Detection"
          desc="Upload any audio to check if it is real voice or synthesised/replayed." />
        <ActionCard to="/users"        icon="👥" title="Manage Users"
          desc="View enrolled users and remove enrollments." />
      </div>

      {/* Pipeline diagram */}
      <h2 className="text-sm font-bold text-muted uppercase tracking-widest mb-4">How It Works</h2>
      <div className="bg-card border border-border rounded-lg p-5">
        <div className="flex items-center gap-2 flex-wrap text-xs">
          {['Audio Input', '→', 'Preprocessing', '→', 'Anti-Spoofing', '→', 'Speaker Verify', '→', 'Decision'].map((s, i) => (
            <span key={i} className={s === '→' ? 'text-muted' : 'px-2.5 py-1 rounded bg-bg border border-border text-text'}>
              {s}
            </span>
          ))}
        </div>
        <div className="mt-4 grid grid-cols-3 gap-3 text-xs">
          <div className="p-3 rounded bg-bg border border-border">
            <p className="text-accent font-bold mb-1">Speaker Encoder</p>
            <p className="text-muted">Resemblyzer GE2E · 256-dim d-vector · Cosine similarity</p>
          </div>
          <div className="p-3 rounded bg-bg border border-border">
            <p className="text-orange font-bold mb-1">Anti-Spoofing</p>
            <p className="text-muted">Pitch jitter · Modulation energy · Spectral flatness · GMM</p>
          </div>
          <div className="p-3 rounded bg-bg border border-border">
            <p className="text-purple font-bold mb-1">Text Verification</p>
            <p className="text-muted">Whisper ASR · Fuzzy passphrase match · 0.6×char + 0.4×word</p>
          </div>
        </div>
      </div>
    </div>
  )
}
