import { useEffect, useState } from 'react'
import { getUsers, getUserStatus, deleteUser } from '../api'

function Badge({ on, label }) {
  return (
    <span className={`px-2 py-0.5 rounded text-xs font-medium ${
      on ? 'bg-green/15 text-green border border-green/30'
         : 'bg-border/50 text-muted border border-border'
    }`}>
      {label}
    </span>
  )
}

function UserRow({ username, onDelete }) {
  const [status,   setStatus]  = useState(null)
  const [deleting, setDeleting] = useState(false)
  const [confirm,  setConfirm]  = useState(false)

  useEffect(() => {
    getUserStatus(username).then(setStatus).catch(() => {})
  }, [username])

  async function handleDelete() {
    if (!confirm) { setConfirm(true); return }
    setDeleting(true)
    try { await deleteUser(username); onDelete(username) }
    catch (e) { alert(e.message) }
    finally { setDeleting(false) }
  }

  return (
    <div className="flex items-center justify-between p-3 border-b border-border last:border-0">
      <div>
        <p className="text-sm font-bold text-text">{username}</p>
        {status && (
          <div className="flex gap-2 mt-1">
            <Badge on={status.enrolled_text_independent} label="Text-Independent" />
            <Badge on={status.enrolled_text_dependent}   label="Text-Dependent" />
          </div>
        )}
      </div>
      <button
        onClick={handleDelete}
        disabled={deleting}
        className={`text-xs px-3 py-1.5 rounded border transition ${
          confirm
            ? 'border-red text-red bg-red/10 hover:bg-red/20'
            : 'border-border text-muted hover:border-red hover:text-red'
        } disabled:opacity-40`}
      >
        {deleting ? 'Removing...' : confirm ? 'Confirm delete?' : 'Remove'}
      </button>
    </div>
  )
}

export default function Users() {
  const [users,   setUsers]   = useState([])
  const [loading, setLoading] = useState(true)
  const [error,   setError]   = useState(null)

  function load() {
    setLoading(true)
    getUsers()
      .then(d => { setUsers(d.users ?? []); setLoading(false) })
      .catch(e => { setError(e.message); setLoading(false) })
  }

  useEffect(() => { load() }, [])

  return (
    <div className="max-w-2xl mx-auto px-4 py-8 fade-in">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-xl font-bold mb-1">Enrolled Users</h1>
          <p className="text-muted text-xs">{users.length} user{users.length !== 1 ? 's' : ''} registered</p>
        </div>
        <button onClick={load} className="text-xs text-muted hover:text-text px-3 py-1.5 border border-border rounded transition">
          ↻ Refresh
        </button>
      </div>

      {loading && <p className="text-muted text-sm">Loading...</p>}
      {error   && <p className="text-red text-sm">{error}</p>}

      {!loading && users.length === 0 && (
        <div className="bg-card border border-border rounded-lg p-8 text-center">
          <p className="text-muted text-sm">No users enrolled yet.</p>
          <p className="text-xs text-muted mt-1">
            Go to <a href="/enroll" className="text-accent underline">Enroll</a> to register a speaker.
          </p>
        </div>
      )}

      {users.length > 0 && (
        <div className="bg-card border border-border rounded-lg overflow-hidden">
          {users.map(u => (
            <UserRow
              key={u}
              username={u}
              onDelete={name => setUsers(prev => prev.filter(x => x !== name))}
            />
          ))}
        </div>
      )}
    </div>
  )
}
