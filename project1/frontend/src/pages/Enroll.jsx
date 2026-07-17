import { useState } from 'react'
import AudioUpload from '../components/AudioUpload'
import { enrollTextIndependent, enrollTextDependent } from '../api'

const TABS = ['Text-Independent', 'Text-Dependent']

export default function Enroll() {
  const [tab,        setTab]       = useState(0)
  const [username,   setUsername]  = useState('')
  const [passphrase, setPhrase]    = useState('my voice is my password')
  const [files,      setFiles]     = useState([])
  const [loading,    setLoading]   = useState(false)
  const [result,     setResult]    = useState(null)
  const [error,      setError]     = useState(null)

  async function handleSubmit(e) {
    e.preventDefault()
    if (!username.trim()) { setError('Enter a username.'); return }
    if (files.length < 3) { setError('Upload at least 3 WAV files.'); return }
    setError(null); setResult(null); setLoading(true)
    try {
      const res = tab === 0
        ? await enrollTextIndependent(username.trim(), files)
        : await enrollTextDependent(username.trim(), files, passphrase)
      setResult(res)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-2xl mx-auto px-4 py-8 fade-in">
      <h1 className="text-xl font-bold mb-1">Enroll User</h1>
      <p className="text-muted text-xs mb-6">Register a new speaker's voice in the system.</p>

      {/* Tabs */}
      <div className="flex border-b border-border mb-6">
        {TABS.map((t, i) => (
          <button key={i} onClick={() => { setTab(i); setResult(null); setError(null) }}
            className={`px-4 py-2 text-xs font-medium border-b-2 transition-colors ` +
              (tab === i ? 'border-accent text-accent' : 'border-transparent text-muted hover:text-text')}>
            {t}
          </button>
        ))}
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Username */}
        <div>
          <label className="block text-xs text-muted mb-1">Username</label>
          <input
            value={username} onChange={e => setUsername(e.target.value)}
            placeholder="e.g. abhiram"
            className="w-full bg-bg border border-border rounded-lg px-3 py-2 text-sm text-text focus:border-accent outline-none"
          />
        </div>

        {/* Passphrase (text-dependent only) */}
        {tab === 1 && (
          <div>
            <label className="block text-xs text-muted mb-1">Passphrase</label>
            <input
              value={passphrase} onChange={e => setPhrase(e.target.value)}
              className="w-full bg-bg border border-border rounded-lg px-3 py-2 text-sm text-text focus:border-accent outline-none"
            />
            <p className="text-xs text-muted mt-1">User must say this exact phrase during authentication.</p>
          </div>
        )}

        {/* Audio upload */}
        <AudioUpload
          label={`Upload ≥3 WAV recordings ${tab === 1 ? '(saying the passphrase)' : '(any speech)'}`}
          multiple
          onChange={f => setFiles(Array.isArray(f) ? f : [f])}
        />
        <p className="text-xs text-muted -mt-2">
          {files.length} file{files.length !== 1 ? 's' : ''} selected
          {files.length > 0 && files.length < 3 ? ' — need at least 3' : ''}
        </p>

        {error && <p className="text-red text-xs bg-red/10 border border-red/30 rounded p-2">{error}</p>}

        <button
          type="submit" disabled={loading}
          className="w-full py-2.5 bg-accent text-bg text-sm font-bold rounded-lg hover:opacity-90 disabled:opacity-50 transition"
        >
          {loading ? 'Enrolling...' : `Enroll (${TABS[tab]})`}
        </button>
      </form>

      {/* Success result */}
      {result && (
        <div className="fade-in mt-5 p-4 rounded-lg border border-green bg-green/5">
          <p className="text-green font-bold text-sm mb-1">✓ Enrolled Successfully</p>
          <p className="text-xs text-muted">
            User: <span className="text-text">{result.username}</span>
            &nbsp;·&nbsp;Mode: <span className="text-text">{result.mode}</span>
            &nbsp;·&nbsp;Samples: <span className="text-text">{result.samples_enrolled}</span>
          </p>
        </div>
      )}
    </div>
  )
}
