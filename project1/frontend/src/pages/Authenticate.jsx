import { useState } from 'react'
import AudioUpload from '../components/AudioUpload'
import ResultCard  from '../components/ResultCard'
import { authTextIndependent, authTextDependent } from '../api'

const TABS = ['Text-Independent', 'Text-Dependent']

export default function Authenticate() {
  const [tab,       setTab]      = useState(0)
  const [username,  setUsername] = useState('')
  const [passphrase,setPhrase]   = useState('my voice is my password')
  const [file,      setFile]     = useState(null)
  const [loading,   setLoading]  = useState(false)
  const [result,    setResult]   = useState(null)
  const [error,     setError]    = useState(null)

  async function handleSubmit(e) {
    e.preventDefault()
    if (!username.trim()) { setError('Enter a username.'); return }
    if (!file)             { setError('Provide an audio file.'); return }
    setError(null); setResult(null); setLoading(true)
    try {
      const res = tab === 0
        ? await authTextIndependent(username.trim(), file)
        : await authTextDependent(username.trim(), file, passphrase)
      setResult(res)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-2xl mx-auto px-4 py-8 fade-in">
      <h1 className="text-xl font-bold mb-1">Authenticate</h1>
      <p className="text-muted text-xs mb-6">Verify a speaker's identity against enrolled voice.</p>

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

        {/* Passphrase */}
        {tab === 1 && (
          <div>
            <label className="block text-xs text-muted mb-1">Expected Passphrase</label>
            <input
              value={passphrase} onChange={e => setPhrase(e.target.value)}
              className="w-full bg-bg border border-border rounded-lg px-3 py-2 text-sm text-text focus:border-accent outline-none"
            />
          </div>
        )}

        {/* Audio */}
        <AudioUpload
          label={tab === 1
            ? 'Upload or record audio — user must say the passphrase'
            : 'Upload or record any speech to verify'}
          multiple={false}
          onChange={setFile}
        />

        {error && <p className="text-red text-xs bg-red/10 border border-red/30 rounded p-2">{error}</p>}

        <button
          type="submit" disabled={loading}
          className="w-full py-2.5 bg-accent text-bg text-sm font-bold rounded-lg hover:opacity-90 disabled:opacity-50 transition"
        >
          {loading ? 'Authenticating...' : `Verify (${TABS[tab]})`}
        </button>
      </form>

      <ResultCard result={result} mode="auth" />
    </div>
  )
}
