import { useState } from 'react'
import AudioUpload from '../components/AudioUpload'
import ResultCard  from '../components/ResultCard'
import { detectSpoof } from '../api'

export default function SpoofDetect() {
  const [file,    setFile]    = useState(null)
  const [loading, setLoading] = useState(false)
  const [result,  setResult]  = useState(null)
  const [error,   setError]   = useState(null)

  async function handleSubmit(e) {
    e.preventDefault()
    if (!file) { setError('Provide an audio file.'); return }
    setError(null); setResult(null); setLoading(true)
    try {
      const res = await detectSpoof(file)
      setResult(res)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-2xl mx-auto px-4 py-8 fade-in">
      <h1 className="text-xl font-bold mb-1">Spoof / Deepfake Detection</h1>
      <p className="text-muted text-xs mb-6">
        Analyse any WAV file to determine if it is real human speech or synthesised/replayed audio.
      </p>

      {/* Feature legend */}
      <div className="grid grid-cols-3 gap-2 mb-6 text-xs">
        {[
          { label: 'Pitch Jitter', desc: 'Natural F0 variation' },
          { label: 'Modulation Energy', desc: 'Syllabic rhythm (4–16 Hz)' },
          { label: 'Spectral Flatness', desc: 'Sub-band noise profile' },
        ].map(({ label, desc }) => (
          <div key={label} className="bg-card border border-border rounded p-2">
            <p className="text-accent font-bold">{label}</p>
            <p className="text-muted">{desc}</p>
          </div>
        ))}
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        <AudioUpload
          label="Upload or record audio to analyse"
          multiple={false}
          onChange={setFile}
        />

        {error && <p className="text-red text-xs bg-red/10 border border-red/30 rounded p-2">{error}</p>}

        <button
          type="submit" disabled={loading}
          className="w-full py-2.5 bg-accent text-bg text-sm font-bold rounded-lg hover:opacity-90 disabled:opacity-50 transition"
        >
          {loading ? 'Analysing...' : 'Detect Spoof'}
        </button>
      </form>

      <ResultCard result={result} mode="spoof" />
    </div>
  )
}
