import ScoreBar from './ScoreBar'

const DECISION_STYLE = {
  ACCEPTED:        { bg: 'border-green',  text: 'text-green',  icon: '✓', label: 'ACCESS GRANTED' },
  REJECTED_VOICE:  { bg: 'border-red',    text: 'text-red',    icon: '✗', label: 'WRONG SPEAKER'  },
  REJECTED_SPOOF:  { bg: 'border-orange', text: 'text-orange', icon: '⚠', label: 'SPOOF DETECTED' },
  REJECTED_TEXT:   { bg: 'border-orange', text: 'text-orange', icon: '✗', label: 'WRONG PASSPHRASE'},
}

export default function ResultCard({ result, mode = 'auth' }) {
  if (!result) return null

  // Spoof-only result
  if (mode === 'spoof') {
    const ok = result.is_real
    return (
      <div className={`fade-in border rounded-lg p-4 mt-4 bg-card ${ok ? 'border-green' : 'border-red'}`}>
        <div className={`text-lg font-bold mb-3 ${ok ? 'text-green' : 'text-red'}`}>
          {ok ? '✓  REAL VOICE' : '✗  SPOOF DETECTED'}
        </div>
        <ScoreBar label="Spoof Score" value={result.score} threshold={0.20} />
        <div className="text-xs text-muted mt-2">
          Attack type: <span className="text-text">{result.attack_type}</span>
          &nbsp;·&nbsp;{result.latency_ms?.toFixed(0)} ms
        </div>
      </div>
    )
  }

  // Auth result
  const style = DECISION_STYLE[result.decision] ?? DECISION_STYLE.REJECTED_VOICE
  return (
    <div className={`fade-in border rounded-lg p-4 mt-4 bg-card ${style.bg}`}>
      <div className={`text-lg font-bold mb-4 ${style.text}`}>
        {style.icon}&nbsp; {style.label}
      </div>

      <ScoreBar label="Anti-Spoof Score" value={result.spoof_score} threshold={0.20} />

      {result.voice_score != null && (
        <ScoreBar label="Voice Score"     value={result.voice_score}  threshold={0.75} />
      )}
      {result.text_score != null && (
        <ScoreBar label="Text Match Score" value={result.text_score}  threshold={0.70} />
      )}
      {result.combined_score != null && (
        <ScoreBar label="Combined Score"   value={result.combined_score} />
      )}

      {result.transcript && (
        <div className="mt-3 p-2 rounded bg-bg border border-border text-xs">
          <span className="text-muted">Transcript: </span>
          <span className="text-accent italic">"{result.transcript}"</span>
        </div>
      )}

      <div className="text-xs text-muted mt-3">
        {result.latency_ms?.toFixed(0)} ms
      </div>
    </div>
  )
}
