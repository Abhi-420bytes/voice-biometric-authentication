export default function ScoreBar({ label, value, threshold, color }) {
  const pct     = Math.round((value ?? 0) * 100)
  const barColor = color ?? (value >= threshold ? '#3fb950' : '#f85149')

  return (
    <div className="mb-3">
      <div className="flex justify-between text-xs mb-1">
        <span className="text-muted">{label}</span>
        <span style={{ color: barColor }} className="font-bold">{(value ?? 0).toFixed(4)}</span>
      </div>
      <div className="relative h-2 bg-border rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, backgroundColor: barColor }}
        />
        {threshold != null && (
          <div
            className="absolute top-0 h-full w-0.5 bg-orange opacity-80"
            style={{ left: `${threshold * 100}%` }}
          />
        )}
      </div>
      {threshold != null && (
        <div className="text-right text-xs text-muted mt-0.5">
          threshold {threshold}
        </div>
      )}
    </div>
  )
}
