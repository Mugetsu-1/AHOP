import { useMemo } from 'react'

const W = 700
const H = 220
const PAD = { top: 20, right: 16, bottom: 28, left: 40 }

export default function InflowChart({ forecast }) {
  const { actual, predicted } = forecast ?? {}

  const chart = useMemo(() => {
    const points = [...(actual ?? []), ...(predicted ?? [])]
    if (points.length === 0) return null
    const values = points.map((p) => p.value)
    const max = Math.max(...values, 1)
    const min = Math.min(...values, 0)
    const range = max - min || 1
    const innerW = W - PAD.left - PAD.right
    const innerH = H - PAD.top - PAD.bottom

    const x = (i) => PAD.left + (i / Math.max(points.length - 1, 1)) * innerW
    const y = (v) => PAD.top + innerH - ((v - min) / range) * innerH

    const path = (data, startIdx) =>
      data.map((p, i) => `${i === 0 ? 'M' : 'L'}${x(startIdx + i).toFixed(1)},${y(p.value).toFixed(1)}`).join(' ')

    const actualPath = actual && actual.length ? path(actual, 0) : ''
    const predictedPath = predicted && predicted.length ? path(predicted, actual?.length ?? 0) : ''

    return { points, max, min, actualPath, predictedPath, actualLen: actual?.length ?? 0 }
  }, [actual, predicted])

  if (!chart) return null

  const dividerX = chart.actualLen > 0 ? PAD.left + (chart.actualLen / Math.max(chart.points.length - 1, 1)) * (W - PAD.left - PAD.right) : null

  return (
    <section className="bg-white rounded-lg shadow p-5">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-lg font-semibold text-slate-800">Hourly Arrivals (24h)</h2>
        <div className="flex gap-3 text-xs text-slate-500">
          <span className="flex items-center gap-1">
            <span className="inline-block w-3 h-3 rounded-full bg-indigo-500" /> Actual
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block w-3 h-3 rounded-full bg-amber-500" /> Predicted
          </span>
        </div>
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Hourly arrival forecast">
        {[0, 0.25, 0.5, 0.75, 1].map((t) => {
          const v = chart.min + t * (chart.max - chart.min)
          return (
            <g key={t}>
              <line
                x1={PAD.left}
                x2={W - PAD.right}
                y1={PAD.top + t * (H - PAD.top - PAD.bottom)}
                y2={PAD.top + t * (H - PAD.top - PAD.bottom)}
                stroke="#e2e8f0"
              />
              <text x={PAD.left - 6} y={PAD.top + t * (H - PAD.top - PAD.bottom) + 3} textAnchor="end" className="fill-slate-400" fontSize="10">
                {Math.round(v)}
              </text>
            </g>
          )
        })}
        {dividerX !== null && (
          <line x1={dividerX} x2={dividerX} y1={PAD.top} y2={H - PAD.bottom} stroke="#cbd5e1" strokeDasharray="4 4" />
        )}
        {chart.actualPath && (
          <path d={chart.actualPath} fill="none" stroke="#6366f1" strokeWidth="2.5" />
        )}
        {chart.predictedPath && (
          <path d={chart.predictedPath} fill="none" stroke="#f59e0b" strokeWidth="2.5" strokeDasharray="6 4" />
        )}
        {chart.points.map((p, i) => (
          <circle
            key={i}
            cx={PAD.left + (i / Math.max(chart.points.length - 1, 1)) * (W - PAD.left - PAD.right)}
            cy={PAD.top + (H - PAD.top - PAD.bottom) - ((p.value - chart.min) / (chart.max - chart.min || 1)) * (H - PAD.top - PAD.bottom)}
            r="2.5"
            className={i < chart.actualLen ? 'fill-indigo-500' : 'fill-amber-500'}
          />
        ))}
      </svg>
    </section>
  )
}
