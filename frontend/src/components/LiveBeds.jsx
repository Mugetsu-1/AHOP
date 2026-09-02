const UNIT_COLORS = {
  ICU: 'bg-red-500',
  TELE: 'bg-amber-500',
  GEN: 'bg-emerald-500',
}

function unitStyle(unitName) {
  const key = (unitName ?? '').toUpperCase()
  for (const [prefix, color] of Object.entries(UNIT_COLORS)) {
    if (key.startsWith(prefix)) return color
  }
  return 'bg-sky-500'
}

export default function LiveBeds({ bedState }) {
  const { total = 0, occupied = 0, byUnit = {} } = bedState ?? {}

  const stats = [
    { label: 'Total Beds', value: total, color: 'text-slate-800' },
    { label: 'Occupied', value: occupied, color: 'text-red-600' },
    { label: 'Available', value: Math.max(0, total - occupied), color: 'text-emerald-600' },
  ]

  return (
    <section className="bg-white rounded-lg shadow p-5">
      <h2 className="text-lg font-semibold text-slate-800 mb-3">Live Bed Status</h2>

      <div className="grid grid-cols-3 gap-3 mb-4">
        {stats.map((s) => (
          <div key={s.label} className="rounded-md bg-slate-50 border border-slate-200 p-3 text-center">
            <div className={`text-2xl font-bold ${s.color}`}>{s.value}</div>
            <div className="text-xs text-slate-500 mt-1">{s.label}</div>
          </div>
        ))}
      </div>

      {Object.keys(byUnit).length === 0 ? (
        <p className="text-sm text-slate-400 py-4 text-center">No beds loaded yet.</p>
      ) : (
        <ul className="space-y-2">
          {Object.entries(byUnit).map(([name, u]) => {
            const pct = u.total ? Math.round((u.occupied / u.total) * 100) : 0
            return (
              <li key={name}>
                <div className="flex items-center justify-between text-sm mb-1">
                  <span className="font-medium text-slate-700">{name}</span>
                  <span className="text-xs text-slate-500">
                    {u.occupied}/{u.total}
                  </span>
                </div>
                <div className="h-2 rounded-full bg-slate-200 overflow-hidden">
                  <div
                    className={`h-full rounded-full ${unitStyle(name)}`}
                    style={{ width: `${pct}%` }}
                  />
                </div>
              </li>
            )
          })}
        </ul>
      )}
    </section>
  )
}
