const TIER_STYLE = {
  HIGH: 'bg-red-100 text-red-700',
  MEDIUM: 'bg-amber-100 text-amber-700',
  LOW: 'bg-emerald-100 text-emerald-700',
}

export default function LiveQueue({ queue }) {
  const patients = queue ?? []
  const sorted = [...patients].sort((a, b) => b.icu_risk - a.icu_risk)

  return (
    <section className="bg-white rounded-lg shadow p-5">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-lg font-semibold text-slate-800">Live Queue</h2>
        <span className="text-xs text-slate-500">{patients.length} waiting</span>
      </div>

      {sorted.length === 0 ? (
        <p className="text-sm text-slate-400 py-6 text-center">No patients waiting.</p>
      ) : (
        <ul className="space-y-2 max-h-96 overflow-y-auto">
          {sorted.map((p) => (
            <li key={p.patient_id} className="border border-slate-200 rounded-md p-3 text-sm">
              <div className="flex items-center justify-between mb-1">
                <span className="font-medium text-slate-800 truncate">{p.chief_complaint}</span>
                <span
                  className={`shrink-0 ml-2 text-xs font-semibold px-2 py-0.5 rounded-full ${
                    TIER_STYLE[p.risk_tier] ?? TIER_STYLE.LOW
                  }`}
                >
                  {p.risk_tier}
                </span>
              </div>
              <div className="flex items-center justify-between text-xs text-slate-500">
                <span>
                  ESI {p.esi_level} · {p.gender} · {p.isolation_required ? 'Isolation' : 'Standard'}
                </span>
                <span>wait {p.wait_minutes}m</span>
              </div>
              <div className="mt-1 flex items-center justify-between text-xs text-slate-400">
                <span>Patient {p.patient_id}</span>
                <span>ICU risk {(p.icu_risk * 100).toFixed(0)}%</span>
              </div>
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}
