const TYPE_STYLE = {
  PATIENT_ARRIVED: 'bg-indigo-100 text-indigo-700',
  telemetry: 'bg-sky-100 text-sky-700',
  PATIENT_DISCHARGED: 'bg-emerald-100 text-emerald-700',
  BED_ALLOCATED: 'bg-amber-100 text-amber-700',
}

const TYPE_LABEL = {
  PATIENT_ARRIVED: 'Arrival',
  telemetry: 'Telemetry',
  PATIENT_DISCHARGED: 'Discharge',
  BED_ALLOCATED: 'Allocation',
}

export default function EventFeed({ events }) {
  const list = events ?? []
  return (
    <section className="bg-white rounded-lg shadow p-5">
      <h2 className="text-lg font-semibold text-slate-800 mb-3">Live Events</h2>
      {list.length === 0 ? (
        <p className="text-sm text-slate-400 py-6 text-center">Waiting for events…</p>
      ) : (
        <ul className="space-y-2 max-h-[32rem] overflow-y-auto">
          {list.map((ev) => (
            <li key={ev.id} className="flex items-start gap-2 text-sm">
              <span
                className={`shrink-0 mt-0.5 text-[10px] font-semibold px-1.5 py-0.5 rounded ${
                  TYPE_STYLE[ev.type] ?? 'bg-slate-100 text-slate-600'
                }`}
              >
                {TYPE_LABEL[ev.type] ?? ev.type}
              </span>
              <div className="min-w-0 flex-1">
                <div className="text-slate-600 truncate">{ev.message}</div>
                <div className="text-[10px] text-slate-400">{ev.timestamp}</div>
              </div>
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}
