import { useState } from 'react'
import { runAllocation } from '../api'

const UNIT_COLORS = {
  ICU: 'bg-red-500',
  TELE: 'bg-amber-500',
  GEN: 'bg-emerald-500',
}

function unitStyle(name) {
  for (const [prefix, cls] of Object.entries(UNIT_COLORS)) {
    if (name.startsWith(prefix)) return cls
  }
  return 'bg-sky-500'
}

export default function BedMatrix({ occupancy }) {
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  async function handleRun() {
    setRunning(true)
    setError(null)
    try {
      setResult(await runAllocation())
    } catch (err) {
      setError(err.message)
    } finally {
      setRunning(false)
    }
  }

  if (!occupancy) return null

  return (
    <section className="bg-white rounded-lg shadow p-5">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-slate-800">Bed Matrix</h2>
        <button
          onClick={handleRun}
          disabled={running}
          className="px-4 py-2 rounded-md bg-indigo-600 text-white text-sm font-medium hover:bg-indigo-700 disabled:opacity-50"
        >
          {running ? 'Optimizing…' : 'Run Allocation'}
        </button>
      </div>

      <div className="flex gap-4 mb-4 text-sm">
        <div className="flex-1 rounded-md bg-slate-50 p-3">
          <div className="text-slate-500">Total Beds</div>
          <div className="text-2xl font-bold text-slate-800">{occupancy.total_beds}</div>
        </div>
        <div className="flex-1 rounded-md bg-slate-50 p-3">
          <div className="text-slate-500">Occupied</div>
          <div className="text-2xl font-bold text-red-600">{occupancy.occupied_beds}</div>
        </div>
        <div className="flex-1 rounded-md bg-slate-50 p-3">
          <div className="text-slate-500">Available</div>
          <div className="text-2xl font-bold text-emerald-600">{occupancy.available_beds}</div>
        </div>
        <div className="flex-1 rounded-md bg-slate-50 p-3">
          <div className="text-slate-500">Occupancy</div>
          <div className="text-2xl font-bold text-indigo-600">{occupancy.occupancy_pct}%</div>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {occupancy.by_unit.map((unit) => {
          const pct = unit.total ? Math.round((unit.occupied / unit.total) * 100) : 0
          return (
            <div key={unit.unit_name} className="border rounded-md p-3">
              <div className="flex items-center justify-between mb-2">
                <span className="font-medium text-slate-700">{unit.unit_name}</span>
                <span className="text-xs text-slate-500">
                  {unit.occupied}/{unit.total}
                </span>
              </div>
              <div className="h-2.5 rounded-full bg-slate-200 overflow-hidden">
                <div
                  className={`h-full ${unitStyle(unit.unit_name)}`}
                  style={{ width: `${pct}%` }}
                />
              </div>
              <div className="mt-1 text-xs text-slate-500">{unit.available} available</div>
            </div>
          )
        })}
      </div>

      {error && (
        <div className="mt-4 rounded-md bg-red-50 text-red-700 p-3 text-sm">{error}</div>
      )}

      {result && (
        <div className="mt-4">
          <h3 className="text-sm font-semibold text-slate-700 mb-2">
            Allocation Result
            <span className="ml-2 text-xs font-normal text-slate-500">
              status: {result.solver_status} · {result.execution_time_ms} ms
            </span>
          </h3>
          {result.assignments_made === 0 ? (
            <p className="text-sm text-slate-500">No pending patients to assign.</p>
          ) : (
            <div className="max-h-64 overflow-y-auto rounded-md border">
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-slate-50 text-slate-500">
                  <tr>
                    <th className="text-left p-2">Patient</th>
                    <th className="text-left p-2">Unit</th>
                    <th className="text-left p-2">Bed</th>
                    <th className="text-right p-2">Wait Δ (min)</th>
                  </tr>
                </thead>
                <tbody>
                  {result.allocations.map((a) => (
                    <tr key={a.assigned_bed_id} className="border-t">
                      <td className="p-2 font-mono text-xs">{a.patient_id}</td>
                      <td className="p-2">{a.unit_name}</td>
                      <td className="p-2">{a.bed_number}</td>
                      <td className="p-2 text-right">-{a.expected_wait_reduction_min}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </section>
  )
}
