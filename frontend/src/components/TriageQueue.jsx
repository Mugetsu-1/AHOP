import { useState } from 'react'
import { assessTriage } from '../api'

const EMPTY = {
  age: 60,
  gender: 'M',
  esi_level: 2,
  chief_complaint: 'Chest pain',
  heart_rate: 96,
  sys_bp: 140,
  dia_bp: 90,
  spo2: 96,
  temp_c: 37.2,
  lactate: 1.8,
  comorbidity_index: 1,
  is_isolation_required: false,
}

const RISK_STYLE = {
  HIGH_RISK: 'bg-red-100 text-red-700',
  MEDIUM_RISK: 'bg-amber-100 text-amber-700',
  LOW_RISK: 'bg-emerald-100 text-emerald-700',
}

export default function TriageQueue() {
  const [form, setForm] = useState(EMPTY)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  function update(key, value) {
    setForm((f) => ({ ...f, [key]: value }))
  }

  async function handleSubmit(e) {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      setResult(await assessTriage(form))
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const numberInput =
    'w-full rounded-md border border-slate-300 px-2 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500'

  return (
    <section className="bg-white rounded-lg shadow p-5">
      <h2 className="text-lg font-semibold text-slate-800 mb-4">Triage Assessment</h2>

      <form onSubmit={handleSubmit} className="grid grid-cols-2 gap-3 text-sm">
        <label className="block">
          <span className="text-slate-600">Age</span>
          <input type="number" className={numberInput} value={form.age} onChange={(e) => update('age', Number(e.target.value))} />
        </label>
        <label className="block">
          <span className="text-slate-600">ESI Level (1–5)</span>
          <input type="number" min="1" max="5" className={numberInput} value={form.esi_level} onChange={(e) => update('esi_level', Number(e.target.value))} />
        </label>
        <label className="block">
          <span className="text-slate-600">Heart Rate</span>
          <input type="number" className={numberInput} value={form.heart_rate} onChange={(e) => update('heart_rate', Number(e.target.value))} />
        </label>
        <label className="block">
          <span className="text-slate-600">Sys BP</span>
          <input type="number" className={numberInput} value={form.sys_bp} onChange={(e) => update('sys_bp', Number(e.target.value))} />
        </label>
        <label className="block">
          <span className="text-slate-600">Dia BP</span>
          <input type="number" className={numberInput} value={form.dia_bp} onChange={(e) => update('dia_bp', Number(e.target.value))} />
        </label>
        <label className="block">
          <span className="text-slate-600">SpO₂</span>
          <input type="number" step="0.1" className={numberInput} value={form.spo2} onChange={(e) => update('spo2', Number(e.target.value))} />
        </label>
        <label className="block">
          <span className="text-slate-600">Temp °C</span>
          <input type="number" step="0.1" className={numberInput} value={form.temp_c} onChange={(e) => update('temp_c', Number(e.target.value))} />
        </label>
        <label className="block">
          <span className="text-slate-600">Lactate</span>
          <input type="number" step="0.1" className={numberInput} value={form.lactate} onChange={(e) => update('lactate', Number(e.target.value))} />
        </label>
        <label className="block col-span-2">
          <span className="text-slate-600">Chief Complaint</span>
          <input type="text" className={numberInput} value={form.chief_complaint} onChange={(e) => update('chief_complaint', e.target.value)} />
        </label>
        <label className="col-span-2 flex items-center gap-2 text-slate-600">
          <input type="checkbox" checked={form.is_isolation_required} onChange={(e) => update('is_isolation_required', e.target.checked)} />
          Isolation required
        </label>
        <button
          type="submit"
          disabled={loading}
          className="col-span-2 rounded-md bg-indigo-600 text-white py-2 font-medium hover:bg-indigo-700 disabled:opacity-50"
        >
          {loading ? 'Assessing…' : 'Assess'}
        </button>
      </form>

      {error && <div className="mt-4 rounded-md bg-red-50 text-red-700 p-3 text-sm">{error}</div>}

      {result && (
        <div className="mt-4 space-y-3">
          <div className="rounded-md bg-slate-50 p-4">
            <div className="flex items-center justify-between mb-3">
              <span className="text-sm text-slate-500 font-mono">{result.patient_id}</span>
              <span className={`text-xs font-semibold px-2 py-1 rounded-full ${RISK_STYLE[result.risk_category] ?? 'bg-slate-200 text-slate-600'}`}>
                {result.risk_category}
              </span>
            </div>
            <div className="grid grid-cols-2 gap-3 text-sm">
              <div>
                <div className="text-slate-500">ICU Escalation Probability</div>
                <div className="text-xl font-bold text-slate-800">
                  {(result.icu_escalation_probability * 100).toFixed(1)}%
                </div>
              </div>
              <div>
                <div className="text-slate-500">Recommended Unit</div>
                <div className="text-xl font-bold text-indigo-600">{result.recommended_unit}</div>
              </div>
            </div>
          </div>
          {result.shap_factors && result.shap_factors.length > 0 && (
            <div>
              <h3 className="text-sm font-semibold text-slate-700 mb-2">Key Risk Drivers</h3>
              <div className="space-y-2">
                {result.shap_factors.map((f) => (
                  <div key={f.feature} className="flex items-center justify-between text-sm">
                    <span className="text-slate-600 capitalize">{f.feature.replace(/_/g, ' ')}</span>
                    <span className={`font-mono ${f.impact >= 0 ? 'text-red-600' : 'text-emerald-600'}`}>
                      {f.impact >= 0 ? '+' : ''}
                      {f.impact.toFixed(4)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </section>
  )
}
