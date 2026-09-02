import { useState } from 'react'
import useRealtime from './realtime'
import InflowChart from './components/InflowChart'
import LiveBeds from './components/LiveBeds'
import LiveQueue from './components/LiveQueue'
import EventFeed from './components/EventFeed'
import './index.css'

function formatSimClock(iso) {
  if (!iso) return '—'
  const d = new Date(iso)
  return isNaN(d) ? iso : d.toLocaleString()
}

const STATUS_STYLE = {
  connecting: 'bg-slate-400',
  connected: 'bg-emerald-500',
  reconnecting: 'bg-amber-500',
}

function Stat({ label, value }) {
  return (
    <div className="rounded-md bg-white shadow p-3 text-center">
      <div className="text-xl font-bold text-slate-800">{value}</div>
      <div className="text-xs text-slate-500 mt-0.5">{label}</div>
    </div>
  )
}

function App() {
  const { status, clock, snapshot, events, beds, error, control } = useRealtime()
  const [speedInput, setSpeedInput] = useState('')
  const [controlError, setControlError] = useState(null)

  const snapshotForecast = snapshot?.forecast ?? { actual: [], predicted: [] }

  async function runControl(action, speed) {
    setControlError(null)
    try {
      await control(action, speed)
    } catch (err) {
      setControlError(err.message)
    }
  }

  async function applySpeed() {
    const speed = parseFloat(speedInput)
    if (!Number.isFinite(speed) || speed <= 0) return
    await runControl('speed', speed)
  }

  return (
    <div className="min-h-screen bg-slate-100">
      <header className="bg-slate-900 text-white">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between gap-4 flex-wrap">
          <div>
            <h1 className="text-xl font-bold">AHOP Live Dashboard</h1>
            <div className="text-xs text-slate-400 flex items-center gap-2 mt-1">
              <span className={`inline-block w-2 h-2 rounded-full ${STATUS_STYLE[status] ?? 'bg-slate-400'}`} />
              {status === 'connected' ? 'Live · connected' : status === 'reconnecting' ? 'Reconnecting…' : 'Connecting…'}
              {clock?.running && !clock?.paused && <span>· running at {clock.speed}×</span>}
              {clock?.paused && <span>· paused</span>}
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => runControl('start')}
              className="px-3 py-1.5 rounded-md bg-emerald-600 hover:bg-emerald-500 text-sm font-medium disabled:opacity-50"
              disabled={status !== 'connected'}
            >
              Start
            </button>
            {clock?.paused ? (
              <button
                onClick={() => runControl('resume')}
                className="px-3 py-1.5 rounded-md bg-indigo-600 hover:bg-indigo-500 text-sm font-medium disabled:opacity-50"
                disabled={status !== 'connected'}
              >
                Resume
              </button>
            ) : (
              <button
                onClick={() => runControl('pause')}
                className="px-3 py-1.5 rounded-md bg-amber-600 hover:bg-amber-500 text-sm font-medium disabled:opacity-50"
                disabled={status !== 'connected'}
              >
                Pause
              </button>
            )}
            <button
              onClick={() => runControl('reset')}
              className="px-3 py-1.5 rounded-md bg-slate-700 hover:bg-slate-600 text-sm font-medium disabled:opacity-50"
              disabled={status !== 'connected'}
            >
              Reset
            </button>
            <div className="flex items-center gap-1.5">
              <input
                type="number"
                min="0.1"
                step="0.5"
                value={speedInput}
                onChange={(e) => setSpeedInput(e.target.value)}
                placeholder={`${clock?.speed ?? 1}×`}
                className="w-20 px-2 py-1.5 rounded-md bg-slate-800 border border-slate-600 text-sm focus:outline-none focus:border-indigo-400"
              />
              <button
                onClick={applySpeed}
                className="px-3 py-1.5 rounded-md bg-indigo-600 hover:bg-indigo-500 text-sm font-medium disabled:opacity-50"
                disabled={status !== 'connected'}
              >
                Set speed
              </button>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-6 space-y-6">
        {(error || controlError) && (
          <div className="rounded-md bg-red-50 text-red-700 p-4 text-sm">
            {error ?? controlError}
            {error && !controlError && '. Is the API running on port 8000?'}
          </div>
        )}

        <div className="flex items-center justify-between flex-wrap gap-2 text-sm text-slate-600">
          <div className="font-medium">
            Sim clock: <span className="text-slate-900">{formatSimClock(clock?.sim_iso)}</span>
            <span className="ml-2 text-slate-400">({clock?.sim_min?.toLocaleString()} min)</span>
          </div>
          <div className="flex items-center gap-3 text-xs text-slate-500">
            <span>Speed {clock?.speed ?? '—'}×</span>
            <span>{clock?.running ? (clock?.paused ? 'Paused' : 'Running') : 'Stopped'}</span>
            <span>Arrivals left {clock?.arrivals_remaining ?? '—'}</span>
            <span>Events sent {snapshot?.events_sent ?? clock?.events_sent ?? '—'}</span>
          </div>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
          <Stat label="In ED" value={clock?.patients_in_ed ?? 0} />
          <Stat label="Seen" value={clock?.patients_seen ?? 0} />
          <Stat label="Discharged" value={clock?.discharged ?? 0} />
          <Stat label="In Queue" value={snapshot?.queue?.length ?? 0} />
          <Stat label="Admitted" value={snapshot?.admitted?.length ?? 0} />
          <Stat label="Allocations" value={snapshot?.allocations_made ?? 0} />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <InflowChart forecast={snapshotForecast} />
          </div>
          <LiveQueue queue={snapshot?.queue} />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <LiveBeds bedState={beds} />
          </div>
          <EventFeed events={events} />
        </div>
      </main>
    </div>
  )
}

export default App
