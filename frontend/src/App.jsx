import { useEffect, useState } from 'react'
import { getMetrics } from './api'
import BedMatrix from './components/BedMatrix'
import InflowChart from './components/InflowChart'
import TriageQueue from './components/TriageQueue'
import './index.css'

function App() {
  const [metrics, setMetrics] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(true)

  async function refresh() {
    try {
      setMetrics(await getMetrics())
      setError(null)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    refresh()
    const id = setInterval(refresh, 60000)
    return () => clearInterval(id)
  }, [])

  return (
    <div className="min-h-screen bg-slate-100">
      <header className="bg-slate-900 text-white">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <h1 className="text-xl font-bold">AHOP Bed Allocation Dashboard</h1>
          <div className="text-xs text-slate-400">
            {metrics ? `Last updated ${new Date(metrics.last_updated_utc).toLocaleTimeString()}` : 'Loading…'}
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-6 space-y-6">
        {error && (
          <div className="rounded-md bg-red-50 text-red-700 p-4 text-sm">
            Failed to reach the backend: {error}. Is the API running on port 8000?
          </div>
        )}
        {loading && !metrics && (
          <div className="text-center text-slate-500 py-20">Loading dashboard…</div>
        )}
        {metrics && (
          <>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2">
                <InflowChart forecast={metrics.arrival_forecast} />
              </div>
              <TriageQueue />
            </div>
            <BedMatrix occupancy={metrics.bed_occupancy} />
          </>
        )}
      </main>
    </div>
  )
}

export default App
