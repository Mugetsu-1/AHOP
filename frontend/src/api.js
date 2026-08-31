const BASE = ''

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      if (body.detail) detail = Array.isArray(body.detail) ? JSON.stringify(body.detail) : body.detail
    } catch {
      /* ignore */
    }
    throw new Error(detail)
  }
  return res.json()
}

export function getMetrics() {
  return request('/api/v1/dashboard/metrics')
}

export function runAllocation(maxSolverTimeSec = 2.0, enforceStrictIsolation = true) {
  return request('/api/v1/allocation/optimize', {
    method: 'POST',
    body: JSON.stringify({ max_solver_time_sec: maxSolverTimeSec, enforce_strict_isolation: enforceStrictIsolation }),
  })
}

export function assessTriage(payload) {
  return request('/api/v1/triage/assess', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}
