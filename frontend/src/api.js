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

export function sendControl(action, speed) {
  return request('/api/v1/realtime/control', {
    method: 'POST',
    body: JSON.stringify(action === 'speed' && speed != null ? { action, speed } : { action }),
  })
}
