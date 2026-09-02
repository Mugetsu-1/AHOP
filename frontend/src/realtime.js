import { useCallback, useEffect, useRef, useState } from 'react'
import { sendControl } from './api'

const RECONNECT_BASE_MS = 1000
const RECONNECT_MAX_MS = 15000
const MAX_EVENTS = 50

// Control frames update state but are never appended to the event feed.
const CONTROL_TYPES = new Set(['hello', 'clock', 'snapshot', 'queue_update'])

function buildMessage(type, payload) {
  const p = payload ?? {}
  switch (type) {
    case 'PATIENT_ARRIVED':
      return `Patient ${p.patient_id} arrived · ESI ${p.esi_level}`
    case 'telemetry':
      return `Patient ${p.patient_id} vitals updated`
    case 'PATIENT_DISCHARGED':
      return `Patient ${p.patient_id} → ${p.disposition ?? 'discharged'}`
    case 'BED_ALLOCATED':
    case 'PATIENT_ADMITTED':
      return `Patient ${p.patient_id} allocated to ${p.unit_name ?? 'a bed'}`
    default:
      return `Event: ${type}`
  }
}

// Tolerate {type, payload}, {event_type, data}, and flat {event, ...} frames.
function parseFrame(data) {
  if (!data || typeof data !== 'object') return null
  const type = data.type ?? data.event_type ?? data.event
  const payload = data.payload ?? data.data ?? data
  if (typeof type !== 'string' || type.length === 0) return null
  return { type, payload }
}

function reconcileBeds(payload) {
  const beds = Array.isArray(payload?.beds) ? payload.beds : []
  const byUnit = {}
  let occupied = 0
  for (const bed of beds) {
    const name = bed?.unit_name ?? 'Unknown'
    byUnit[name] ??= { total: 0, occupied: 0 }
    byUnit[name].total += 1
    if (bed?.status === 'OCCUPIED') {
      byUnit[name].occupied += 1
      occupied += 1
    }
  }
  return {
    total: payload?.bed_summary?.total ?? beds.length,
    occupied: payload?.bed_summary?.occupied ?? occupied,
    byUnit,
  }
}

function appendEvent(type, payload) {
  return (prev) =>
    [
      {
        id: Math.random().toString(36).slice(2),
        timestamp: new Date().toLocaleTimeString(),
        type,
        message: buildMessage(type, payload),
      },
      ...prev,
    ].slice(0, MAX_EVENTS)
}

export function useRealtime() {
  const [status, setStatus] = useState('connecting')
  const [clock, setClock] = useState(null)
  const [snapshot, setSnapshot] = useState(null)
  const [events, setEvents] = useState([])
  const [beds, setBeds] = useState({ total: 0, occupied: 0, byUnit: {} })
  const [error, setError] = useState(null)
  const wsRef = useRef(null)
  const retryRef = useRef(0)
  const reconnectTimerRef = useRef(null)
  const connectRef = useRef(null)

  const handleMessage = useCallback((msg) => {
    if (!msg || typeof msg.type !== 'string') return
    console.log('WS Data:', msg)
    const { type, payload } = msg

    switch (type) {
      case 'hello':
      case 'clock':
        setClock(payload)
        break
      case 'snapshot':
        setSnapshot(payload)
        setClock(payload.clock)
        setBeds(reconcileBeds(payload))
        break
      case 'queue_update':
        setSnapshot((prev) => (prev ? { ...prev, queue: payload.queue } : prev))
        break
      case 'PATIENT_ARRIVED':
      case 'telemetry':
        setEvents(appendEvent(type, payload))
        break
      case 'PATIENT_DISCHARGED':
        setEvents(appendEvent(type, payload))
        if (payload?.unit_name || payload?.bed_id) {
          setBeds((prev) => {
            const name = payload.unit_name ?? null
            const byUnit = name ? { ...prev.byUnit } : prev.byUnit
            if (name) {
              const unit = byUnit[name] ?? { total: 0, occupied: 0 }
              byUnit[name] = { ...unit, occupied: Math.max(0, unit.occupied - 1) }
            }
            return { ...prev, occupied: Math.max(0, prev.occupied - 1), byUnit }
          })
        }
        break
      case 'BED_ALLOCATED':
      case 'PATIENT_ADMITTED':
        setEvents(appendEvent(type, payload))
        if (payload?.unit_name) {
          setBeds((prev) => {
            const byUnit = { ...prev.byUnit }
            const unit = byUnit[payload.unit_name] ?? { total: 0, occupied: 0 }
            byUnit[payload.unit_name] = { ...unit, occupied: unit.occupied + 1 }
            return { ...prev, occupied: prev.occupied + 1, byUnit }
          })
        }
        break
      default:
        if (!CONTROL_TYPES.has(type)) setEvents(appendEvent(type, payload))
        break
    }
  }, [])

  const connect = useCallback(() => {
    wsRef.current?.close()
    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const url = `${proto}://${window.location.host}/api/v1/realtime/ws`
    const ws = new WebSocket(url)
    wsRef.current = ws

    ws.onopen = () => {
      retryRef.current = 0
      setStatus('connected')
      setError(null)
    }
    ws.onmessage = (ev) => {
      try {
        const msg = parseFrame(JSON.parse(ev.data))
        if (msg) handleMessage(msg)
      } catch {
        /* ignore malformed frame */
      }
    }
    ws.onerror = () => setError('Realtime connection error')
    ws.onclose = () => {
      setStatus('reconnecting')
      const delay = Math.min(RECONNECT_BASE_MS * 2 ** retryRef.current, RECONNECT_MAX_MS)
      retryRef.current += 1
      reconnectTimerRef.current = setTimeout(() => connectRef.current(), delay)
    }
  }, [handleMessage])

  useEffect(() => {
    connectRef.current = connect
  }, [connect])

  useEffect(() => {
    connect()
    return () => {
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current)
      wsRef.current?.close()
    }
  }, [connect])

  const control = useCallback(async (action, speed) => {
    const res = await sendControl(action, speed)
    if (res?.sim_iso) setClock(res)
    return res
  }, [])

  return { status, clock, snapshot, events, beds, error, control }
}

export default useRealtime
