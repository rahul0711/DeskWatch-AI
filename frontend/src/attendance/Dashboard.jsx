import { useEffect, useRef, useState } from 'react'
import { AuthError, getToken } from '../api'
import { fetchAttendance } from './api'
import './Dashboard.css'

const MAX_UNKNOWN = 12
const MAX_RECONNECT_DELAY_MS = 10000

function todayStr() {
  return new Date().toISOString().slice(0, 10)
}

export default function Dashboard({ onAuthError, onEnrollFromCrop }) {
  const [frame, setFrame] = useState(null) // { image, camera_name, timestamp } -- boxes are pre-drawn server-side
  const [attendance, setAttendance] = useState([])
  const [unknowns, setUnknowns] = useState([])
  const [wsStatus, setWsStatus] = useState('connecting')
  const [cooldownAlert, setCooldownAlert] = useState(null)
  const [punchAlert, setPunchAlert] = useState(null)
  const [voiceEnabled, setVoiceEnabled] = useState(true)

  const wsRef = useRef(null)
  const reconnectDelayRef = useRef(1000)
  const lastSpokenRef = useRef({})
  const voiceEnabledRef = useRef(voiceEnabled)
  voiceEnabledRef.current = voiceEnabled

  const speak = (text, key) => {
    if (!voiceEnabledRef.current) return
    if (typeof window === 'undefined' || !('speechSynthesis' in window)) return
    const now = Date.now()
    const last = lastSpokenRef.current[key] || 0
    // Throttle speech for the same user/key so it doesn't repeat within 15s
    if (now - last < 15000) return
    lastSpokenRef.current[key] = now

    try {
      window.speechSynthesis.cancel() // stop any previous utterance
      const utterance = new SpeechSynthesisUtterance(text)
      utterance.rate = 1.0
      utterance.pitch = 1.0
      window.speechSynthesis.speak(utterance)
    } catch {
      // ignore speech errors in browsers with strict audio autoplay policies
    }
  }

  useEffect(() => {
    let timer
    if (cooldownAlert) {
      timer = setTimeout(() => setCooldownAlert(null), 6000)
    }
    return () => clearTimeout(timer)
  }, [cooldownAlert])

  useEffect(() => {
    let timer
    if (punchAlert) {
      timer = setTimeout(() => setPunchAlert(null), 6000)
    }
    return () => clearTimeout(timer)
  }, [punchAlert])

  // Auto-expire unknown faces after 30 seconds (or ttl_seconds)
  useEffect(() => {
    const timer = setInterval(() => {
      const now = Date.now()
      setUnknowns((prev) => {
        const remaining = prev.filter((u) => {
          const ttlMs = (u.ttl_seconds || 30) * 1000
          const age = now - (u.receivedAt || Date.parse(u.timestamp) || now)
          return age < ttlMs
        })
        return remaining.length === prev.length ? prev : remaining
      })
    }, 1000)
    return () => clearInterval(timer)
  }, [])

  useEffect(() => {
    let cancelled = false
    fetchAttendance(todayStr())
      .then((rows) => {
        if (!cancelled) setAttendance(rows.slice().reverse())
      })
      .catch((err) => {
        if (err instanceof AuthError) onAuthError()
      })
    return () => {
      cancelled = true
    }
  }, [onAuthError])

  useEffect(() => {
    let cancelled = false
    let ws

    function connect() {
      if (cancelled) return
      const token = getToken()
      const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
      ws = new WebSocket(`${proto}://${window.location.host}/ws/events?token=${encodeURIComponent(token || '')}`)
      wsRef.current = ws

      ws.onopen = () => {
        setWsStatus('connected')
        reconnectDelayRef.current = 1000
      }
      ws.onclose = () => {
        if (cancelled) return
        setWsStatus('disconnected')
        const delay = reconnectDelayRef.current
        reconnectDelayRef.current = Math.min(delay * 2, MAX_RECONNECT_DELAY_MS)
        setTimeout(connect, delay)
      }
      ws.onerror = () => ws.close()
      ws.onmessage = (evt) => {
        const msg = JSON.parse(evt.data)
        if (msg.type === 'frame') {
          setFrame(msg)
        } else if (msg.type === 'attendance') {
          setAttendance((prev) => {
            const now = Date.now()
            const msgTime = msg.timestamp ? new Date(msg.timestamp).getTime() : now
            const isDuplicate = prev.some((a) => {
              if (a.user_id !== msg.user_id) return false
              const aTime = a.timestamp ? new Date(a.timestamp).getTime() : now
              return Math.abs(msgTime - aTime) < 600000 // 10 minutes
            })
            if (isDuplicate) {
              return prev
            }
            return [msg, ...prev].slice(0, 100)
          })
          setPunchAlert({
            id: Date.now(),
            name: msg.name,
            employee_id: msg.employee_id,
            message: msg.message || `Punch registered for ${msg.name}`,
          })
          speak(`${msg.name}, attendance marked`, `att-${msg.user_id}`)
        } else if (msg.type === 'cooldown') {
          setCooldownAlert({
            id: Date.now(),
            name: msg.name,
            employee_id: msg.employee_id,
            message: msg.message || "You're done punching for like 10 minutes",
            remaining_minutes: msg.remaining_minutes,
          })
          speak(msg.spoken_message || `${msg.name}, you are done punching`, `cool-${msg.user_id}`)
        } else if (msg.type === 'unknown') {
          const entry = { ...msg, receivedAt: Date.now() }
          setUnknowns((prev) => [entry, ...prev.filter((u) => u.track_id !== msg.track_id)].slice(0, MAX_UNKNOWN))
        }
      }
    }

    connect()
    return () => {
      cancelled = true
      wsRef.current?.close()
    }
  }, [])

  return (
    <div className="dashboard">
      <section className="dashboard__panel dashboard__live">
        <div className="dashboard__panel-header">
          <h2>
            Live view {frame ? `— ${frame.camera_name}` : ''}
            <span className={`dashboard__ws-status dashboard__ws-status--${wsStatus}`}>{wsStatus}</span>
          </h2>
          <button
            type="button"
            className={`dashboard__voice-btn ${voiceEnabled ? 'dashboard__voice-btn--active' : ''}`}
            onClick={() => setVoiceEnabled((v) => !v)}
            title={voiceEnabled ? 'Voice feedback is ON (click to mute)' : 'Voice feedback is MUTED (click to enable)'}
          >
            {voiceEnabled ? '🔊 Voice ON' : '🔇 Mute'}
          </button>
        </div>

        {cooldownAlert && (
          <div className="dashboard__banner dashboard__banner--cooldown" role="alert">
            <span className="dashboard__banner-icon">⏳</span>
            <div className="dashboard__banner-text">
              <strong>{cooldownAlert.name}</strong> ({cooldownAlert.employee_id}):{' '}
              <span>{cooldownAlert.message}</span>
            </div>
          </div>
        )}

        {punchAlert && !cooldownAlert && (
          <div className="dashboard__banner dashboard__banner--punch" role="alert">
            <span className="dashboard__banner-icon">✅</span>
            <div className="dashboard__banner-text">
              <strong>{punchAlert.name}</strong> ({punchAlert.employee_id}):{' '}
              <span>{punchAlert.message}</span>
            </div>
          </div>
        )}

        <div className="dashboard__live-frame">
          {frame ? (
            <img src={frame.image} alt="live camera feed" />
          ) : (
            <p className="dashboard__hint">Waiting for first frame…</p>
          )}
        </div>
      </section>

      <section className="dashboard__panel">
        <h2>Today's attendance ({attendance.length})</h2>
        <div className="dashboard__attendance-list">
          {attendance.length === 0 && <p className="dashboard__hint">No attendance events yet today.</p>}
          {attendance.map((a, i) => (
            <div key={a.id ?? `${a.track_id}-${i}`} className="dashboard__attendance-row">
              <img src={a.thumbnail_url} alt={a.name} />
              <div className="dashboard__attendance-info">
                <strong>{a.name}</strong>
                <span>{a.employee_id}</span>
                <span>{new Date(a.timestamp).toLocaleTimeString()}</span>
                <span>
                  conf {Number(a.confidence).toFixed(2)} · {a.face_width_px}px
                  {a.low_confidence && <span className="dashboard__low-conf"> low-conf</span>}
                </span>
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="dashboard__panel">
        <h2>Unknown faces</h2>
        <div className="dashboard__unknown-grid">
          {unknowns.length === 0 && <p className="dashboard__hint">No unrecognized faces recently.</p>}
          {unknowns.map((u, i) => {
            const ttl = (u.ttl_seconds || 30) * 1000
            const age = Date.now() - (u.receivedAt || Date.parse(u.timestamp) || Date.now())
            const secondsLeft = Math.max(0, Math.ceil((ttl - age) / 1000))
            return (
              <div key={`${u.track_id}-${i}`} className="dashboard__unknown-card">
                <img src={u.thumbnail_url} alt="unknown face" />
                <span className="dashboard__unknown-timer">⏳ {secondsLeft}s left</span>
                <span className="dashboard__hint">
                  {u.face_width_px}px{u.best_score != null ? ` · best ${u.best_score.toFixed(2)}` : ''}
                </span>
                <button type="button" onClick={() => onEnrollFromCrop(u.thumbnail_url)}>
                  Enroll this person
                </button>
              </div>
            )
          })}
        </div>
      </section>
    </div>
  )
}
