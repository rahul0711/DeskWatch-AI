import { useCallback, useEffect, useRef, useState } from 'react'
import { AuthError, fetchCameras, setViewMode, streamUrl } from './api'
import ActivityTable from './ActivityTable'
import './LiveCameras.css'

// True while the browser tab is visible. Hidden tabs drop their video
// streams and pause polling (see LiveStream / load below).
function usePageVisible() {
  const [visible, setVisible] = useState(() => !document.hidden)
  useEffect(() => {
    const onChange = () => setVisible(!document.hidden)
    document.addEventListener('visibilitychange', onChange)
    return () => document.removeEventListener('visibilitychange', onChange)
  }, [])
  return visible
}

// One camera's MJPEG stream. Each stream is a request that never finishes,
// and the browser allows only 6 connections per site -- so a stream must be
// released the moment it is not needed, or the page runs out of
// connections and every request (video, tables, status) freezes:
//   * no stream for a camera that isn't connected (nothing to show anyway)
//   * no stream while the tab is hidden
//   * on unmount, src is cleared explicitly: removing an <img> does not
//     reliably abort its multipart request in Chrome until garbage collection.
// src is set here rather than as a JSX prop: StrictMode runs this effect's
// cleanup and then the effect again on mount, and React would not re-apply an
// unchanged src prop after the cleanup blanked it.
function LiveStream({ cam, pageVisible }) {
  const ref = useRef(null)
  const online = cam.connection_state === 'connected'
  const show = online && pageVisible
  useEffect(() => {
    const img = ref.current
    if (!img) return undefined
    img.src = streamUrl(cam.id)
    return () => {
      img.src = ''
    }
  }, [cam.id, show])
  if (!show) {
    return (
      <div className="live__video live__placeholder">
        {online ? 'Paused while this tab is hidden' : `Camera ${cam.connection_state || 'offline'} — no video`}
      </div>
    )
  }
  return <img ref={ref} className="live__video" alt={`${cam.name} live`} />
}

const STATE_COLORS = {
  connected: 'var(--ok)',
  connecting: 'var(--warn)',
  reconnecting: 'var(--warn)',
  stopped: 'var(--bad)',
}

// Fast enough for the live phone-violation banner to feel immediate.
const POLL_MS = 1500

// Body-movement sub-views. These narrow what the overlay draws; they do not
// change what is detected or recorded, so the Activity table stays complete
// whichever one is selected.
const ACTIVITY_FILTERS = [
  { key: 'all', label: 'All', hint: 'Every person, with all activity labels' },
  { key: 'phone', label: 'Phone tracking', hint: 'Only people detected using a phone' },
  { key: 'body', label: 'Body tracking', hint: 'Posture and movement only -- sitting, standing, walking, movement score' },
  { key: 'monitor', label: 'Monitor', hint: 'Only people facing a detected monitor' },
  { key: 'computer', label: 'Computer', hint: 'Only people with hands at a laptop, keyboard or mouse' },
  { key: 'debug', label: 'Debug', hint: 'Raw measurements: posture, head yaw/pitch, phone bbox + hand distance, monitor bbox + distance, movement score, skeleton' },
]

// Live view of every configured camera, with a selector to watch one camera
// full-width or all of them side by side. Video is the MJPEG stream served by
// app/web/server.py; boxes are drawn server-side.
function fmtSeconds(sec) {
  const s = Math.round(sec || 0)
  return s >= 60 ? `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, '0')}s` : `${s}s`
}

// Live-view focus (app/web/server.py _coordinate_focus).
const FOCUS_LABELS = { phone: 'phones', body: 'posture', monitor: 'monitor use', computer: 'computer use' }

export default function LiveCameras({ onAuthError }) {
  const [cameras, setCameras] = useState([])
  const [error, setError] = useState(null)
  const [selected, setSelected] = useState('all') // 'all' | camera id
  const [busy, setBusy] = useState(false)
  const pageVisible = usePageVisible()
  const inFlight = useRef(false)

  // One status request at a time, and none while the tab is hidden -- a
  // timer that keeps firing while earlier requests are still queued is what
  // snowballs a slow moment into a frozen page.
  const load = useCallback(async () => {
    if (inFlight.current || document.hidden) return
    inFlight.current = true
    try {
      const list = await fetchCameras()
      setCameras(list)
      setError(null)
    } catch (err) {
      if (err instanceof AuthError) onAuthError()
      else setError('Camera server not reachable -- is app/web/server.py running?')
    } finally {
      inFlight.current = false
    }
  }, [onAuthError])

  useEffect(() => {
    load()
    const timer = setInterval(load, POLL_MS)
    return () => clearInterval(timer)
  }, [load, pageVisible])

  // Applies to every visible camera that supports it, so the choice doesn't
  // have to be repeated per tile in the side-by-side view.
  const applyView = useCallback(
    async (targets, patch) => {
      setBusy(true)
      try {
        await Promise.all(
          targets
            .filter((c) => (patch.viewMode === 'face' ? c.face_enabled !== false : c.activity_available))
            .map((c) => setViewMode(c.id, patch)),
        )
        setError(null)
        await load()
      } catch (err) {
        if (err instanceof AuthError) onAuthError()
        else setError(err.message)
      } finally {
        setBusy(false)
      }
    },
    [load, onAuthError],
  )

  const visible = selected === 'all' ? cameras : cameras.filter((c) => c.id === selected)
  const allLabel = cameras.length === 2 ? 'Both' : 'All'
  // The controls reflect the first visible camera that has activity.
  const lead = visible.find((c) => c.activity_available) || visible[0]
  const viewMode = lead?.view_mode || 'face'
  const activityFilter = lead?.activity_filter || 'all'
  const anyActivity = visible.some((c) => c.activity_available)
  const anyFace = visible.some((c) => c.face_enabled !== false)

  return (
    <div className="live">
      <div className="camera-selector">
        <span className="camera-selector__label">Camera</span>
        {cameras.map((cam, i) => (
          <button
            key={cam.id}
            className={`camera-tab${selected === cam.id ? ' camera-tab--active' : ''}`}
            onClick={() => setSelected(cam.id)}
          >
            Camera {i + 1} · {cam.name}
          </button>
        ))}
        {cameras.length > 1 && (
          <button
            className={`camera-tab${selected === 'all' ? ' camera-tab--active' : ''}`}
            onClick={() => setSelected('all')}
          >
            {allLabel}
          </button>
        )}
      </div>

      <div className="camera-selector">
        <span className="camera-selector__label">Detect</span>
        <button
          className={`camera-tab${viewMode === 'face' ? ' camera-tab--active' : ''}`}
          disabled={busy || !anyFace}
          onClick={() => applyView(visible, { viewMode: 'face' })}
          title={anyFace ? 'Face detection and recognition boxes' : 'Face is switched off for this camera (Admin → Cameras)'}
        >
          Face
        </button>
        <button
          className={`camera-tab${viewMode === 'activity' ? ' camera-tab--active' : ''}`}
          disabled={busy || !anyActivity}
          onClick={() => applyView(visible, { viewMode: 'activity' })}
          title={
            anyActivity
              ? 'Body movement: posture, movement, phone, monitor and computer interaction'
              : 'Body movement is switched off for this camera (Admin → Cameras)'
          }
        >
          Body movement
        </button>
      </div>

      {viewMode === 'activity' && anyActivity && (
        <div className="camera-selector camera-selector--sub">
          <span className="camera-selector__label">Show</span>
          {ACTIVITY_FILTERS.map((f) => (
            <button
              key={f.key}
              className={`camera-tab${activityFilter === f.key ? ' camera-tab--active' : ''}`}
              disabled={busy}
              title={f.hint}
              onClick={() => applyView(visible, { activityFilter: f.key })}
            >
              {f.label}
            </button>
          ))}
          <span className="camera-selector__note">
            Phone, Body, Monitor and Computer focus the GPU on the camera you are watching (other cameras pause body
            movement until you leave). All and Debug keep every camera running.
          </span>
        </div>
      )}

      {error && <div className="app__error">{error}</div>}
      {!error && cameras.length === 0 && <div className="app__loading">Loading cameras…</div>}

      <div className={`live__grid${visible.length > 1 ? ' live__grid--multi' : ''}`}>
        {visible.map((cam) => (
          <section key={cam.id} className="live__tile">
            <header className="live__header">
              <span
                className="live__dot"
                style={{ background: STATE_COLORS[cam.connection_state] || 'var(--text-dim)' }}
              />
              <span className="live__name">{cam.name}</span>
              <span className="live__meta">
                {cam.connection_state} · {cam.capture_fps?.toFixed(0)} fps ·{' '}
                {cam.view_mode === 'activity'
                  ? `${cam.activity_people} tracked`
                  : `${cam.faces} face${cam.faces === 1 ? '' : 's'}`}
              </span>
            </header>
            {cam.access_alerts?.length > 0 && (
              <div className="live__alert" role="alert">
                🚫 {cam.access_alerts.map((a) => a.name).join(', ')}
                {cam.access_alerts.length === 1 ? ' is' : ' are'} not welcome here
                <span className="live__alert-time"> · seen {cam.access_alerts[0].last_seen?.slice(11)}</span>
              </div>
            )}
            {cam.phone_violations?.length > 0 && (
              <div className="live__alert live__alert--phone" role="alert">
                📵 Phone use:{' '}
                {cam.phone_violations
                  .map((v) => `${v.name} · ${fmtSeconds(v.seconds)} · ${Math.round(v.confidence * 100)}%`)
                  .join('  |  ')}
              </div>
            )}
            {cam.activity_focus && (
              <div className="live__focus">
                Focused on {FOCUS_LABELS[cam.activity_focus] || cam.activity_focus} — body movement on the other
                cameras is paused while you watch, so this one gets the GPU.
              </div>
            )}
            {cam.activity_paused && (
              <div className="live__focus live__focus--paused">
                Body movement paused — another camera is being watched in a focused view. Resumes when that view
                closes or goes back to All.
              </div>
            )}
            {/* key on id so switching views reopens the stream cleanly */}
            <LiveStream key={cam.id} cam={cam} pageVisible={pageVisible} />
            {/* Per-user activity sessions & totals, with the admin's "Track:" selection */}
            {cam.view_mode === 'activity' && cam.activity_available && (
              <ActivityTable cameraId={cam.id} onAuthError={onAuthError} activityFilter={cam.activity_filter} />
            )}
          </section>
        ))}
      </div>
    </div>
  )
}
