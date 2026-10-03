import { useCallback, useEffect, useState } from 'react'
import { AuthError, fetchCameras, setViewMode, streamUrl } from './api'
import './LiveCameras.css'

const STATE_COLORS = {
  connected: 'var(--ok)',
  connecting: 'var(--warn)',
  reconnecting: 'var(--warn)',
  stopped: 'var(--bad)',
}

const POLL_MS = 3000

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
export default function LiveCameras({ onAuthError }) {
  const [cameras, setCameras] = useState([])
  const [error, setError] = useState(null)
  const [selected, setSelected] = useState('all') // 'all' | camera id
  const [busy, setBusy] = useState(false)

  const load = useCallback(async () => {
    try {
      const list = await fetchCameras()
      setCameras(list)
      setError(null)
    } catch (err) {
      if (err instanceof AuthError) onAuthError()
      else setError('Camera server not reachable -- is app/web/server.py running?')
    }
  }, [onAuthError])

  useEffect(() => {
    load()
    const timer = setInterval(load, POLL_MS)
    return () => clearInterval(timer)
  }, [load])

  // Applies to every visible camera that supports it, so the choice doesn't
  // have to be repeated per tile in the side-by-side view.
  const applyView = useCallback(
    async (targets, patch) => {
      setBusy(true)
      try {
        await Promise.all(
          targets
            .filter((c) => c.activity_available || patch.viewMode === 'face')
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
          disabled={busy}
          onClick={() => applyView(visible, { viewMode: 'face' })}
          title="Face detection and recognition boxes"
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
              : 'Activity analysis is not enabled for this camera (see ACTIVITY_CAMERAS in .env)'
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
            Filters the overlay only — every activity is still tracked and recorded.
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
            {/* key on id so switching views reopens the stream cleanly */}
            <img key={cam.id} className="live__video" src={streamUrl(cam.id)} alt={`${cam.name} live`} />
          </section>
        ))}
      </div>
    </div>
  )
}
