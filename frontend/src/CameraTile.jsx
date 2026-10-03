import { useEffect, useState, useCallback } from 'react'
import { AuthError, removeCamera, setHeadCountSource, setShowHud, streamUrl } from './api'
import ActivityTable from './ActivityTable'
import HeadCountTable from './HeadCountTable'
import RecognitionTable from './RecognitionTable'
import OcrTable from './OcrTable'
import ModeScopeDialog from './ModeScopeDialog'

const STATE_COLORS = {
  connected: 'var(--ok)',
  connecting: 'var(--warn)',
  reconnecting: 'var(--warn)',
  stopped: 'var(--bad)',
}

const MODES = [
  { key: 'face',       label: '👤 Face',            title: 'Count by face detection only (YOLOv11)' },
  { key: 'person',     label: '🧍 Body',            title: 'Count by body/head silhouette' },
  { key: 'attendance', label: '📋 Face Attendance', title: 'Face Attendance using InsightFace (SCRFD + ArcFace)' },
  { key: 'activity',   label: '🪑 Activity',        title: 'Track sitting time and phone/computer use per person' },
  { key: 'ocr',        label: '🔤 OCR',             title: 'Scene-text OCR: walls/signs/boards + text carried by moving people' },
]

export default function CameraTile({ status, allCameras = [], onAuthError, onRemoved }) {
  const {
    id,
    name,
    connection_state: state,
    resolution,
    capture_fps: captureFps,
    process_fps: processFps,
    latency_ms: latencyMs,
    faces = 0,
    heads = 0,
    activity_people: activityPeople = 0,
    ocr_regions: ocrRegions = 0,
    ocr_confirmed: ocrConfirmed = 0,
    head_count_source: serverSource,
    show_hud: serverShowHud,
  } = status

  // Which mode(s) are on for this camera right now -- a camera can run
  // several at once (e.g. Activity + OCR together), and each camera keeps
  // its own independent set. Starts from whatever the server reports.
  const [activeModes, setActiveModes] = useState(() => new Set([serverSource || 'face']))
  // The one mode actually PATCHed to the server -- it drives the video
  // overlay and (for face/person/attendance) which count column the
  // head-count-per-minute table shows, since the backend only tracks a
  // single "current" source per camera. The other active modes still
  // render their own independent tables (activity/ocr/recognitions),
  // which don't depend on this value at all.
  const [hudSource, setHudSource] = useState(serverSource || 'face')
  const [showHud, setLocalShowHud] = useState(serverShowHud !== undefined ? serverShowHud : false)
  const [switching, setSwitching] = useState(false)
  const [hudSwitching, setHudSwitching] = useState(false)
  const [removing, setRemoving] = useState(false)
  const [scopeDialog, setScopeDialog] = useState(null) // { key, label } | null

  // Another tile's "apply to multiple cameras" dialog can turn a mode on
  // for this camera without this component ever calling setHeadCountSource
  // itself -- pick that up on the next poll.
  useEffect(() => {
    if (serverSource && serverSource !== hudSource) {
      setHudSource(serverSource)
      setActiveModes((prev) => (prev.has(serverSource) ? prev : new Set(prev).add(serverSource)))
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [serverSource])

  // Turns `newMode` on for every camera id in `cameraIds` (one PATCH each)
  // and makes it the overlay/hud source on each of those cameras.
  const applyModeToCameras = useCallback(async (newMode, cameraIds) => {
    setSwitching(true)
    try {
      await Promise.all(cameraIds.map((camId) => setHeadCountSource(camId, newMode)))
      if (cameraIds.includes(id)) {
        setHudSource(newMode)
        setActiveModes((prev) => new Set(prev).add(newMode))
      }
    } catch (err) {
      if (err instanceof AuthError) onAuthError()
    } finally {
      setSwitching(false)
    }
  }, [id, onAuthError])

  // Turns `key` off for this camera only. Always keeps at least one mode
  // active. If the mode being turned off was driving the overlay, hand
  // that job to whichever mode is still active.
  const removeMode = useCallback(async (key) => {
    if (activeModes.size <= 1 || switching) return
    const remaining = Array.from(activeModes).filter((m) => m !== key)
    setActiveModes(new Set(remaining))
    if (key !== hudSource) return
    const fallback = remaining[0]
    setSwitching(true)
    try {
      await setHeadCountSource(id, fallback)
      setHudSource(fallback)
    } catch (err) {
      if (err instanceof AuthError) onAuthError()
    } finally {
      setSwitching(false)
    }
  }, [activeModes, hudSource, switching, id, onAuthError])

  const handleModeClick = useCallback((key, label) => {
    if (switching) return
    if (activeModes.has(key)) {
      removeMode(key)
      return
    }
    // Face Attendance already drives real recognition + logging for this
    // camera, so turning it on applies immediately -- no camera-scope
    // dialog. The other modes are just live counting/display, so offer to
    // turn them on for this camera only, or for several cameras at once.
    if (key === 'attendance') {
      applyModeToCameras(key, [id])
      return
    }
    setScopeDialog({ key, label })
  }, [switching, activeModes, removeMode, applyModeToCameras, id])

  const handleHudToggle = useCallback(async () => {
    if (hudSwitching) return
    setHudSwitching(true)
    const target = !showHud
    try {
      await setShowHud(id, target)
      setLocalShowHud(target)
    } catch (err) {
      if (err instanceof AuthError) onAuthError()
    } finally {
      setHudSwitching(false)
    }
  }, [id, showHud, hudSwitching, onAuthError])

  const handleRemove = useCallback(async () => {
    if (removing) return
    if (!window.confirm(`Remove ${name}?`)) return
    setRemoving(true)
    try {
      await removeCamera(id)
      onRemoved?.()
    } catch (err) {
      if (err instanceof AuthError) onAuthError()
      setRemoving(false)
    }
  }, [id, name, removing, onAuthError, onRemoved])

  const color = STATE_COLORS[state] || 'var(--text-dim)'

  // One badge per active mode, shown in the stats bar.
  const countBadges = []
  if (activeModes.has('face')) countBadges.push(`${faces} face${faces === 1 ? '' : 's'}`)
  if (activeModes.has('attendance')) countBadges.push(`${faces} face${faces === 1 ? '' : 's'} (Att)`)
  if (activeModes.has('person') || activeModes.has('head')) countBadges.push(`${heads} bod${heads === 1 ? 'y' : 'ies'}`)
  if (activeModes.has('activity')) countBadges.push(`${activityPeople} tracked`)
  if (activeModes.has('ocr')) countBadges.push(`${ocrConfirmed}/${ocrRegions} text`)

  return (
    <div className="camera-tile">
      {/* ---- Header ---- */}
      <div className="camera-tile__header">
        <span className="camera-tile__dot" style={{ background: color }} />
        <span className="camera-tile__name">{name}</span>
        <button
          className={`hud-toggle-btn${showHud ? ' hud-toggle-btn--active' : ''}`}
          disabled={hudSwitching}
          onClick={handleHudToggle}
          title="Toggle overlay stats on camera feed"
        >
          {showHud ? '📺 HUD On' : '📺 HUD Off'}
        </button>
        <span className="camera-tile__state">{state}</span>
        <button
          className="camera-tile__remove"
          disabled={removing}
          onClick={handleRemove}
          title="Remove this camera"
        >
          ✕
        </button>
      </div>

      {/* ---- Mode toggle buttons (check several to run them together) ---- */}
      <div className="camera-tile__mode-bar">
        <span className="camera-tile__mode-label">Count by:</span>
        {MODES.map(({ key, label, title }) => (
          <button
            key={key}
            id={`mode-btn-${id}-${key}`}
            title={title}
            disabled={switching}
            className={`mode-btn${activeModes.has(key) ? ' mode-btn--active' : ''}`}
            onClick={() => handleModeClick(key, label)}
          >
            {label}
          </button>
        ))}
      </div>

      {scopeDialog && (
        <ModeScopeDialog
          modeLabel={scopeDialog.label}
          cameras={allCameras}
          defaultCameraId={id}
          onCancel={() => setScopeDialog(null)}
          onApply={(cameraIds) => {
            applyModeToCameras(scopeDialog.key, cameraIds)
            setScopeDialog(null)
          }}
        />
      )}

      {/* ---- Camera feed (left) + results table (right) ---- */}
      <div className="camera-tile__body">
        <div className="camera-tile__left">
          {/* ---- Live video ---- */}
          <div className="camera-tile__video">
            {/* MJPEG connection stays open — changing src restarts it, so we use
                the stable per-camera URL and let the server render the overlay. */}
            <img src={streamUrl(id)} alt={`${name} live feed`} />
          </div>

          {/* ---- Stats bar ---- */}
          <div className="camera-tile__stats">
            <span>{resolution ? `${resolution[0]}×${resolution[1]}` : '—'}</span>
            <span>{captureFps.toFixed(1)} cap fps</span>
            <span>{processFps.toFixed(1)} proc fps</span>
            <span>{latencyMs.toFixed(0)} ms</span>
            {countBadges.map((badge) => (
              <span key={badge} className="camera-tile__faces">{badge}</span>
            ))}
          </div>
        </div>

        <div className="camera-tile__right">
          {/* ---- Head count table (face/person/attendance share this) ---- */}
          {(activeModes.has('face') || activeModes.has('person') || activeModes.has('head') || activeModes.has('attendance')) && (
            <HeadCountTable
              cameraId={id}
              headCountSource={hudSource}
              onAuthError={onAuthError}
            />
          )}

          {/* ---- Activity (sitting/phone/computer) results ---- */}
          {activeModes.has('activity') && (
            <ActivityTable cameraId={id} onAuthError={onAuthError} />
          )}

          {/* ---- OCR (scene-text) results ---- */}
          {activeModes.has('ocr') && (
            <OcrTable cameraId={id} onAuthError={onAuthError} />
          )}

          {/* ---- Recognize API results ---- */}
          {activeModes.has('attendance') && (
            <RecognitionTable cameraId={id} onAuthError={onAuthError} />
          )}
        </div>
      </div>
    </div>
  )
}
