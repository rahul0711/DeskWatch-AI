import { useEffect, useState } from 'react'
import { AuthError, fetchActivity, fetchActivitySettings, hasRole, saveActivitySettings } from './api'

const POLL_INTERVAL_MS = 5000

function formatDuration(totalSeconds) {
  const total = Math.max(0, Math.round(totalSeconds || 0))
  const h = Math.floor(total / 3600)
  const m = Math.floor((total % 3600) / 60)
  const s = total % 60
  if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
  return `${m}:${String(s).padStart(2, '0')}`
}

// Activity labels are inferences with a confidence, never assertions about
// what someone was doing -- "monitor interaction", not "working".
const ACTIVITY_LABELS = {
  SITTING: 'Sitting',
  STANDING: 'Standing',
  WALKING: 'Walking',
  STATIONARY: 'Low movement',
  PHONE_INTERACTION: 'Phone',
  MONITOR_INTERACTION: 'Monitor',
  COMPUTER_INTERACTION: 'Computer',
  ABSENT: 'Away',
  UNKNOWN: '--',
}

// Duration columns, in table order. `key` is the activity type the admin
// selects; `field` is the per-user total in the API row.
const COLUMNS = [
  { key: 'SITTING', field: 'sitting_seconds', label: 'Sit', title: 'Total time seated' },
  { key: 'STANDING', field: 'standing_seconds', label: 'Stand', title: 'Total time standing' },
  { key: 'WALKING', field: 'walking_seconds', label: 'Walk', title: 'Total time moving around' },
  { key: 'STATIONARY', field: 'low_movement_seconds', label: 'Still', title: 'Total time with movement below the stationary threshold' },
  { key: 'PHONE_INTERACTION', field: 'phone_seconds', label: 'Phone', title: 'Time a phone was detected in use (not merely present), summed over every session' },
  { key: 'MONITOR_INTERACTION', field: 'monitor_seconds', label: 'Monitor', title: 'Time facing a detected monitor -- interaction, not confirmed attention' },
  { key: 'COMPUTER_INTERACTION', field: 'computer_seconds', label: 'PC', title: 'Time with hands at a laptop/keyboard/mouse' },
]

const ACTIVITY_COLORS = {
  SITTING: '#ffa500',
  STANDING: '#39d353',
  WALKING: '#58c4ff',
  STATIONARY: '#9aa0a6',
  PHONE_INTERACTION: '#ff6b4a',
  MONITOR_INTERACTION: '#3cc8c8',
  COMPUTER_INTERACTION: '#ffc857',
  UNKNOWN: '#9aa0a6',
}

const HEAD_LABELS = {
  FORWARD: 'Forward',
  LEFT: 'Left',
  RIGHT: 'Right',
  DOWN: 'Down',
  UP: 'Up',
  TOWARD_MONITOR: 'At monitor',
  UNKNOWN: '--',
}

// Only a Super Admin may change which activities are tracked (applies to every camera).
function ActivitySelector({ enabled, onChange, saving, error, readOnly }) {
  return (
    <div className="activity__selector" title="Which activities are analysed and stored (applies to every camera)">
      <span className="activity__dim">Track:</span>
      {COLUMNS.map((c) => (
        <label key={c.key} className="activity__choice">
          <input
            type="checkbox"
            checked={enabled.includes(c.key)}
            disabled={saving || readOnly}
            onChange={(e) =>
              onChange(e.target.checked ? [...enabled, c.key] : enabled.filter((k) => k !== c.key))
            }
          />
          {ACTIVITY_LABELS[c.key]}
        </label>
      ))}
      {error && <span className="activity__error">{error}</span>}
    </div>
  )
}

// The overlay's "Show" filter also decides what this table shows.
// Focus filters rank people by one activity; body/all/debug show columns.
const FOCUS = {
  phone: { key: 'PHONE_INTERACTION', field: 'phone_seconds', flag: 'phone', label: 'Phone', verb: 'phone use' },
  monitor: { key: 'MONITOR_INTERACTION', field: 'monitor_seconds', flag: 'monitor', label: 'Monitor', verb: 'monitor time' },
  computer: { key: 'COMPUTER_INTERACTION', field: 'computer_seconds', flag: 'computer', label: 'Computer', verb: 'computer use' },
}
const BODY_KEYS = ['SITTING', 'STANDING', 'WALKING', 'STATIONARY']

function personLabel(row) {
  return row.person_name && row.person_name !== 'Unknown' ? row.person_name : `#${row.track_id}`
}

function PersonCell({ row, top }) {
  return (
    <td title={row.track_ids?.length ? `tracker ids: ${row.track_ids.join(', ')}` : undefined}>
      {top && <span className="activity__top" title="Highest total for this activity">★ </span>}
      {row.person_name && row.person_name !== 'Unknown' ? (
        <strong>{row.person_name}</strong>
      ) : (
        <span className="activity__anon">#{row.track_id}</span>
      )}
    </td>
  )
}

// Length of the open session, from the server's session record.
function openSessionSeconds(sess) {
  if (!sess?.session_start) return 0
  const end = sess.current_state === 'GRACE' && sess.last_confirmed ? Date.parse(sess.last_confirmed) : Date.now()
  return Math.max(0, (end - Date.parse(sess.session_start)) / 1000)
}

function FocusTable({ rows, focus }) {
  // Ranked by total time for this activity -- the "best" answer to
  // "who used the phone most" is the first row.
  const ranked = [...rows].sort((a, b) => (b[focus.field] || 0) - (a[focus.field] || 0))
  const leader = ranked.find((r) => (r[focus.field] || 0) > 0)
  const activeNow = ranked.filter((r) => r[focus.flag]).length

  return (
    <>
      <div className="activity__summary">
        {leader ? (
          <>
            Most {focus.verb}: <strong>{personLabel(leader)}</strong> — {formatDuration(leader[focus.field])}
            {' · '}
          </>
        ) : (
          <>No {focus.verb} recorded yet · </>
        )}
        {activeNow} in use now
      </div>
      <div className="activity__scroll">
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Person</th>
              <th title="Is a session open right now, and for how long">Now</th>
              <th title={`Total ${focus.verb}, summed over every session (never reset)`}>Total</th>
              <th title="Separate sessions so far (a gap longer than the grace period starts a new one)">Sessions</th>
              {focus.flag === 'monitor' && <th title="Head/face orientation from pose keypoints">Head</th>}
              <th title="Time since first seen in frame (this visit)">Present</th>
            </tr>
          </thead>
          <tbody>
            {ranked.map((row, i) => {
              const sess = row.sessions?.[focus.key]
              const on = row[focus.flag]
              const total = row[focus.field] || 0
              const count = (sess?.sessions_closed || 0) + (on ? 1 : 0)
              return (
                <tr
                  key={row.subject_id || row.track_id}
                  className={[
                    !row.present ? 'activity__away' : '',
                    row === leader ? 'activity__leader' : '',
                    total === 0 && !on ? 'activity__zero' : '',
                  ].join(' ')}
                >
                  <td className="activity__dim">{i + 1}</td>
                  <PersonCell row={row} top={row === leader} />
                  <td>
                    {on ? (
                      <span className="activity__badge" style={{ color: ACTIVITY_COLORS[focus.key] }}>
                        ● In use {formatDuration(openSessionSeconds(sess))}
                      </span>
                    ) : (
                      <span className="activity__dim">{row.present ? '—' : 'Away'}</span>
                    )}
                  </td>
                  <td><strong>{formatDuration(total)}</strong></td>
                  <td>{count}</td>
                  {focus.flag === 'monitor' && (
                    <td>{HEAD_LABELS[row.head_orientation] || row.head_orientation || '--'}</td>
                  )}
                  <td className="activity__dim">{formatDuration(row.present_seconds)}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </>
  )
}

export default function ActivityTable({ cameraId, onAuthError, activityFilter = 'all' }) {
  const [rows, setRows] = useState([])
  const [enabled, setEnabled] = useState(COLUMNS.map((c) => c.key))
  const [saving, setSaving] = useState(false)
  const [settingsError, setSettingsError] = useState(null)

  useEffect(() => {
    fetchActivitySettings()
      .then((s) => setEnabled(s.enabled_activities))
      .catch((err) => {
        if (err instanceof AuthError) onAuthError()
      })
  }, [onAuthError])

  async function changeEnabled(next) {
    setSaving(true)
    setSettingsError(null)
    try {
      const saved = await saveActivitySettings(next)
      setEnabled(saved.enabled_activities)
    } catch (err) {
      if (err instanceof AuthError) onAuthError()
      else setSettingsError(err.message)
    } finally {
      setSaving(false)
    }
  }

  const filter = (activityFilter || 'all').toLowerCase()
  const focus = FOCUS[filter]
  const isBody = filter === 'body'
  const isDebug = filter === 'debug'
  const columns = COLUMNS.filter(
    (c) => enabled.includes(c.key) && (!isBody || BODY_KEYS.includes(c.key)),
  )

  useEffect(() => {
    let cancelled = false

    let inFlight = false

    async function poll() {
      // One request at a time, none while the tab is hidden (see LiveCameras).
      if (inFlight || document.hidden) return
      inFlight = true
      try {
        const data = await fetchActivity(cameraId)
        if (!cancelled) {
          // People in frame first (longest-present first), then users who
          // have stepped away -- their totals are kept, never reset.
          setRows([...data].sort((a, b) =>
            (b.present ? 1 : 0) - (a.present ? 1 : 0) ||
            (b.present_seconds || 0) - (a.present_seconds || 0)))
        }
      } catch (err) {
        if (cancelled) return
        if (err instanceof AuthError) onAuthError()
      } finally {
        inFlight = false
      }
    }

    poll()
    const timer = setInterval(poll, POLL_INTERVAL_MS)
    return () => {
      cancelled = true
      clearInterval(timer)
    }
  }, [cameraId, onAuthError])

  return (
    <div className="activity">
      <div className="activity__title">Activity &amp; durations (per person)</div>
      <ActivitySelector
        enabled={enabled}
        onChange={changeEnabled}
        saving={saving}
        error={settingsError}
        readOnly={!hasRole('SUPER_ADMIN')}
      />
      {focus && !enabled.includes(focus.key) ? (
        <p className="activity__empty">
          {focus.label} tracking is turned off -- tick &quot;{ACTIVITY_LABELS[focus.key]}&quot; under Track: to record it.
        </p>
      ) : rows.length === 0 ? (
        <p className="activity__empty">No one tracked yet -- appears when a person is in frame.</p>
      ) : focus ? (
        <FocusTable rows={rows} focus={focus} />
      ) : (
        <div className="activity__scroll">
          <table>
            <thead>
              <tr>
                <th>Person</th>
                <th title={isBody ? 'Current posture/motion and how long it has held' : 'Current activity and how long it has held'}>Now</th>
                {!isBody && <th title="Head/face orientation from pose keypoints">Head</th>}
                {isBody && <th title="Movement score: fraction of body size moved per second">Move</th>}
                {columns.map((c) => (
                  <th key={c.key} title={c.title}>{c.label}</th>
                ))}
                <th title="Time since first seen in frame (this visit)">Present</th>
                {isDebug && <th>Tracker ids</th>}
                {isDebug && <th>Identity</th>}
                {isDebug && <th title="Session state per activity (P=pending, A=active, G=grace)">Sessions</th>}
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => {
                const activity = (isBody ? row.primary_activity : row.activity) || 'UNKNOWN'
                const since = isBody ? row.primary_seconds : row.activity_seconds
                return (
                  <tr key={row.subject_id || row.track_id} className={row.present ? '' : 'activity__away'}>
                    <PersonCell row={row} />
                    <td>
                      <span
                        className="activity__badge"
                        style={{ color: ACTIVITY_COLORS[activity] || '#9aa0a6' }}
                      >
                        {ACTIVITY_LABELS[activity] || activity}
                      </span>
                      {' '}
                      <span className="activity__dim">{formatDuration(since)}</span>
                    </td>
                    {!isBody && <td>{HEAD_LABELS[row.head_orientation] || row.head_orientation || '--'}</td>}
                    {isBody && <td className="activity__dim">{(row.movement_score || 0).toFixed(2)}</td>}
                    {columns.map((c) => (
                      <td key={c.key}>{formatDuration(row[c.field])}</td>
                    ))}
                    <td className="activity__dim">{formatDuration(row.present_seconds)}</td>
                    {isDebug && <td className="activity__dim">{(row.track_ids || []).join(', ') || '--'}</td>}
                    {isDebug && (
                      <td className="activity__dim">
                        {row.user_id ? `${row.user_id} (${row.identity_source || '?'}, ${(row.identity_confidence || 0).toFixed(2)})` : 'unidentified'}
                      </td>
                    )}
                    {isDebug && (
                      <td className="activity__dim">
                        {Object.entries(row.sessions || {})
                          .filter(([, v]) => v.current_state !== 'IDLE')
                          .map(([k, v]) => `${ACTIVITY_LABELS[k] || k}:${v.current_state[0]}`)
                          .join(' ') || '--'}
                      </td>
                    )}
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
