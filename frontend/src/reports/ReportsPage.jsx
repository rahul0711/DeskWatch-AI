import { useEffect, useState } from 'react'
import { AuthError, getUser } from '../api'
import { fetchActivityDaily, fetchActivitySessions } from '../db/api'
import { duration, timeOfDay, todayIst } from './format'
import '../admin/Admin.css'

const COLUMNS = [
  ['present_seconds', 'Present'],
  ['sitting_seconds', 'Sitting'],
  ['standing_seconds', 'Standing'],
  ['walking_seconds', 'Walking'],
  ['low_movement_seconds', 'Low movement'],
  ['phone_seconds', 'Phone'],
  ['monitor_seconds', 'Monitor'],
  ['computer_seconds', 'Computer'],
]

// Stored activity time per person for one day, from MySQL via the .NET backend.
// Only the logged-in login's cameras are included.
export default function ReportsPage({ onAuthError }) {
  const cameras = getUser()?.cameras || []
  const [date, setDate] = useState(todayIst)
  const [cameraId, setCameraId] = useState('')
  const [rows, setRows] = useState(null)
  const [error, setError] = useState(null)
  const [person, setPerson] = useState(null) // row whose sessions are shown
  const [sessions, setSessions] = useState(null)

  useEffect(() => {
    let cancelled = false
    setRows(null)
    setPerson(null)
    fetchActivityDaily(date, cameraId)
      .then((r) => {
        if (cancelled) return
        setRows(r)
        setError(null)
      })
      .catch((err) => (err instanceof AuthError ? onAuthError() : setError(err.message)))
    return () => {
      cancelled = true
    }
  }, [date, cameraId, onAuthError])

  useEffect(() => {
    if (!person) return undefined
    let cancelled = false
    setSessions(null)
    fetchActivitySessions({ date, cameraId, employeeId: person.employee_id, pageSize: 200 })
      .then((d) => !cancelled && setSessions(d))
      .catch((err) => (err instanceof AuthError ? onAuthError() : setError(err.message)))
    return () => {
      cancelled = true
    }
  }, [person, date, cameraId, onAuthError])

  return (
    <div className="admin">
      <section className="admin__panel">
        <h2>Activity report</h2>
        <div className="admin__filters">
          <label>
            Day (India time)
            <input type="date" value={date} max={todayIst()} onChange={(e) => e.target.value && setDate(e.target.value)} />
          </label>
          <label>
            Camera
            <select value={cameraId} onChange={(e) => setCameraId(e.target.value)}>
              <option value="">All my cameras</option>
              {cameras.map((c) => (
                <option key={c.camera_id} value={c.camera_id}>
                  Camera {c.camera_no} · {c.name}
                </option>
              ))}
            </select>
          </label>
          <span className="admin__hint">Activities overlap (sitting + monitor at once), so columns can add up to more than Present.</span>
        </div>
        {error && <p className="admin__error">{error}</p>}
        {rows === null && !error && <p className="admin__hint">Loading…</p>}
        {rows?.length === 0 && <p className="admin__hint">No activity stored for this day.</p>}
        {rows?.length > 0 && (
          <div className="admin__table-wrap">
            <table className="admin__table admin__table--numbers">
              <thead>
                <tr>
                  <th>Person</th>
                  {COLUMNS.map(([key, label]) => (
                    <th key={key}>{label}</th>
                  ))}
                  <th>First seen</th>
                  <th>Last seen</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr
                    key={r.employee_id ?? 'unknown'}
                    className={
                      r.employee_id
                        ? `admin__row--click${person?.employee_id === r.employee_id ? ' admin__row--selected' : ''}`
                        : ''
                    }
                    onClick={() => r.employee_id && setPerson(r)}
                    title={r.employee_id ? "Show this person's sessions" : undefined}
                  >
                    <td>
                      <strong>{r.person_name}</strong>
                      <div className="admin__sub">{r.employee_id ?? 'face not recognised'}</div>
                    </td>
                    {COLUMNS.map(([key]) => (
                      <td key={key}>
                        {duration(r[key])}
                        {key === 'phone_seconds' && r.phone_sessions > 0 && <div className="admin__sub">{r.phone_sessions}×</div>}
                      </td>
                    ))}
                    <td>{timeOfDay(r.first_seen_ist)}</td>
                    <td>{timeOfDay(r.last_seen_ist)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {person && (
        <section className="admin__panel">
          <h2>
            Sessions · {person.person_name}
            <button className="admin__close" onClick={() => setPerson(null)}>
              ×
            </button>
          </h2>
          {sessions === null && <p className="admin__hint">Loading…</p>}
          {sessions && (
            <>
              <p className="admin__hint">
                {sessions.total} session{sessions.total === 1 ? '' : 's'}
                {sessions.total > sessions.items.length ? ` (newest ${sessions.items.length} shown)` : ''}
              </p>
              <div className="admin__table-wrap">
                <table className="admin__table">
                  <thead>
                    <tr>
                      <th>Activity</th>
                      <th>Camera</th>
                      <th>Start</th>
                      <th>End</th>
                      <th>Duration</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sessions.items.map((s) => (
                      <tr key={s.id}>
                        <td>{s.activity_name || s.activity_type}</td>
                        <td className="admin__sub">{s.camera_name || s.camera_id}</td>
                        <td>{new Date(s.start_time).toLocaleTimeString()}</td>
                        <td>{new Date(s.end_time).toLocaleTimeString()}</td>
                        <td>{duration(s.duration_seconds)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </section>
      )}
    </div>
  )
}
