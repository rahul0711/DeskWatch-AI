import { useEffect, useState } from 'react'
import { AuthError, getUser } from '../api'
import { fetchActivityDaily, fetchActivitySessions } from '../db/api'
import { duration, timeOfDay, todayIst } from './format'
import '../admin/Admin.css'

// One page per body-movement activity, from what the cameras recorded
// (MySQL activity_sessions via the .NET backend). Only the logged-in login's
// cameras are included.
export const ACTIVITY_PAGES = {
  phone: {
    title: 'Phone usage',
    intro: 'Who had a phone in their hand, for how long, and each time they picked it up.',
    columns: [['phone_seconds', 'Phone time']],
    extra: ['phone_sessions', 'Times'],
    types: ['PHONE_INTERACTION'],
  },
  posture: {
    title: 'Sitting & standing',
    intro: 'How long each person sat, stood and walked, and when each posture started and ended.',
    columns: [
      ['sitting_seconds', 'Sitting'],
      ['standing_seconds', 'Standing'],
      ['walking_seconds', 'Walking'],
    ],
    types: ['SITTING', 'STANDING', 'WALKING'],
  },
  monitor: {
    title: 'Monitor use',
    intro: 'How long each person faced a monitor, and each session.',
    columns: [['monitor_seconds', 'Monitor time']],
    types: ['MONITOR_INTERACTION'],
  },
  computer: {
    title: 'Computer use',
    intro: 'How long each person had their hands on a keyboard, mouse or laptop, and each session.',
    columns: [['computer_seconds', 'Computer time']],
    types: ['COMPUTER_INTERACTION'],
  },
}

export default function ActivityPage({ page, onAuthError }) {
  const cfg = ACTIVITY_PAGES[page]
  const cameras = getUser()?.cameras || []
  const [date, setDate] = useState(todayIst)
  const [cameraId, setCameraId] = useState('')
  const [rows, setRows] = useState(null)
  const [error, setError] = useState(null)
  const [person, setPerson] = useState(null)
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
  }, [date, cameraId, page, onAuthError])

  useEffect(() => {
    if (!person) return undefined
    let cancelled = false
    setSessions(null)
    const params = { date, cameraId, employeeId: person.employee_id, pageSize: 500 }
    if (cfg.types.length === 1) params.activityType = cfg.types[0]
    fetchActivitySessions(params)
      .then((d) => {
        if (cancelled) return
        setSessions(d.items.filter((s) => cfg.types.includes(s.activity_type)))
      })
      .catch((err) => (err instanceof AuthError ? onAuthError() : setError(err.message)))
    return () => {
      cancelled = true
    }
  }, [person, date, cameraId, cfg, onAuthError])

  const total = (r) => cfg.columns.reduce((sum, [key]) => sum + (r[key] || 0), 0)
  const shown = (rows || []).filter((r) => total(r) > 0).sort((a, b) => total(b) - total(a))
  const named = shown.filter((r) => r.employee_id)
  const top = named[0]
  const sumAll = shown.reduce((s, r) => s + total(r), 0)

  return (
    <div className="admin">
      <section className="admin__panel">
        <h2>{cfg.title}</h2>
        <p className="admin__hint">{cfg.intro}</p>
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
        </div>

        {rows && (
          <div className="attendance__stats">
            <div className="attendance__stat">
              People
              <strong>{named.length}</strong>
            </div>
            <div className="attendance__stat">
              Total time
              <strong>{duration(sumAll)}</strong>
            </div>
            <div className="attendance__stat">
              Most
              <strong className="activity__top">{top ? top.person_name : '—'}</strong>
              {top && <span>{duration(total(top))}</span>}
            </div>
          </div>
        )}

        {error && <p className="admin__error">{error}</p>}
        {rows === null && !error && <p className="admin__hint">Loading…</p>}
        {rows && shown.length === 0 && <p className="admin__hint">Nothing recorded for this day.</p>}
        {shown.length > 0 && (
          <div className="admin__table-wrap">
            <table className="admin__table admin__table--numbers">
              <thead>
                <tr>
                  <th>Person</th>
                  {cfg.columns.map(([key, label]) => (
                    <th key={key}>{label}</th>
                  ))}
                  {cfg.extra && <th>{cfg.extra[1]}</th>}
                  <th>First seen</th>
                  <th>Last seen</th>
                </tr>
              </thead>
              <tbody>
                {shown.map((r) => (
                  <tr
                    key={r.employee_id ?? 'unknown'}
                    className={
                      r.employee_id
                        ? `admin__row--click${person?.employee_id === r.employee_id ? ' admin__row--selected' : ''}`
                        : ''
                    }
                    onClick={() => r.employee_id && setPerson(r)}
                    title={r.employee_id ? 'Show each session' : undefined}
                  >
                    <td>
                      <strong>{r.person_name}</strong>
                      <div className="admin__sub">{r.employee_id ?? 'face not recognised -- not registered or not seen clearly'}</div>
                    </td>
                    {cfg.columns.map(([key]) => (
                      <td key={key}>{duration(r[key])}</td>
                    ))}
                    {cfg.extra && <td>{r[cfg.extra[0]] || '—'}</td>}
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
            {cfg.title} · {person.person_name}
            <button className="admin__close" onClick={() => setPerson(null)}>
              ×
            </button>
          </h2>
          {sessions === null && <p className="admin__hint">Loading…</p>}
          {sessions?.length === 0 && <p className="admin__hint">No sessions recorded.</p>}
          {sessions?.length > 0 && (
            <div className="admin__table-wrap">
              <table className="admin__table">
                <thead>
                  <tr>
                    {cfg.types.length > 1 && <th>Activity</th>}
                    <th>Camera</th>
                    <th>Start</th>
                    <th>End</th>
                    <th>Duration</th>
                  </tr>
                </thead>
                <tbody>
                  {sessions.map((s) => (
                    <tr key={s.id}>
                      {cfg.types.length > 1 && <td>{s.activity_name || s.activity_type}</td>}
                      <td className="admin__sub">{s.camera_name || s.camera_id}</td>
                      <td>{new Date(s.start_time).toLocaleTimeString()}</td>
                      <td>{new Date(s.end_time).toLocaleTimeString()}</td>
                      <td>{duration(s.duration_seconds)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      )}
    </div>
  )
}
