import { useEffect, useState } from 'react'
import { AuthError, getUser } from '../api'
import { fetchActivitySessions } from '../db/api'
import { duration, todayIst } from './format'
import '../admin/Admin.css'

// Every confirmed phone use is a violation. They are the PHONE_INTERACTION
// sessions the cameras stored (activity_sessions), newest first.
export default function ViolationsPage({ onAuthError }) {
  const cameras = getUser()?.cameras || []
  const [date, setDate] = useState(todayIst)
  const [cameraId, setCameraId] = useState('')
  const [person, setPerson] = useState('')
  const [items, setItems] = useState(null)
  const [total, setTotal] = useState(0)
  const [error, setError] = useState(null)

  useEffect(() => {
    let cancelled = false
    setItems(null)
    fetchActivitySessions({ date, cameraId, activityType: 'PHONE_INTERACTION', pageSize: 500 })
      .then((d) => {
        if (cancelled) return
        setItems(d.items)
        setTotal(d.total)
        setError(null)
      })
      .catch((err) => (err instanceof AuthError ? onAuthError() : setError(err.message)))
    return () => {
      cancelled = true
    }
  }, [date, cameraId, onAuthError])

  const people = [...new Set((items || []).map((v) => v.person_name))].sort()
  const shown = (items || []).filter((v) => !person || v.person_name === person)
  const phoneTime = shown.reduce((s, v) => s + (v.duration_seconds || 0), 0)

  return (
    <div className="admin">
      <section className="admin__panel">
        <h2>Violations</h2>
        <p className="admin__hint">
          Every confirmed phone use is a violation: a phone seen in the person&apos;s hand and moving with it, whose
          phone-use confidence reached at least 50%. Recorded by cameras with body movement on (not while another
          view, such as Monitor, is focused on that camera).
        </p>
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
          <label>
            Person
            <select value={person} onChange={(e) => setPerson(e.target.value)}>
              <option value="">Everyone</option>
              {people.map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
          </label>
        </div>

        {items && (
          <div className="attendance__stats">
            <div className="attendance__stat attendance__stat--bad">
              Violations
              <strong>{shown.length}</strong>
            </div>
            <div className="attendance__stat">
              People
              <strong>{person ? 1 : people.length}</strong>
            </div>
            <div className="attendance__stat">
              Phone time
              <strong>{duration(phoneTime)}</strong>
            </div>
          </div>
        )}

        {error && <p className="admin__error">{error}</p>}
        {items === null && !error && <p className="admin__hint">Loading…</p>}
        {items && shown.length === 0 && <p className="admin__hint">No violations for this day.</p>}
        {total > (items?.length || 0) && (
          <p className="admin__hint">Showing the newest {items.length} of {total}.</p>
        )}
        {shown.length > 0 && (
          <div className="admin__table-wrap">
            <table className="admin__table">
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Person</th>
                  <th>Camera</th>
                  <th>Duration</th>
                  <th title="Share of the checks during this phone use that saw the phone">Seen in checks</th>
                </tr>
              </thead>
              <tbody>
                {shown.map((v) => (
                  <tr key={v.id}>
                    <td>
                      {new Date(v.start_time).toLocaleTimeString()}
                      <div className="admin__sub">to {new Date(v.end_time).toLocaleTimeString()}</div>
                    </td>
                    <td>
                      <strong>{v.person_name}</strong>
                      <div className="admin__sub">{v.employee_id ?? 'face not recognised'}</div>
                    </td>
                    <td className="admin__sub">{v.camera_name || v.camera_id}</td>
                    <td>{duration(v.duration_seconds)}</td>
                    <td>{Math.round((v.confidence || 0) * 100)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  )
}
