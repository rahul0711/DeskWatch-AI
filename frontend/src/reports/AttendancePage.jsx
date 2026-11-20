import { useEffect, useMemo, useState } from 'react'
import { AuthError } from '../api'
import { fetchAttendanceDaily } from '../db/api'
import { timeOfDay, todayIst } from './format'
import '../admin/Admin.css'

const FILTERS = [
  ['all', 'All'],
  ['present', 'Present'],
  ['absent', 'Absent'],
]

function downloadCsv(date, rows) {
  const cell = (v) => `"${String(v ?? '').replace(/"/g, '""')}"`
  const lines = [
    ['Employee ID', 'Name', 'Status', 'First in', 'Last seen', 'Punches', 'Cameras'],
    ...rows.map((r) => [
      r.employee_id,
      r.name,
      r.present ? 'Present' : 'Absent',
      r.present ? timeOfDay(r.first_in) : '',
      r.present ? timeOfDay(r.last_seen) : '',
      r.punches,
      r.cameras.join('; '),
    ]),
  ]
  const blob = new Blob([lines.map((l) => l.map(cell).join(',')).join('\n')], { type: 'text/csv' })
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = `attendance-${date}.csv`
  a.click()
  URL.revokeObjectURL(a.href)
}

// Who came and who did not, for one India day: every enrolled person with their
// first/last punch. Admin and Super Admin only.
export default function AttendancePage({ onAuthError }) {
  const [date, setDate] = useState(todayIst)
  const [rows, setRows] = useState(null)
  const [error, setError] = useState(null)
  const [filter, setFilter] = useState('all')
  const [search, setSearch] = useState('')
  const [refreshKey, setRefreshKey] = useState(0)

  useEffect(() => {
    let cancelled = false
    fetchAttendanceDaily(date)
      .then((r) => {
        if (cancelled) return
        setRows(r)
        setError(null)
      })
      .catch((err) => (err instanceof AuthError ? onAuthError() : setError(err.message)))
    return () => {
      cancelled = true
    }
  }, [date, refreshKey, onAuthError])

  // Today's list fills in as people arrive -- refresh it every minute.
  useEffect(() => {
    if (date !== todayIst()) return undefined
    const timer = setInterval(() => setRefreshKey((k) => k + 1), 60000)
    return () => clearInterval(timer)
  }, [date])

  const present = rows?.filter((r) => r.present).length ?? 0
  const total = rows?.length ?? 0

  const shown = useMemo(() => {
    const q = search.trim().toLowerCase()
    return (rows || [])
      .filter((r) => filter === 'all' || (filter === 'present') === r.present)
      .filter((r) => !q || r.name.toLowerCase().includes(q) || r.employee_id.toLowerCase().includes(q))
      // Present first (earliest arrival first), then absent by name.
      .sort((a, b) => (a.present === b.present ? (a.first_in || a.name).localeCompare(b.first_in || b.name) : a.present ? -1 : 1))
  }, [rows, filter, search])

  return (
    <div className="admin">
      <section className="admin__panel">
        <h2>Attendance</h2>
        <div className="admin__filters">
          <label>
            Day (India time)
            <input
              type="date"
              value={date}
              max={todayIst()}
              onChange={(e) => {
                if (!e.target.value) return
                setRows(null)
                setDate(e.target.value)
              }}
            />
          </label>
          <label>
            Search
            <input type="search" placeholder="Name or employee ID" value={search} onChange={(e) => setSearch(e.target.value)} />
          </label>
          <div className="admin__sections">
            {FILTERS.map(([key, label]) => (
              <button key={key} className={filter === key ? 'is-active' : ''} onClick={() => setFilter(key)}>
                {label}
              </button>
            ))}
            <button onClick={() => downloadCsv(date, shown)} disabled={!shown.length}>
              Export CSV
            </button>
          </div>
        </div>

        {rows && (
          <div className="attendance__stats">
            <div className="attendance__stat">
              Enrolled
              <strong>{total}</strong>
            </div>
            <div className="attendance__stat attendance__stat--ok">
              Present
              <strong>{present}</strong>
            </div>
            <div className="attendance__stat attendance__stat--bad">
              Absent
              <strong>{total - present}</strong>
            </div>
            <div className="attendance__stat">
              Attendance
              <strong>{total ? Math.round((present / total) * 100) : 0}%</strong>
            </div>
          </div>
        )}

        {error && <p className="admin__error">{error}</p>}
        {rows === null && !error && <p className="admin__hint">Loading…</p>}
        {rows?.length === 0 && <p className="admin__hint">Nobody is enrolled yet. Add people on the Register tab.</p>}
        {rows?.length > 0 && shown.length === 0 && <p className="admin__hint">No one matches.</p>}
        {shown.length > 0 && (
          <div className="admin__table-wrap">
            <table className="admin__table">
              <thead>
                <tr>
                  <th>Person</th>
                  <th>Status</th>
                  <th>First in</th>
                  <th>Last seen</th>
                  <th>Punches</th>
                  <th>Cameras</th>
                </tr>
              </thead>
              <tbody>
                {shown.map((r) => (
                  <tr key={r.user_id}>
                    <td>
                      <div className="attendance__person">
                        {r.photo_url ? (
                          <img className="attendance__photo" src={r.photo_url} alt="" />
                        ) : (
                          <span className="attendance__photo" />
                        )}
                        <div>
                          <strong>{r.name}</strong>
                          <div className="admin__sub">{r.employee_id}</div>
                        </div>
                      </div>
                    </td>
                    <td>
                      <span className={`attendance__status attendance__status--${r.present ? 'present' : 'absent'}`}>
                        {r.present ? 'Present' : 'Absent'}
                      </span>
                    </td>
                    <td>{timeOfDay(r.first_in)}</td>
                    <td>{timeOfDay(r.last_seen)}</td>
                    <td>{r.punches || '—'}</td>
                    <td className="admin__sub">{r.cameras.join(', ') || '—'}</td>
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
