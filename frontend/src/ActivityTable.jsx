import { useEffect, useState } from 'react'
import { AuthError, fetchActivity } from './api'

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
  UNKNOWN: '--',
}

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

export default function ActivityTable({ cameraId, onAuthError }) {
  const [rows, setRows] = useState([])

  useEffect(() => {
    let cancelled = false

    async function poll() {
      try {
        const data = await fetchActivity(cameraId)
        if (!cancelled) {
          // Longest-present first: the people actually at desks matter more
          // than whoever just walked into frame.
          setRows([...data].sort((a, b) => (b.present_seconds || 0) - (a.present_seconds || 0)))
        }
      } catch (err) {
        if (cancelled) return
        if (err instanceof AuthError) onAuthError()
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
      <div className="activity__title">Activity &amp; durations</div>
      {rows.length === 0 ? (
        <p className="activity__empty">No one tracked yet -- appears when a person is in frame.</p>
      ) : (
        <div className="activity__scroll">
          <table>
            <thead>
              <tr>
                <th>Person</th>
                <th title="Current activity and how long it has held">Now</th>
                <th title="Head/face orientation from pose keypoints">Head</th>
                <th title="Total time seated">Sit</th>
                <th title="Total time standing">Stand</th>
                <th title="Total time moving around">Walk</th>
                <th title="Total time with movement below the stationary threshold">Still</th>
                <th title="Time a phone was detected in use (not merely present)">Phone</th>
                <th title="Time facing a detected monitor -- interaction, not confirmed attention">Monitor</th>
                <th title="Time with hands at a laptop/keyboard/mouse">PC</th>
                <th title="Time since first seen in frame">Present</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => {
                const activity = row.activity || 'UNKNOWN'
                return (
                  <tr key={row.track_id}>
                    <td>
                      {row.person_name && row.person_name !== 'Unknown' ? (
                        <strong>{row.person_name}</strong>
                      ) : (
                        <span className="activity__anon">#{row.track_id}</span>
                      )}
                    </td>
                    <td>
                      <span
                        className="activity__badge"
                        style={{ color: ACTIVITY_COLORS[activity] || '#9aa0a6' }}
                      >
                        {ACTIVITY_LABELS[activity] || activity}
                      </span>
                      {' '}
                      <span className="activity__dim">{formatDuration(row.activity_seconds)}</span>
                    </td>
                    <td>{HEAD_LABELS[row.head_orientation] || row.head_orientation || '--'}</td>
                    <td>{formatDuration(row.sitting_seconds)}</td>
                    <td>{formatDuration(row.standing_seconds)}</td>
                    <td>{formatDuration(row.walking_seconds)}</td>
                    <td>{formatDuration(row.low_movement_seconds)}</td>
                    <td>{formatDuration(row.phone_seconds)}</td>
                    <td>{formatDuration(row.monitor_seconds)}</td>
                    <td>{formatDuration(row.computer_seconds)}</td>
                    <td className="activity__dim">{formatDuration(row.present_seconds)}</td>
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
