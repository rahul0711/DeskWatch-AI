import { useEffect, useState } from 'react'
import { AuthError, fetchHeadCounts } from './api'

const POLL_INTERVAL_MS = 10000

export default function HeadCountTable({ cameraId, headCountSource, onAuthError }) {
  const [rows, setRows] = useState([])

  useEffect(() => {
    let cancelled = false

    async function poll() {
      try {
        const data = await fetchHeadCounts(cameraId)
        if (!cancelled) {
          setRows([...data].reverse()) // most recent minute on top
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

  const showFaces = headCountSource === 'face'
  const showPeople = headCountSource === 'person' || headCountSource === 'head'
  const showAttendance = headCountSource === 'attendance'

  return (
    <div className="head-count">
      <div className="head-count__title">Head count per minute</div>
      {rows.length === 0 ? (
        <p className="head-count__empty">No data yet -- collecting…</p>
      ) : (
        <div className="head-count__scroll">
          <table>
            <thead>
              <tr>
                <th>Minute</th>
                {showFaces && <th title="Avg faces detected (YOLOv11)">Faces</th>}
                {showPeople && <th title="Avg people/bodies detected">People</th>}
                {showAttendance && <th title="Avg faces detected (InsightFace Attendance)">Faces (Att)</th>}
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.minute}>
                  <td>{row.minute.slice(11)}</td>
                  {showFaces && <td>{row.avg_faces ?? 0}</td>}
                  {showPeople && <td>{row.avg_people ?? 0}</td>}
                  {showAttendance && <td>{row.avg_faces ?? 0}</td>}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
