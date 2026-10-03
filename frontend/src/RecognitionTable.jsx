import { useEffect, useRef, useState } from 'react'
import { AuthError, fetchRecognitions } from './api'

const POLL_INTERVAL_MS = 5000

// Response shape isn't guaranteed by the external /Recognize API (see
// app/web/recognize_client.py), so this reads defensively and falls back
// to an "unknown shape" badge rather than assuming fields exist.
function summarizeResponse(response) {
  if (!response || typeof response !== 'object') {
    return { kind: 'error', label: 'No response' }
  }
  if (response.error) {
    return { kind: 'error', label: response.raw || response.error }
  }
  // The API uses two different conventions seen in the wild: {match: bool,
  // score, threshold, message} for a not-recognised result, and
  // {success: bool, message} for a punch-in/out confirmation -- both are
  // "did this crop get accepted", just spelled differently.
  if (response.match === true || response.success === true) {
    const name = response.name || response.EmployeeName || response.employee_name
    return { kind: 'match', label: response.message || (name ? `Matched: ${name}` : 'Matched') }
  }
  if (response.match === false || response.success === false) {
    return { kind: 'no-match', label: response.message || 'Not recognised' }
  }
  return { kind: 'unknown', label: 'Unrecognised response shape' }
}

export default function RecognitionTable({ cameraId, onAuthError }) {
  const [rows, setRows] = useState([])
  const [expandedIds, setExpandedIds] = useState(() => new Set())
  const seenIdsRef = useRef(new Set())

  const toggleExpanded = (id) => {
    setExpandedIds((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  useEffect(() => {
    let cancelled = false

    async function poll() {
      try {
        const data = await fetchRecognitions(cameraId)
        if (cancelled) return

        // Log every event we haven't already logged, so the console shows
        // each /Recognize response exactly once, as soon as it's polled in.
        for (const row of data) {
          if (!seenIdsRef.current.has(row.id)) {
            seenIdsRef.current.add(row.id)
            console.log(`[Recognize] camera=${cameraId} track=#${row.track_id} @ ${row.timestamp}`, row.response)
          }
        }

        setRows([...data].reverse()) // most recent event on top
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
    <div className="recognitions">
      <div className="recognitions__title">Face Attendance & Recognition Results</div>
      {rows.length === 0 ? (
        <p className="recognitions__empty">No attendance events yet -- appears when a face is detected in Attendance mode.</p>
      ) : (
        <div className="recognitions__scroll">
          <table>
            <thead>
              <tr>
                <th>Time</th>
                <th>Track</th>
                <th>Saved Image (PC)</th>
                <th>Result</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => {
                const summary = summarizeResponse(row.response)
                const isExpanded = expandedIds.has(row.id)
                return (
                  <tr key={row.id}>
                    <td>{row.timestamp.slice(11)}</td>
                    <td>#{row.track_id}</td>
                    <td title={row.image_path || 'No image saved'}>
                      <span className="recognitions__saved-path">
                        {row.image_path ? `💾 ${row.image_path.split(/[\\/]/).pop()}` : '—'}
                      </span>
                    </td>
                    <td>
                      <button
                        type="button"
                        className={`recognitions__badge recognitions__badge--${summary.kind}`}
                        onClick={() => toggleExpanded(row.id)}
                        title="Click to toggle raw API response"
                      >
                        {summary.label}
                      </button>
                      {isExpanded && (
                        <pre className="recognitions__response">{JSON.stringify(row.response, null, 2)}</pre>
                      )}
                    </td>
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
