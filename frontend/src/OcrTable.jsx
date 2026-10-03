import { useEffect, useState } from 'react'
import { AuthError, fetchOcr } from './api'

const POLL_INTERVAL_MS = 5000

export default function OcrTable({ cameraId, onAuthError }) {
  const [rows, setRows] = useState([])

  useEffect(() => {
    let cancelled = false

    async function poll() {
      try {
        const data = await fetchOcr(cameraId)
        if (!cancelled) setRows(data)
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
      <div className="activity__title">Detected text (walls, signs, boards, carried items)</div>
      {rows.length === 0 ? (
        <p className="activity__empty">No text detected yet -- appears as soon as OCR finds a readable region.</p>
      ) : (
        <div className="activity__scroll">
          <table>
            <thead>
              <tr>
                <th>Region</th>
                <th title="Multi-frame-voted text reading">Text</th>
                <th title="Average confidence of the winning reading, 0-1">Confidence</th>
                <th title="How many recent frames agreed on this reading">Votes</th>
                <th title="Static (wall/sign) or on a tracked moving person/object">Kind</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.track_id}>
                  <td>#{row.track_id}</td>
                  <td className={row.confirmed ? 'ocr__text ocr__text--confirmed' : 'ocr__text'}>
                    {row.text ? `"${row.text}"` : '…'}
                  </td>
                  <td>{row.confidence.toFixed(2)}</td>
                  <td>{row.votes}</td>
                  <td>{row.moving ? '🚶 moving' : '📌 static'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
