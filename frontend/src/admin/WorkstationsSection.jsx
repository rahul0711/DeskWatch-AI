import { useCallback, useEffect, useRef, useState } from 'react'
import { AuthError, getToken } from '../api'
import { createWorkstation, deleteWorkstation, fetchDbCameras, fetchWorkstations, updateWorkstation } from '../db/api'

// ---- Workstations ------------------------------------------------------------------
// Admin + Super Admin: on a still from the camera, drag a box around a chair
// (seat), then around the screen the person in it faces. The camera server
// picks them up within ~5 s and uses them for sitting/standing and "looking at
// their own screen". Boxes are stored as fractions (0..1) of the frame.

const SEAT_COLOR = '#3b82f6'
const SCREEN_COLOR = '#f59e0b'

export default function WorkstationsSection({ onAuthError }) {
  const [cameras, setCameras] = useState([])
  const [cameraId, setCameraId] = useState('')
  const [list, setList] = useState([])
  const [image, setImage] = useState(null)
  const [error, setError] = useState(null)
  const [draft, setDraft] = useState(null) // { id?, name, seat, screen }
  const [step, setStep] = useState('seat')
  const [busy, setBusy] = useState(false)

  const fail = useCallback((err) => (err instanceof AuthError ? onAuthError() : setError(err.message)), [onAuthError])

  useEffect(() => {
    fetchDbCameras()
      .then((c) => {
        const active = c.filter((cam) => cam.is_active)
        setCameras(active)
        if (active.length) setCameraId((id) => id || active[0].camera_id)
      })
      .catch(fail)
  }, [fail])

  const loadList = useCallback(() => {
    if (!cameraId) return
    fetchWorkstations(cameraId)
      .then((l) => {
        setList(l)
        setError(null)
      })
      .catch(fail)
  }, [cameraId, fail])

  const loadImage = useCallback(async () => {
    if (!cameraId) return
    setImage(null)
    try {
      const res = await fetch(`/api/cameras/${encodeURIComponent(cameraId)}/snapshot`, {
        headers: { Authorization: `Bearer ${getToken()}` },
      })
      if (res.status === 401) throw new AuthError()
      if (!res.ok) {
        const data = await res.json().catch(() => null)
        throw new Error(data?.detail || `Snapshot failed (${res.status})`)
      }
      const url = URL.createObjectURL(await res.blob())
      setImage((old) => {
        if (old) URL.revokeObjectURL(old)
        return url
      })
    } catch (err) {
      fail(err)
    }
  }, [cameraId, fail])

  useEffect(() => {
    setDraft(null)
    loadList()
    loadImage()
  }, [loadList, loadImage])

  function startNew() {
    setDraft({ name: `Desk ${list.length + 1}`, seat: null, screen: null })
    setStep('seat')
  }

  function startEdit(ws) {
    setDraft({ id: ws.id, name: ws.name, seat: ws.seat, screen: ws.screen })
    setStep('seat')
  }

  function onBox(box) {
    if (!draft) return
    if (step === 'seat') {
      setDraft({ ...draft, seat: box })
      setStep('screen')
    } else {
      setDraft({ ...draft, screen: box })
    }
  }

  async function save() {
    setBusy(true)
    try {
      const body = { name: draft.name.trim(), seat: draft.seat, screen: draft.screen }
      if (draft.id) await updateWorkstation(cameraId, draft.id, body)
      else await createWorkstation(cameraId, body)
      setDraft(null)
      loadList()
    } catch (err) {
      fail(err)
    } finally {
      setBusy(false)
    }
  }

  async function remove(ws) {
    if (!confirm(`Delete workstation "${ws.name}"?`)) return
    try {
      await deleteWorkstation(cameraId, ws.id)
      loadList()
    } catch (err) {
      fail(err)
    }
  }

  const shown = list.filter((ws) => ws.id !== draft?.id)

  return (
    <section className="admin__panel">
      <h2>Workstations</h2>
      <p className="admin__hint">
        Mark where people sit and which screen is theirs. Someone in a seat box counts towards sitting even when their
        knees are hidden by the desk, and looking at their own screen counts as monitor use even when the screen is
        not detected. Click <b>New workstation</b>, drag a box around the chair (blue), then around that person&apos;s
        screen (orange; optional), name it and save.
      </p>
      {error && <p className="admin__error">{error}</p>}

      <div className="admin__form-actions">
        <select value={cameraId} onChange={(e) => setCameraId(e.target.value)}>
          {cameras.map((c) => (
            <option key={c.camera_id} value={c.camera_id}>
              {c.name} ({c.camera_id})
            </option>
          ))}
        </select>
        <button onClick={loadImage}>Refresh picture</button>
        {!draft && (
          <button className="admin__primary" onClick={startNew} disabled={!image}>
            New workstation
          </button>
        )}
      </div>

      {draft && (
        <div className="admin__form-actions ws__draft">
          <input value={draft.name} maxLength={100} onChange={(e) => setDraft({ ...draft, name: e.target.value })} />
          <button className={step === 'seat' ? 'is-active' : ''} onClick={() => setStep('seat')}>
            {draft.seat ? 'Redraw seat' : 'Draw seat'}
          </button>
          <button className={step === 'screen' ? 'is-active' : ''} onClick={() => setStep('screen')} disabled={!draft.seat}>
            {draft.screen ? 'Redraw screen' : 'Draw screen'}
          </button>
          {draft.screen && <button onClick={() => setDraft({ ...draft, screen: null })}>No screen</button>}
          <button className="admin__primary" onClick={save} disabled={busy || !draft.seat || !draft.name.trim()}>
            Save
          </button>
          <button onClick={() => setDraft(null)}>Cancel</button>
          <span className="admin__hint">Now drag around the {step === 'seat' ? 'chair' : 'screen'}.</span>
        </div>
      )}

      {image ? (
        <ZoneCanvas image={image} workstations={shown} draft={draft} drawing={!!draft} onBox={onBox} />
      ) : (
        !error && <p className="admin__hint">Loading picture…</p>
      )}

      {list.length > 0 && (
        <div className="admin__table-wrap">
          <table className="admin__table">
            <thead>
              <tr>
                <th>Name</th>
                <th>Seat</th>
                <th>Screen</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {list.map((ws) => (
                <tr key={ws.id}>
                  <td>{ws.name}</td>
                  <td className="admin__sub">{pct(ws.seat)}</td>
                  <td className="admin__sub">{ws.screen ? pct(ws.screen) : '—'}</td>
                  <td className="admin__actions">
                    <button onClick={() => startEdit(ws)} disabled={!!draft}>
                      Edit
                    </button>
                    <button onClick={() => remove(ws)}>Delete</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}

function pct(b) {
  return b.map((v) => `${Math.round(v * 100)}%`).join(', ')
}

// The picture with saved zones drawn on it; when drawing, a drag gives a box
// (fractions of the picture) to onBox.
function ZoneCanvas({ image, workstations, draft, drawing, onBox }) {
  const wrap = useRef(null)
  const [drag, setDrag] = useState(null) // { x0, y0, x1, y1 } fractions

  function at(e) {
    const r = wrap.current.getBoundingClientRect()
    const clamp = (v) => Math.min(1, Math.max(0, v))
    return [clamp((e.clientX - r.left) / r.width), clamp((e.clientY - r.top) / r.height)]
  }

  function down(e) {
    if (!drawing) return
    e.preventDefault()
    e.currentTarget.setPointerCapture(e.pointerId)
    const [x, y] = at(e)
    setDrag({ x0: x, y0: y, x1: x, y1: y })
  }

  function move(e) {
    if (!drag) return
    const [x, y] = at(e)
    setDrag({ ...drag, x1: x, y1: y })
  }

  function up() {
    if (!drag) return
    const box = [Math.min(drag.x0, drag.x1), Math.min(drag.y0, drag.y1), Math.max(drag.x0, drag.x1), Math.max(drag.y0, drag.y1)]
    setDrag(null)
    if (box[2] - box[0] > 0.005 && box[3] - box[1] > 0.005) onBox(box.map((v) => Math.round(v * 10000) / 10000))
  }

  return (
    <div
      ref={wrap}
      className={`ws__canvas${drawing ? ' ws__canvas--drawing' : ''}`}
      onPointerDown={down}
      onPointerMove={move}
      onPointerUp={up}
    >
      <img src={image} alt="Camera still" draggable={false} />
      {workstations.map((ws) => (
        <Zones key={ws.id} name={ws.name} seat={ws.seat} screen={ws.screen} dim={drawing} />
      ))}
      {draft && <Zones name={draft.name} seat={draft.seat} screen={draft.screen} />}
      {drag && (
        <Box
          box={[Math.min(drag.x0, drag.x1), Math.min(drag.y0, drag.y1), Math.max(drag.x0, drag.x1), Math.max(drag.y0, drag.y1)]}
          color="#ffffff"
        />
      )}
    </div>
  )
}

function Zones({ name, seat, screen, dim }) {
  return (
    <>
      {seat && <Box box={seat} color={SEAT_COLOR} label={name} dim={dim} />}
      {screen && <Box box={screen} color={SCREEN_COLOR} dim={dim} />}
    </>
  )
}

function Box({ box, color, label, dim }) {
  const [x1, y1, x2, y2] = box
  return (
    <div
      className="ws__box"
      style={{
        left: `${x1 * 100}%`,
        top: `${y1 * 100}%`,
        width: `${(x2 - x1) * 100}%`,
        height: `${(y2 - y1) * 100}%`,
        borderColor: color,
        opacity: dim ? 0.45 : 1,
      }}
    >
      {label && (
        <span className="ws__label" style={{ background: color }}>
          {label}
        </span>
      )}
    </div>
  )
}
