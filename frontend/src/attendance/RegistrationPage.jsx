import { useCallback, useEffect, useRef, useState } from 'react'
import { addFace, deleteFace, deleteUser, fetchUsers, registerUser, renameUser } from './api'
import { AuthError } from '../api'
import './RegistrationPage.css'

const ANGLE_GUIDES = [
  'Look straight at the camera (frontal)',
  'Turn slightly left',
  'Turn slightly right',
  'Tilt slightly up',
  'Tilt slightly down',
]

let nextImageId = 1

export default function RegistrationPage({ onAuthError, prefillImage, onPrefillConsumed }) {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const fileInputRef = useRef(null)
  const [streamError, setStreamError] = useState(null)
  const [captured, setCaptured] = useState([]) // { id, previewUrl, file, filename }
  const [name, setName] = useState('')
  const [employeeId, setEmployeeId] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [submitResults, setSubmitResults] = useState(null)
  const [submitError, setSubmitError] = useState(null)

  const [users, setUsers] = useState(null)
  const [usersError, setUsersError] = useState(null)
  const [editingId, setEditingId] = useState(null)
  const [editName, setEditName] = useState('')
  const [savingName, setSavingName] = useState(false)

  const guideIndex = Math.min(captured.length, ANGLE_GUIDES.length - 1)

  const loadUsers = useCallback(async () => {
    try {
      const data = await fetchUsers()
      setUsers(data)
      setUsersError(null)
    } catch (err) {
      if (err instanceof AuthError) onAuthError()
      else setUsersError(err.message)
    }
  }, [onAuthError])

  useEffect(() => {
    loadUsers()
  }, [loadUsers])

  useEffect(() => {
    let stream
    navigator.mediaDevices
      ?.getUserMedia({ video: { width: 640, height: 480 } })
      .then((s) => {
        stream = s
        if (videoRef.current) videoRef.current.srcObject = s
      })
      .catch((err) => setStreamError(err.message))
    return () => {
      stream?.getTracks().forEach((t) => t.stop())
    }
  }, [])

  function addCapturedFile(file, filename) {
    const previewUrl = URL.createObjectURL(file)
    setCaptured((prev) => [...prev, { id: nextImageId++, previewUrl, file, filename }])
  }

  useEffect(() => {
    if (!prefillImage) return
    addCapturedFile(prefillImage, `unknown_${Date.now()}.jpg`)
    onPrefillConsumed?.()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [prefillImage])

  function handleCapture() {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas || video.readyState < 2) return
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    const ctx = canvas.getContext('2d')
    ctx.drawImage(video, 0, 0)
    canvas.toBlob((blob) => {
      if (blob) addCapturedFile(blob, `webcam_${Date.now()}.jpg`)
    }, 'image/jpeg', 0.92)
  }

  function handleFileChange(e) {
    for (const file of e.target.files) {
      addCapturedFile(file, file.name)
    }
    e.target.value = ''
  }

  function removeCaptured(id) {
    setCaptured((prev) => {
      const found = prev.find((c) => c.id === id)
      if (found) URL.revokeObjectURL(found.previewUrl)
      return prev.filter((c) => c.id !== id)
    })
  }

  async function handleSubmit(e) {
    e.preventDefault()
    if (!name.trim() || !employeeId.trim() || captured.length === 0) return
    setSubmitting(true)
    setSubmitError(null)
    setSubmitResults(null)
    try {
      const result = await registerUser(name.trim(), employeeId.trim(), captured)
      setSubmitResults(result.images)
      const anyAccepted = result.images.some((r) => r.accepted)
      if (anyAccepted) {
        setName('')
        setEmployeeId('')
        captured.forEach((c) => URL.revokeObjectURL(c.previewUrl))
        setCaptured([])
        loadUsers()
      }
    } catch (err) {
      if (err instanceof AuthError) onAuthError()
      else setSubmitError(err.message)
    } finally {
      setSubmitting(false)
    }
  }

  async function handleAddAngle(userId, file) {
    try {
      await addFace(userId, { file, filename: file.name || `angle_${Date.now()}.jpg` })
      loadUsers()
    } catch (err) {
      if (err instanceof AuthError) onAuthError()
      else alert(err.message)
    }
  }

  async function handleDeleteFace(faceId) {
    try {
      await deleteFace(faceId)
      loadUsers()
    } catch (err) {
      if (err instanceof AuthError) onAuthError()
      else alert(err.message)
    }
  }

  function startEdit(user) {
    setEditingId(user.id)
    setEditName(user.name)
  }

  function cancelEdit() {
    setEditingId(null)
    setEditName('')
  }

  async function saveEdit(userId) {
    const newName = editName.trim()
    if (!newName) return
    setSavingName(true)
    try {
      const updated = await renameUser(userId, newName)
      setUsers((prev) => prev.map((u) => (u.id === userId ? { ...u, name: updated.name } : u)))
      cancelEdit()
    } catch (err) {
      if (err instanceof AuthError) onAuthError()
      else alert(err.message)
    } finally {
      setSavingName(false)
    }
  }

  async function handleDeleteUser(userId) {
    if (!confirm('Delete this user and all their stored faces?')) return
    try {
      await deleteUser(userId)
      loadUsers()
    } catch (err) {
      if (err instanceof AuthError) onAuthError()
      else alert(err.message)
    }
  }

  return (
    <div className="registration">
      <section className="registration__panel">
        <h2>Register a person</h2>

        <div className="registration__capture">
          <div className="registration__camera">
            {streamError ? (
              <p className="registration__camera-error">Webcam unavailable: {streamError}</p>
            ) : (
              <video ref={videoRef} autoPlay playsInline muted />
            )}
            <canvas ref={canvasRef} style={{ display: 'none' }} />
            <p className="registration__guide">{ANGLE_GUIDES[guideIndex]}</p>
            <div className="registration__camera-actions">
              <button type="button" onClick={handleCapture} disabled={!!streamError}>
                📸 Capture angle
              </button>
              <button type="button" onClick={() => fileInputRef.current?.click()}>
                Upload photo(s)
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                multiple
                style={{ display: 'none' }}
                onChange={handleFileChange}
              />
            </div>
          </div>

          <div className="registration__thumbs">
            {captured.length === 0 && <p className="registration__hint">No angles captured yet</p>}
            {captured.map((c) => (
              <div key={c.id} className="registration__thumb">
                <img src={c.previewUrl} alt="captured angle" />
                <button type="button" className="registration__thumb-remove" onClick={() => removeCaptured(c.id)}>
                  ×
                </button>
              </div>
            ))}
          </div>
        </div>

        <form className="registration__form" onSubmit={handleSubmit}>
          <label>
            Name
            <input value={name} onChange={(e) => setName(e.target.value)} required />
          </label>
          <label>
            Employee ID
            <input value={employeeId} onChange={(e) => setEmployeeId(e.target.value)} required />
          </label>
          <button type="submit" disabled={submitting || captured.length === 0}>
            {submitting ? 'Registering…' : `Register (${captured.length} angle${captured.length === 1 ? '' : 's'})`}
          </button>
        </form>

        {submitError && <p className="registration__error">{submitError}</p>}

        {submitResults && (
          <ul className="registration__results">
            {submitResults.map((r, i) => (
              <li key={i} className={r.accepted ? 'registration__result--ok' : 'registration__result--fail'}>
                {r.accepted
                  ? `✓ ${r.filename} — stored (${r.face_width_px}px face)`
                  : `✗ ${r.filename} — ${r.reason}`}
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="registration__panel">
        <h2>Enrolled people</h2>
        {usersError && <p className="registration__error">Backend unreachable: {usersError}</p>}
        {users === null && !usersError && <p className="registration__hint">Loading…</p>}
        {users !== null && users.length === 0 && <p className="registration__hint">No one registered yet.</p>}

        <div className="registration__users">
          {users?.map((u) => (
            <div key={u.id} className="registration__user-card">
              <div className="registration__user-header">
                {editingId === u.id ? (
                  <form
                    className="registration__edit-form"
                    onSubmit={(e) => {
                      e.preventDefault()
                      saveEdit(u.id)
                    }}
                  >
                    <input
                      className="registration__edit-input"
                      value={editName}
                      onChange={(e) => setEditName(e.target.value)}
                      onKeyDown={(e) => e.key === 'Escape' && cancelEdit()}
                      disabled={savingName}
                      autoFocus
                    />
                    <span className="registration__user-empid"> · {u.employee_id}</span>
                    <button type="submit" className="registration__edit-user" disabled={savingName || !editName.trim()}>
                      {savingName ? 'Saving…' : 'Save'}
                    </button>
                    <button type="button" className="registration__edit-user" onClick={cancelEdit} disabled={savingName}>
                      Cancel
                    </button>
                  </form>
                ) : (
                  <div>
                    <strong>{u.name}</strong>
                    <span className="registration__user-empid"> · {u.employee_id}</span>
                  </div>
                )}
                <div className="registration__user-actions">
                  {editingId !== u.id && (
                    <button type="button" className="registration__edit-user" onClick={() => startEdit(u)}>
                      Edit
                    </button>
                  )}
                  <button type="button" className="registration__delete-user" onClick={() => handleDeleteUser(u.id)}>
                    Delete
                  </button>
                </div>
              </div>
              <div className="registration__thumbs">
                {u.faces.map((f) => (
                  <div key={f.id} className="registration__thumb registration__thumb--small">
                    <img src={f.thumbnail_url} alt="stored angle" title={`${f.face_width_px}px, score ${f.det_score.toFixed(2)}`} />
                    <button type="button" className="registration__thumb-remove" onClick={() => handleDeleteFace(f.id)}>
                      ×
                    </button>
                  </div>
                ))}
                <label className="registration__add-angle">
                  +
                  <input
                    type="file"
                    accept="image/*"
                    style={{ display: 'none' }}
                    onChange={(e) => {
                      const file = e.target.files[0]
                      if (file) handleAddAngle(u.id, file)
                      e.target.value = ''
                    }}
                  />
                </label>
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  )
}
