import { useCallback, useEffect, useState } from 'react'
import { AuthError, getUser, hasRole } from '../api'
import {
  createLogin,
  deleteLogin,
  fetchAudit,
  fetchDbCameras,
  setCameraFeatures,
  fetchLogins,
  fetchRoles,
  resetLoginPassword,
  setLoginCameras,
  updateCamera,
  updateLogin,
} from '../db/api'
import WorkstationsSection from './WorkstationsSection'
import './Admin.css'

const ROLE_NAMES = { SUPER_ADMIN: 'Super Admin', ADMIN: 'Admin', VIEWER: 'Viewer' }

function formatTime(iso) {
  return iso ? new Date(iso).toLocaleString() : '—'
}

// Super Admin: every login, cameras, audit log. Admin: Viewer logins on their
// own cameras. The .NET backend enforces the same rules.
export default function AdminPage({ onAuthError }) {
  const isSuperAdmin = hasRole('SUPER_ADMIN')
  const [section, setSection] = useState('logins')

  return (
    <div className="admin">
      <nav className="admin__sections">
        <button className={section === 'logins' ? 'is-active' : ''} onClick={() => setSection('logins')}>
          Logins
        </button>
        <button className={section === 'cameras' ? 'is-active' : ''} onClick={() => setSection('cameras')}>
          Cameras
        </button>
        <button className={section === 'workstations' ? 'is-active' : ''} onClick={() => setSection('workstations')}>
          Workstations
        </button>
        {isSuperAdmin && (
          <button className={section === 'audit' ? 'is-active' : ''} onClick={() => setSection('audit')}>
            Audit log
          </button>
        )}
      </nav>
      {section === 'logins' && <LoginsSection onAuthError={onAuthError} />}
      {section === 'cameras' && <CamerasSection isSuperAdmin={isSuperAdmin} onAuthError={onAuthError} />}
      {section === 'workstations' && <WorkstationsSection onAuthError={onAuthError} />}
      {section === 'audit' && isSuperAdmin && <AuditSection onAuthError={onAuthError} />}
    </div>
  )
}

// ---- Logins -------------------------------------------------------------------

function LoginsSection({ onAuthError }) {
  const me = getUser()
  const [logins, setLogins] = useState(null)
  const [roles, setRoles] = useState([])
  const [cameras, setCameras] = useState([])
  const [error, setError] = useState(null)
  const [editing, setEditing] = useState(null) // login id
  const [busyId, setBusyId] = useState(null)

  const fail = useCallback(
    (err) => {
      if (err instanceof AuthError) onAuthError()
      else setError(err.message)
    },
    [onAuthError],
  )

  const load = useCallback(async () => {
    try {
      const [l, r, c] = await Promise.all([fetchLogins(), fetchRoles(), fetchDbCameras()])
      setLogins(l)
      setRoles(r)
      setCameras(c.filter((cam) => cam.is_active))
      setError(null)
    } catch (err) {
      fail(err)
    }
  }, [fail])

  useEffect(() => {
    load()
  }, [load])

  async function run(id, action) {
    setBusyId(id)
    setError(null)
    try {
      await action()
      await load()
      return true
    } catch (err) {
      fail(err)
      return false
    } finally {
      setBusyId(null)
    }
  }

  function cameraNames(login) {
    if (login.all_cameras) return 'All cameras'
    if (!login.camera_ids.length) return 'No cameras'
    return login.camera_ids.map((id) => cameras.find((c) => c.camera_id === id)?.name || id).join(', ')
  }

  return (
    <section className="admin__panel">
      <h2>Logins</h2>
      <p className="admin__hint">
        {me?.role === 'SUPER_ADMIN'
          ? 'Super Admins manage every login. Admins and Viewers only see the cameras ticked for them.'
          : 'You can create and manage Viewer logins on your own cameras.'}
      </p>
      {error && <p className="admin__error">{error}</p>}

      <CreateLoginForm roles={roles} cameras={cameras} me={me} onCreate={(body) => run('new', () => createLogin(body))} />

      {logins === null ? (
        <p className="admin__hint">Loading…</p>
      ) : (
        <div className="admin__table-wrap">
          <table className="admin__table">
            <thead>
              <tr>
                <th>Login</th>
                <th>Role</th>
                <th>Cameras</th>
                <th>Status</th>
                <th>Last login</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {logins.map((l) =>
                editing === l.id ? (
                  <EditLoginRow
                    key={l.id}
                    login={l}
                    roles={roles}
                    cameras={cameras}
                    me={me}
                    busy={busyId === l.id}
                    onCancel={() => setEditing(null)}
                    onSave={async (changes, cameraIds) => {
                      const ok = await run(l.id, async () => {
                        await updateLogin(l.id, changes)
                        const role = changes.role || l.role
                        if (role !== 'SUPER_ADMIN' && !changes.all_cameras) await setLoginCameras(l.id, cameraIds)
                      })
                      if (ok) setEditing(null)
                    }}
                  />
                ) : (
                  <tr key={l.id} className={l.is_active ? '' : 'admin__row--off'}>
                    <td>
                      <strong>{l.full_name}</strong>
                      <div className="admin__sub">
                        {l.username}
                        {l.email ? ` · ${l.email}` : ''}
                        {l.id === me?.id ? ' · you' : ''}
                      </div>
                    </td>
                    <td>
                      <span className={`admin__role admin__role--${l.role.toLowerCase()}`}>{ROLE_NAMES[l.role] || l.role}</span>
                    </td>
                    <td className="admin__cams">{cameraNames(l)}</td>
                    <td>
                      {!l.is_active
                        ? 'Disabled'
                        : l.locked_until && new Date(l.locked_until) > new Date()
                          ? 'Locked'
                          : l.must_change_password
                            ? 'Must set password'
                            : 'Active'}
                    </td>
                    <td className="admin__sub">{formatTime(l.last_login_at)}</td>
                    <td className="admin__actions">
                      {l.id !== me?.id && (
                        <>
                          <button disabled={busyId === l.id} onClick={() => setEditing(l.id)}>
                            Edit
                          </button>
                          <button
                            disabled={busyId === l.id}
                            onClick={() => run(l.id, () => updateLogin(l.id, { is_active: !l.is_active }))}
                          >
                            {l.is_active ? 'Disable' : 'Enable'}
                          </button>
                          <button
                            disabled={busyId === l.id}
                            onClick={() => {
                              const pw = prompt(`New temporary password for ${l.username} (at least 8 characters).\nThey must change it at next login.`)
                              if (pw) run(l.id, () => resetLoginPassword(l.id, pw)).then((ok) => ok && alert('Password reset.'))
                            }}
                          >
                            Reset password
                          </button>
                          <button
                            className="admin__danger"
                            disabled={busyId === l.id}
                            onClick={() => confirm(`Delete login ${l.username}? This cannot be undone.`) && run(l.id, () => deleteLogin(l.id))}
                          >
                            Delete
                          </button>
                        </>
                      )}
                    </td>
                  </tr>
                ),
              )}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}

function CameraChecklist({ cameras, selected, onChange, disabled }) {
  return (
    <div className="admin__checklist">
      {cameras.map((c) => (
        <label key={c.camera_id}>
          <input
            type="checkbox"
            checked={selected.includes(c.camera_id)}
            disabled={disabled}
            onChange={(e) =>
              onChange(e.target.checked ? [...selected, c.camera_id] : selected.filter((id) => id !== c.camera_id))
            }
          />
          Camera {c.camera_no} · {c.name}
        </label>
      ))}
    </div>
  )
}

function CreateLoginForm({ roles, cameras, me, onCreate }) {
  const empty = { username: '', full_name: '', email: '', role: '', all_cameras: false, camera_ids: [], password: '' }
  const [open, setOpen] = useState(false)
  const [form, setForm] = useState(empty)
  const role = form.role || roles[roles.length - 1]?.code || ''
  const set = (k) => (e) => setForm((f) => ({ ...f, [k]: e.target.type === 'checkbox' ? e.target.checked : e.target.value }))

  if (!open) {
    return (
      <button className="admin__primary" onClick={() => setOpen(true)} disabled={!roles.length}>
        + New login
      </button>
    )
  }

  return (
    <form
      className="admin__form"
      onSubmit={async (e) => {
        e.preventDefault()
        const ok = await onCreate({ ...form, role, email: form.email || null })
        if (ok) {
          setForm(empty)
          setOpen(false)
        }
      }}
    >
      <label>
        Username
        <input value={form.username} onChange={set('username')} required minLength={3} autoComplete="off" />
      </label>
      <label>
        Full name
        <input value={form.full_name} onChange={set('full_name')} required />
      </label>
      <label>
        Email (optional)
        <input type="email" value={form.email} onChange={set('email')} />
      </label>
      <label>
        Role
        <select value={role} onChange={set('role')}>
          {roles.map((r) => (
            <option key={r.code} value={r.code}>
              {r.name}
            </option>
          ))}
        </select>
      </label>
      <label>
        Temporary password
        <input type="text" value={form.password} onChange={set('password')} required minLength={8} autoComplete="off" />
      </label>
      {role !== 'SUPER_ADMIN' && (
        <div className="admin__form-wide">
          {me?.all_cameras && (
            <label className="admin__inline">
              <input type="checkbox" checked={form.all_cameras} onChange={set('all_cameras')} /> All cameras (including ones
              added later)
            </label>
          )}
          {!form.all_cameras && (
            <CameraChecklist
              cameras={cameras}
              selected={form.camera_ids}
              onChange={(ids) => setForm((f) => ({ ...f, camera_ids: ids }))}
            />
          )}
        </div>
      )}
      <div className="admin__form-wide admin__form-actions">
        <button type="submit" className="admin__primary">
          Create login
        </button>
        <button type="button" onClick={() => setOpen(false)}>
          Cancel
        </button>
        <span className="admin__hint">They must change the password at first login.</span>
      </div>
    </form>
  )
}

function EditLoginRow({ login, roles, cameras, me, busy, onCancel, onSave }) {
  const [fullName, setFullName] = useState(login.full_name)
  const [email, setEmail] = useState(login.email || '')
  const [role, setRole] = useState(login.role)
  const [allCameras, setAllCameras] = useState(login.all_cameras && login.role !== 'SUPER_ADMIN')
  const [cameraIds, setCameraIds] = useState(login.camera_ids)

  return (
    <tr className="admin__row--editing">
      <td>
        <input value={fullName} onChange={(e) => setFullName(e.target.value)} placeholder="Full name" />
        <input value={email} onChange={(e) => setEmail(e.target.value)} placeholder="Email" type="email" />
        <div className="admin__sub">{login.username}</div>
      </td>
      <td>
        <select value={role} onChange={(e) => setRole(e.target.value)}>
          {roles.map((r) => (
            <option key={r.code} value={r.code}>
              {r.name}
            </option>
          ))}
        </select>
      </td>
      <td colSpan={3}>
        {role === 'SUPER_ADMIN' ? (
          <span className="admin__hint">Super Admins see every camera.</span>
        ) : (
          <>
            {(me?.all_cameras || allCameras) && (
              <label className="admin__inline">
                <input type="checkbox" checked={allCameras} onChange={(e) => setAllCameras(e.target.checked)} /> All
                cameras
              </label>
            )}
            {!allCameras && <CameraChecklist cameras={cameras} selected={cameraIds} onChange={setCameraIds} disabled={busy} />}
          </>
        )}
      </td>
      <td className="admin__actions">
        <button
          className="admin__primary"
          disabled={busy || !fullName.trim()}
          onClick={() =>
            onSave({ full_name: fullName.trim(), email: email.trim(), role, all_cameras: role !== 'SUPER_ADMIN' && allCameras }, cameraIds)
          }
        >
          {busy ? 'Saving…' : 'Save'}
        </button>
        <button disabled={busy} onClick={onCancel}>
          Cancel
        </button>
      </td>
    </tr>
  )
}

// ---- Cameras ---------------------------------------------------------------------
// Admin + Super Admin: face / body-movement switches for their cameras.
// Super Admin only: rename, activate/deactivate.

function CamerasSection({ isSuperAdmin, onAuthError }) {
  const [cameras, setCameras] = useState(null)
  const [error, setError] = useState(null)

  const load = useCallback(() => {
    fetchDbCameras()
      .then((c) => {
        setCameras(c)
        setError(null)
      })
      .catch((err) => (err instanceof AuthError ? onAuthError() : setError(err.message)))
  }, [onAuthError])

  useEffect(() => {
    load()
  }, [load])

  async function save(cameraId, changes) {
    try {
      await updateCamera(cameraId, changes)
      load()
    } catch (err) {
      if (err instanceof AuthError) onAuthError()
      else setError(err.message)
    }
  }

  async function saveFeatures(cameraId, features) {
    try {
      await setCameraFeatures(cameraId, features)
      load()
    } catch (err) {
      if (err instanceof AuthError) onAuthError()
      else setError(err.message)
    }
  }

  return (
    <section className="admin__panel">
      <h2>Cameras</h2>
      <p className="admin__hint">
        Choose what each camera analyses: Face (attendance, names, restricted-area alerts), Body movement (posture,
        phone, monitor, computer), or both. A change restarts only that camera, within about 10 seconds. Body
        movement shows names only on cameras that also have Face on.
      </p>
      {error && <p className="admin__error">{error}</p>}
      {cameras && (
        <div className="admin__table-wrap">
          <table className="admin__table">
            <thead>
              <tr>
                <th>No.</th>
                <th>Id</th>
                <th>Name</th>
                <th>IP</th>
                <th>Brand</th>
                <th>Active</th>
                <th>Face</th>
                <th>Body movement</th>
                {isSuperAdmin && <th />}
              </tr>
            </thead>
            <tbody>
              {cameras.map((c) => (
                <tr key={c.camera_id} className={c.is_active ? '' : 'admin__row--off'}>
                  <td>{c.camera_no}</td>
                  <td className="admin__sub">{c.camera_id}</td>
                  <td>{c.name}</td>
                  <td className="admin__sub">{c.ip_address || '—'}</td>
                  <td className="admin__sub">{c.brand || '—'}</td>
                  <td>{c.is_active ? 'Yes' : 'No'}</td>
                  <td>
                    <input
                      type="checkbox"
                      checked={c.face_enabled}
                      aria-label={`Face on ${c.name}`}
                      onChange={(e) => saveFeatures(c.camera_id, { face_enabled: e.target.checked })}
                    />
                  </td>
                  <td>
                    <input
                      type="checkbox"
                      checked={c.activity_enabled}
                      aria-label={`Body movement on ${c.name}`}
                      onChange={(e) => saveFeatures(c.camera_id, { activity_enabled: e.target.checked })}
                    />
                  </td>
                  {isSuperAdmin && (
                    <td className="admin__actions">
                      <button
                        onClick={() => {
                          const name = prompt('Camera name', c.name)
                          if (name && name.trim() && name.trim() !== c.name) save(c.camera_id, { name: name.trim() })
                        }}
                      >
                        Rename
                      </button>
                      <button onClick={() => save(c.camera_id, { is_active: !c.is_active })}>
                        {c.is_active ? 'Deactivate' : 'Activate'}
                      </button>
                    </td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}

// ---- Audit log (Super Admin) ---------------------------------------------------------

function AuditSection({ onAuthError }) {
  const [page, setPage] = useState(1)
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    let cancelled = false
    fetchAudit(page, 50)
      .then((d) => !cancelled && setData(d))
      .catch((err) => (err instanceof AuthError ? onAuthError() : setError(err.message)))
    return () => {
      cancelled = true
    }
  }, [page, onAuthError])

  const pages = data ? Math.max(1, Math.ceil(data.total / data.page_size)) : 1

  return (
    <section className="admin__panel">
      <h2>Audit log</h2>
      {error && <p className="admin__error">{error}</p>}
      {data && (
        <>
          <div className="admin__table-wrap">
            <table className="admin__table">
              <thead>
                <tr>
                  <th>When</th>
                  <th>Who</th>
                  <th>Action</th>
                  <th>Target</th>
                  <th>Details</th>
                  <th>IP</th>
                </tr>
              </thead>
              <tbody>
                {data.items.map((a) => (
                  <tr key={a.id}>
                    <td className="admin__sub">{formatTime(a.at)}</td>
                    <td>{a.username || '—'}</td>
                    <td>
                      <code>{a.action}</code>
                    </td>
                    <td className="admin__sub">{a.target_type ? `${a.target_type} ${a.target_id ?? ''}` : '—'}</td>
                    <td className="admin__details">{a.details || ''}</td>
                    <td className="admin__sub">{a.ip_address || ''}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="admin__pager">
            <button disabled={page <= 1} onClick={() => setPage((p) => p - 1)}>
              ‹ Newer
            </button>
            <span className="admin__sub">
              Page {page} of {pages} · {data.total} entries
            </span>
            <button disabled={page >= pages} onClick={() => setPage((p) => p + 1)}>
              Older ›
            </button>
          </div>
        </>
      )}
    </section>
  )
}
