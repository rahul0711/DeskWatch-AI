// Talks to the .NET backend (CCTV_Backend_Project) through the /api/db proxy in
// vite.config.js. Same bearer token as ../api.js. Errors carry the backend's
// `detail` message; 401 becomes AuthError so callers can return to login.

import { AuthError, getToken, setUser } from '../api'

async function request(method, path, body) {
  const headers = {}
  const token = getToken()
  if (token) headers.Authorization = `Bearer ${token}`
  const options = { method, headers }
  if (body !== undefined) {
    headers['Content-Type'] = 'application/json'
    options.body = JSON.stringify(body)
  }
  const res = await fetch(`/api/db${path}`, options)
  const data = await res.json().catch(() => null)
  if (res.status === 401) throw new AuthError(data?.detail)
  if (!res.ok) {
    const err = new Error(data?.detail || `${method} /api/db${path} -> ${res.status}`)
    err.code = data?.code
    err.status = res.status
    throw err
  }
  return data
}

function qs(params) {
  const p = new URLSearchParams()
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== '') p.set(k, v)
  }
  const s = p.toString()
  return s ? `?${s}` : ''
}

// ---- Auth ------------------------------------------------------------------
export async function fetchMe() {
  const me = await request('GET', '/auth/me')
  setUser(me)
  return me
}

export async function changePassword(currentPassword, newPassword) {
  const me = await request('POST', '/auth/change-password', {
    current_password: currentPassword,
    new_password: newPassword,
  })
  setUser(me)
  return me
}

// ---- Cameras -----------------------------------------------------------------
export const fetchDbCameras = () => request('GET', '/cameras')
export const updateCamera = (cameraId, changes) => request('PATCH', `/cameras/${encodeURIComponent(cameraId)}`, changes)
// { face_enabled?, activity_enabled? } -- Admin and Super Admin.
export const setCameraFeatures = (cameraId, features) =>
  request('PUT', `/cameras/${encodeURIComponent(cameraId)}/features`, features)

// ---- Logins (dashboard accounts) ---------------------------------------------
export const fetchLogins = () => request('GET', '/logins')
export const fetchRoles = () => request('GET', '/roles')
export const createLogin = (login) => request('POST', '/logins', login)
export const updateLogin = (id, changes) => request('PATCH', `/logins/${id}`, changes)
export const setLoginCameras = (id, cameraIds) => request('PUT', `/logins/${id}/cameras`, { camera_ids: cameraIds })
export const resetLoginPassword = (id, password) => request('POST', `/logins/${id}/reset-password`, { password })
export const deleteLogin = (id) => request('DELETE', `/logins/${id}`)

// ---- Tracked people ------------------------------------------------------------
export const fetchPersons = () => request('GET', '/persons')
export const setPersonCameraAccess = (id, blockedCameras) =>
  request('PUT', `/persons/${id}/camera-access`, { blocked_cameras: blockedCameras })

// ---- Reports ---------------------------------------------------------------------
export const fetchDbAttendance = (date) => request('GET', `/attendance${qs({ date })}`)
export const fetchAttendanceDaily = (date) => request('GET', `/attendance/daily${qs({ date })}`)
export const fetchActivityDaily = (date, cameraId) => request('GET', `/activity/daily${qs({ date, cameraId })}`)
export const fetchActivitySessions = (params) => request('GET', `/activity/sessions${qs(params)}`)
export const fetchAudit = (page = 1, pageSize = 50) => request('GET', `/audit${qs({ page, pageSize })}`)

// Workstations: seat + screen zones per camera (Admin -> Workstations)
export const fetchWorkstations = (cameraId) => request('GET', `/cameras/${encodeURIComponent(cameraId)}/workstations`)
export const createWorkstation = (cameraId, ws) =>
  request('POST', `/cameras/${encodeURIComponent(cameraId)}/workstations`, ws)
export const updateWorkstation = (cameraId, id, ws) =>
  request('PUT', `/cameras/${encodeURIComponent(cameraId)}/workstations/${id}`, ws)
export const deleteWorkstation = (cameraId, id) =>
  request('DELETE', `/cameras/${encodeURIComponent(cameraId)}/workstations/${id}`)
