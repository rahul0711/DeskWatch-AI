// Talks to the FastAPI backend (app/web/server.py). In dev, vite.config.js
// proxies /api/* to it; in production, serve this build from behind the
// same reverse proxy as the API so these relative paths keep working.

const TOKEN_KEY = 'cctv_token'
const USER_KEY = 'cctv_user'

export class AuthError extends Error {
  constructor(message = 'Not authenticated') {
    super(message)
    this.name = 'AuthError'
  }
}

export function getToken() {
  return sessionStorage.getItem(TOKEN_KEY)
}

export function setToken(token) {
  sessionStorage.setItem(TOKEN_KEY, token)
}

export function clearToken() {
  sessionStorage.removeItem(TOKEN_KEY)
  sessionStorage.removeItem(USER_KEY)
}

// The logged-in login from the .NET backend: { id, username, full_name, role,
// role_name, must_change_password, all_cameras, cameras[] }.
export function getUser() {
  try {
    return JSON.parse(sessionStorage.getItem(USER_KEY) || 'null')
  } catch {
    return null
  }
}

export function setUser(user) {
  sessionStorage.setItem(USER_KEY, JSON.stringify(user))
}

// Role checks for showing/hiding buttons. The backends enforce the same rules.
export function hasRole(...roles) {
  return roles.includes(getUser()?.role)
}

function authHeaders() {
  const token = getToken()
  return token ? { Authorization: `Bearer ${token}` } : {}
}

// Logins live in MySQL and are checked by the .NET backend
// (CCTV_Backend_Project, proxied at /api/db). Its token is accepted by the
// Python servers too.
export async function login(username, password) {
  const res = await fetch('/api/db/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  })
  const body = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(body.detail || `Login failed (${res.status})`)
  setToken(body.token)
  setUser(body.user)
  return body.user
}

export async function logout() {
  try {
    await fetch('/api/db/auth/logout', { method: 'POST', headers: authHeaders() })
  } finally {
    clearToken()
  }
}

export async function fetchCameras() {
  const res = await fetch('/api/cameras', { headers: authHeaders() })
  if (res.status === 401) throw new AuthError()
  if (!res.ok) throw new Error(`GET /api/cameras -> ${res.status}`)
  return res.json()
}

export async function addCamera(camera) {
  const res = await fetch('/api/cameras', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify(camera),
  })
  if (res.status === 401) throw new AuthError()
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail || `POST /api/cameras -> ${res.status}`)
  }
  return res.json()
}

export async function testCameraConnection(camera) {
  const res = await fetch('/api/cameras/test-connection', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify(camera),
  })
  if (res.status === 401) throw new AuthError()
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail || `POST /api/cameras/test-connection -> ${res.status}`)
  }
  return res.json()
}

export async function removeCamera(cameraId) {
  const res = await fetch(`/api/cameras/${encodeURIComponent(cameraId)}`, {
    method: 'DELETE',
    headers: authHeaders(),
  })
  if (res.status === 401) throw new AuthError()
  if (!res.ok) throw new Error(`DELETE /api/cameras -> ${res.status}`)
  return res.json()
}

export async function fetchHeadCounts(cameraId) {
  const res = await fetch(`/api/cameras/${encodeURIComponent(cameraId)}/head-counts`, {
    headers: authHeaders(),
  })
  if (res.status === 401) throw new AuthError()
  if (!res.ok) throw new Error(`GET head-counts -> ${res.status}`)
  return res.json()
}

export async function fetchRecognitions(cameraId) {
  const res = await fetch(`/api/cameras/${encodeURIComponent(cameraId)}/recognitions`, {
    headers: authHeaders(),
  })
  if (res.status === 401) throw new AuthError()
  if (!res.ok) throw new Error(`GET recognitions -> ${res.status}`)
  return res.json()
}

export async function fetchActivity(cameraId) {
  const res = await fetch(`/api/cameras/${encodeURIComponent(cameraId)}/activity`, {
    headers: authHeaders(),
  })
  if (res.status === 401) throw new AuthError()
  if (!res.ok) throw new Error(`GET activity -> ${res.status}`)
  return res.json()
}

// Admin-selected activity list: which activities are analysed and stored.
export async function fetchActivitySettings() {
  const res = await fetch('/api/activity/settings', { headers: authHeaders() })
  if (res.status === 401) throw new AuthError()
  if (!res.ok) throw new Error(`GET activity settings -> ${res.status}`)
  return res.json()
}

export async function saveActivitySettings(enabledActivities) {
  const res = await fetch('/api/activity/settings', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ enabled_activities: enabledActivities }),
  })
  if (res.status === 401) throw new AuthError()
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}))
    throw new Error(detail.detail || `PUT activity settings -> ${res.status}`)
  }
  return res.json()
}

export async function fetchOcr(cameraId) {
  const res = await fetch(`/api/cameras/${encodeURIComponent(cameraId)}/ocr`, {
    headers: authHeaders(),
  })
  if (res.status === 401) throw new AuthError()
  if (!res.ok) throw new Error(`GET ocr -> ${res.status}`)
  return res.json()
}

export async function setHeadCountSource(cameraId, source) {
  const res = await fetch(`/api/cameras/${encodeURIComponent(cameraId)}/head-count-source`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ source }),
  })
  if (res.status === 401) throw new AuthError()
  if (!res.ok) throw new Error(`PATCH head-count-source -> ${res.status}`)
  return res.json()
}

export async function setShowHud(cameraId, showHud) {
  const res = await fetch(`/api/cameras/${encodeURIComponent(cameraId)}/show-hud`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify({ show_hud: showHud }),
  })
  if (res.status === 401) throw new AuthError()
  if (!res.ok) throw new Error(`PATCH show-hud -> ${res.status}`)
  return res.json()
}

export function streamUrl(cameraId) {
  // <img> tags can't send custom headers, so the token rides along as a
  // query param here instead of the Authorization header used elsewhere.
  const token = getToken()
  return `/api/cameras/${encodeURIComponent(cameraId)}/stream?token=${encodeURIComponent(token || '')}`
}

// Live-overlay view selection: 'face' (face/attendance boxes) or
// 'activity' (body movement), with a sub-filter for the activity view.
// Rendering only -- both pipelines keep running either way, so switching
// never drops a recognition or resets an activity timer.
export async function setViewMode(cameraId, { viewMode, activityFilter } = {}) {
  const body = {}
  if (viewMode !== undefined) body.view_mode = viewMode
  if (activityFilter !== undefined) body.activity_filter = activityFilter
  const res = await fetch(`/api/cameras/${encodeURIComponent(cameraId)}/view-mode`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify(body),
  })
  if (res.status === 401) throw new AuthError()
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}))
    throw new Error(detail.detail || `PATCH view-mode -> ${res.status}`)
  }
  return res.json()
}
