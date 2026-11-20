// Registering people and their face photos goes to app/web/attendance_server.py
// (it needs the face model). Reading people / attendance and camera blocking go
// to the .NET backend (../db/api.js). Same bearer token as ../api.js.

import { AuthError, getToken } from '../api'
import { fetchDbAttendance, fetchPersons, setPersonCameraAccess } from '../db/api'

function authHeaders() {
  const token = getToken()
  return token ? { Authorization: `Bearer ${token}` } : {}
}

// Same shape as the Python GET /api/users; blocked_cameras only lists cameras
// the logged-in login has.
export function fetchUsers() {
  return fetchPersons()
}

// images: array of { file: Blob, filename: string }
export async function registerUser(name, employeeId, images) {
  const form = new FormData()
  form.append('name', name)
  form.append('employee_id', employeeId)
  for (const img of images) {
    form.append('images', img.file, img.filename)
  }
  const res = await fetch('/api/users', {
    method: 'POST',
    headers: authHeaders(),
    body: form,
  })
  if (res.status === 401) throw new AuthError()
  const body = await res.json().catch(() => ({}))
  if (!res.ok) {
    throw new Error(body.detail || `POST /api/users -> ${res.status}`)
  }
  return body
}

export async function addFace(userId, image) {
  const form = new FormData()
  form.append('image', image.file, image.filename)
  const res = await fetch(`/api/users/${encodeURIComponent(userId)}/faces`, {
    method: 'POST',
    headers: authHeaders(),
    body: form,
  })
  if (res.status === 401) throw new AuthError()
  const body = await res.json().catch(() => ({}))
  if (!res.ok) {
    throw new Error(body.detail || `POST /api/users/${userId}/faces -> ${res.status}`)
  }
  return body
}

export async function renameUser(userId, name) {
  const res = await fetch(`/api/users/${encodeURIComponent(userId)}`, {
    method: 'PATCH',
    headers: { ...authHeaders(), 'Content-Type': 'application/json' },
    body: JSON.stringify({ name }),
  })
  if (res.status === 401) throw new AuthError()
  const body = await res.json().catch(() => ({}))
  if (!res.ok) {
    throw new Error(body.detail || `PATCH /api/users/${userId} -> ${res.status}`)
  }
  return body
}

export async function deleteUser(userId) {
  const res = await fetch(`/api/users/${encodeURIComponent(userId)}`, {
    method: 'DELETE',
    headers: authHeaders(),
  })
  if (res.status === 401) throw new AuthError()
  const body = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(body.detail || `DELETE /api/users/${userId} -> ${res.status}`)
  return body
}

export async function deleteFace(faceId) {
  const res = await fetch(`/api/faces/${encodeURIComponent(faceId)}`, {
    method: 'DELETE',
    headers: authHeaders(),
  })
  if (res.status === 401) throw new AuthError()
  const body = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(body.detail || `DELETE /api/faces/${faceId} -> ${res.status}`)
  return body
}

// `date` is a UTC day (YYYY-MM-DD), as before.
export function fetchAttendance(date) {
  return fetchDbAttendance(date)
}

// Cameras this person is NOT allowed on. Being recognized on one of them
// raises a "not welcome" alert on that camera's live view. Only cameras the
// logged-in login has are changed; blocks on its other cameras are kept.
export function setCameraAccess(userId, blockedCameras) {
  return setPersonCameraAccess(userId, blockedCameras)
}
