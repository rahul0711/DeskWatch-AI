// Talks to app/web/attendance_server.py (the new SCRFD/AdaFace/FAISS
// backend). Reuses the same bearer token as ../api.js (single shared login).

import { AuthError, getToken } from '../api'

function authHeaders() {
  const token = getToken()
  return token ? { Authorization: `Bearer ${token}` } : {}
}

export async function fetchUsers() {
  const res = await fetch('/api/users', { headers: authHeaders() })
  if (res.status === 401) throw new AuthError()
  if (!res.ok) throw new Error(`GET /api/users -> ${res.status}`)
  return res.json()
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
  if (!res.ok) throw new Error(`DELETE /api/users/${userId} -> ${res.status}`)
  return res.json()
}

export async function deleteFace(faceId) {
  const res = await fetch(`/api/faces/${encodeURIComponent(faceId)}`, {
    method: 'DELETE',
    headers: authHeaders(),
  })
  if (res.status === 401) throw new AuthError()
  if (!res.ok) throw new Error(`DELETE /api/faces/${faceId} -> ${res.status}`)
  return res.json()
}

export async function fetchAttendance(date) {
  const qs = date ? `?date=${encodeURIComponent(date)}` : ''
  const res = await fetch(`/api/attendance${qs}`, { headers: authHeaders() })
  if (res.status === 401) throw new AuthError()
  if (!res.ok) throw new Error(`GET /api/attendance -> ${res.status}`)
  return res.json()
}
