import { useState } from 'react'
import { AuthError } from './api'
import { changePassword } from './db/api'

// Shown instead of the app while a login must change its temporary password,
// and from the header's "Change password" button.
export default function ChangePasswordPage({ user, forced, onDone, onCancel, onAuthError }) {
  const [current, setCurrent] = useState('')
  const [next, setNext] = useState('')
  const [confirm, setConfirm] = useState('')
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)

  async function handleSubmit(e) {
    e.preventDefault()
    setError(null)
    if (next !== confirm) {
      setError('The two new passwords do not match')
      return
    }
    setBusy(true)
    try {
      onDone(await changePassword(current, next))
    } catch (err) {
      if (err instanceof AuthError) onAuthError()
      else setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="login">
      <form className="login__card" onSubmit={handleSubmit}>
        <h1>{forced ? 'Set your password' : 'Change password'}</h1>
        <p className="login__subtitle">
          {forced
            ? `Welcome, ${user.full_name}. Replace the temporary password before you continue.`
            : `Signed in as ${user.username}. Other devices will be signed out.`}
        </p>

        <label className="login__field">
          Current password
          <input
            type="password"
            value={current}
            onChange={(e) => setCurrent(e.target.value)}
            autoComplete="current-password"
            autoFocus
            required
          />
        </label>
        <label className="login__field">
          New password (at least 8 characters)
          <input
            type="password"
            value={next}
            onChange={(e) => setNext(e.target.value)}
            autoComplete="new-password"
            minLength={8}
            required
          />
        </label>
        <label className="login__field">
          Repeat new password
          <input
            type="password"
            value={confirm}
            onChange={(e) => setConfirm(e.target.value)}
            autoComplete="new-password"
            minLength={8}
            required
          />
        </label>

        {error && <p className="login__error">{error}</p>}

        <button type="submit" disabled={busy}>
          {busy ? 'Saving…' : 'Save password'}
        </button>
        {onCancel && (
          <button type="button" className="login__secondary" onClick={onCancel} disabled={busy}>
            {forced ? 'Log out' : 'Cancel'}
          </button>
        )}
      </form>
    </div>
  )
}
