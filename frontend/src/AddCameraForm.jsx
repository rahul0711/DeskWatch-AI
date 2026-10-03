import { useState } from 'react'
import { addCamera, testCameraConnection, AuthError } from './api'

const emptyTapo = {
  name: '',
  emailOrUser: '',
  password: '',
  ip: '',
  stream: '/stream1',
  transport: 'tcp',
}

const emptyGeneric = {
  name: '',
  ip: '',
  rtsp_port: '554',
  username: '',
  password: '',
  rtsp_path: '/Streaming/Channels/102',
  transport: 'tcp',
}

const GENERIC_PRESETS = [
  { key: 'sub', label: 'Sub-stream — lower res (/Streaming/Channels/102)', path: '/Streaming/Channels/102' },
  { key: 'main', label: 'Main stream — full res (/Streaming/Channels/101)', path: '/Streaming/Channels/101' },
  { key: 'tapo_hd', label: 'TP-Link Tapo HD (/stream1)', path: '/stream1' },
  { key: 'tapo_sd', label: 'TP-Link Tapo SD (/stream2)', path: '/stream2' },
  { key: 'custom', label: 'Custom path…', path: '' },
]

export default function AddCameraForm({ onAdded, onAuthError }) {
  const [open, setOpen] = useState(false)
  const [cameraType, setCameraType] = useState('tapo') // 'tapo' | 'generic'
  const [tapoForm, setTapoForm] = useState(emptyTapo)
  const [genericForm, setGenericForm] = useState(emptyGeneric)
  const [streamPreset, setStreamPreset] = useState('sub')
  const [testing, setTesting] = useState(false)
  const [testResult, setTestResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  function updateTapo(field) {
    return (e) => {
      setTestResult(null)
      setError(null)
      setTapoForm((f) => ({ ...f, [field]: e.target.value }))
    }
  }

  function updateGeneric(field) {
    return (e) => {
      setTestResult(null)
      setError(null)
      setGenericForm((f) => ({ ...f, [field]: e.target.value }))
    }
  }

  function handleGenericPresetChange(e) {
    const key = e.target.value
    setStreamPreset(key)
    const preset = GENERIC_PRESETS.find((p) => p.key === key)
    if (preset && preset.key !== 'custom') {
      setGenericForm((f) => ({ ...f, rtsp_path: preset.path }))
    }
    setTestResult(null)
  }

  function close() {
    setOpen(false)
    setTapoForm(emptyTapo)
    setGenericForm(emptyGeneric)
    setStreamPreset('sub')
    setTestResult(null)
    setError(null)
  }

  function getCameraPayload() {
    if (cameraType === 'tapo') {
      return {
        name: tapoForm.name.trim() || 'Tapo Camera',
        ip: tapoForm.ip.trim(),
        rtsp_port: 554,
        username: tapoForm.emailOrUser.trim(),
        password: tapoForm.password,
        rtsp_path: tapoForm.stream,
        transport: tapoForm.transport,
      }
    } else {
      return {
        name: genericForm.name.trim() || 'Camera',
        ip: genericForm.ip.trim(),
        rtsp_port: Number(genericForm.rtsp_port) || 554,
        username: genericForm.username.trim(),
        password: genericForm.password,
        rtsp_path: genericForm.rtsp_path.trim() || '/Streaming/Channels/102',
        transport: genericForm.transport,
      }
    }
  }

  async function handleTestConnection(e) {
    e.preventDefault()
    setTesting(true)
    setTestResult(null)
    setError(null)
    try {
      const payload = getCameraPayload()
      if (!payload.ip) {
        setError('Please enter a camera IP address first')
        return
      }
      const res = await testCameraConnection(payload)
      setTestResult(res)
    } catch (err) {
      if (err instanceof AuthError) onAuthError()
      else setTestResult({ ok: false, message: err.message })
    } finally {
      setTesting(false)
    }
  }

  async function handleSubmit(e) {
    e.preventDefault()
    setBusy(true)
    setError(null)
    try {
      const payload = getCameraPayload()
      if (!payload.ip) {
        setError('Please enter a camera IP address')
        return
      }
      await addCamera(payload)
      close()
      onAdded()
    } catch (err) {
      if (err instanceof AuthError) onAuthError()
      else setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  if (!open) {
    return (
      <button className="add-camera-btn" onClick={() => setOpen(true)}>
        + Add camera
      </button>
    )
  }

  return (
    <div className="add-camera-modal-backdrop" onClick={(e) => e.target === e.currentTarget && close()}>
      <div className="add-camera-modal" role="dialog" aria-modal="true">
        {/* Modal Header */}
        <div className="add-camera-modal__header">
          <div>
            <h2>Add New Camera</h2>
            <p className="add-camera-modal__subtitle">Connect a TP-Link Tapo camera or standard RTSP/ONVIF feed</p>
          </div>
          <button className="add-camera-modal__close" onClick={close} title="Close">
            ✕
          </button>
        </div>

        {/* Segmented Tab Switcher */}
        <div className="add-camera-tabs">
          <button
            type="button"
            className={`add-camera-tab${cameraType === 'tapo' ? ' add-camera-tab--active' : ''}`}
            onClick={() => { setCameraType('tapo'); setTestResult(null); setError(null) }}
          >
            <span className="add-camera-tab__icon">🌐</span>
            <span>TP-Link Tapo</span>
          </button>
          <button
            type="button"
            className={`add-camera-tab${cameraType === 'generic' ? ' add-camera-tab--active' : ''}`}
            onClick={() => { setCameraType('generic'); setTestResult(null); setError(null) }}
          >
            <span className="add-camera-tab__icon">📹</span>
            <span>Generic IP / ONVIF</span>
          </button>
        </div>

        <form onSubmit={handleSubmit} className="add-camera-form">
          {cameraType === 'tapo' ? (
            <>
              {/* Tapo Quick Setup Guide Card */}
              <div className="tapo-help-box">
                <div className="tapo-help-box__header">
                  <span className="tapo-help-box__bulb">💡</span>
                  <span className="tapo-help-box__title">Quick Tapo Setup Guide</span>
                </div>
                <div className="tapo-help-box__steps">
                  <div className="tapo-help-box__step">
                    <span className="tapo-help-box__badge">1</span>
                    <span>Make sure PC &amp; Camera are on the same Wi-Fi network.</span>
                  </div>
                  <div className="tapo-help-box__step">
                    <span className="tapo-help-box__badge">2</span>
                    <span>In <strong>Tapo App &gt; ⚙️ Settings &gt; Device Info</strong>, find the <strong>IP Address</strong>.</span>
                  </div>
                  <div className="tapo-help-box__step">
                    <span className="tapo-help-box__badge">3</span>
                    <span>Enter your <strong>TP-Link ID (Email) &amp; Password</strong> (or create a login in <em>Advanced Settings &gt; Camera Account</em>).</span>
                  </div>
                </div>
              </div>

              {/* Tapo Form Grid */}
              <div className="add-camera-form__grid">
                <div className="form-field">
                  <label htmlFor="tapo-name">
                    Camera Name
                  </label>
                  <input
                    id="tapo-name"
                    value={tapoForm.name}
                    onChange={updateTapo('name')}
                    placeholder="e.g. Tapo C200 Living Room"
                  />
                </div>

                <div className="form-field">
                  <label htmlFor="tapo-ip">
                    Camera IP Address <span className="req">*</span>
                  </label>
                  <input
                    id="tapo-ip"
                    value={tapoForm.ip}
                    onChange={updateTapo('ip')}
                    placeholder="e.g. 192.168.1.50"
                    required
                    autoFocus
                  />
                </div>

                <div className="form-field">
                  <label htmlFor="tapo-user">
                    TP-Link ID (Email) or Username
                  </label>
                  <input
                    id="tapo-user"
                    value={tapoForm.emailOrUser}
                    onChange={updateTapo('emailOrUser')}
                    placeholder="user@gmail.com or admin"
                    autoComplete="username"
                  />
                </div>

                <div className="form-field">
                  <label htmlFor="tapo-pass">
                    Password
                  </label>
                  <input
                    id="tapo-pass"
                    type="password"
                    value={tapoForm.password}
                    onChange={updateTapo('password')}
                    placeholder="Tapo password"
                    autoComplete="new-password"
                  />
                </div>

                <div className="form-field">
                  <label htmlFor="tapo-stream">
                    Stream Quality
                  </label>
                  <select id="tapo-stream" value={tapoForm.stream} onChange={updateTapo('stream')}>
                    <option value="/stream1">HD 1080p/2K (/stream1) — Recommended</option>
                    <option value="/stream2">SD 360p/720p (/stream2)</option>
                  </select>
                </div>

                <div className="form-field">
                  <label htmlFor="tapo-transport">
                    Transport
                  </label>
                  <select id="tapo-transport" value={tapoForm.transport} onChange={updateTapo('transport')}>
                    <option value="tcp">TCP (Recommended)</option>
                    <option value="udp">UDP</option>
                  </select>
                </div>
              </div>
            </>
          ) : (
            <div className="add-camera-form__grid">
              <div className="form-field">
                <label htmlFor="gen-name">
                  Camera Name
                </label>
                <input
                  id="gen-name"
                  value={genericForm.name}
                  onChange={updateGeneric('name')}
                  placeholder="e.g. Office Hikvision"
                />
              </div>

              <div className="form-field">
                <label htmlFor="gen-ip">
                  IP Address <span className="req">*</span>
                </label>
                <input
                  id="gen-ip"
                  value={genericForm.ip}
                  onChange={updateGeneric('ip')}
                  placeholder="e.g. 192.168.1.64"
                  required
                  autoFocus
                />
              </div>

              <div className="form-field">
                <label htmlFor="gen-port">
                  RTSP Port
                </label>
                <input
                  id="gen-port"
                  value={genericForm.rtsp_port}
                  onChange={updateGeneric('rtsp_port')}
                  placeholder="554"
                />
              </div>

              <div className="form-field">
                <label htmlFor="gen-user">
                  Username
                </label>
                <input
                  id="gen-user"
                  value={genericForm.username}
                  onChange={updateGeneric('username')}
                  placeholder="admin"
                  autoComplete="username"
                />
              </div>

              <div className="form-field">
                <label htmlFor="gen-pass">
                  Password
                </label>
                <input
                  id="gen-pass"
                  type="password"
                  value={genericForm.password}
                  onChange={updateGeneric('password')}
                  autoComplete="new-password"
                />
              </div>

              <div className="form-field">
                <label htmlFor="gen-preset">
                  Stream Preset
                </label>
                <select id="gen-preset" value={streamPreset} onChange={handleGenericPresetChange}>
                  {GENERIC_PRESETS.map((p) => (
                    <option key={p.key} value={p.key}>{p.label}</option>
                  ))}
                </select>
              </div>

              {streamPreset === 'custom' && (
                <div className="form-field form-field--full">
                  <label htmlFor="gen-path">
                    Custom RTSP Path
                  </label>
                  <input
                    id="gen-path"
                    value={genericForm.rtsp_path}
                    onChange={updateGeneric('rtsp_path')}
                    placeholder="/Streaming/Channels/102"
                  />
                </div>
              )}

              <div className="form-field">
                <label htmlFor="gen-transport">
                  Transport
                </label>
                <select id="gen-transport" value={genericForm.transport} onChange={updateGeneric('transport')}>
                  <option value="tcp">TCP</option>
                  <option value="udp">UDP</option>
                </select>
              </div>
            </div>
          )}

          {/* Test Connection Live Result */}
          {testResult && (
            <div className={`connection-test-result ${testResult.ok ? 'connection-test-result--ok' : 'connection-test-result--fail'}`}>
              <span className="connection-test-result__icon">{testResult.ok ? '✓' : '⚠'}</span>
              <span>{testResult.message}</span>
            </div>
          )}

          {error && <p className="add-camera-form__error">⚠ {error}</p>}

          {/* Modal Actions */}
          <div className="add-camera-form__actions">
            <button
              type="button"
              className="add-camera-form__test-btn"
              onClick={handleTestConnection}
              disabled={testing || busy}
            >
              {testing ? '⏳ Testing…' : '🧪 Test Connection'}
            </button>
            <div className="add-camera-form__actions-spacer" />
            <button
              type="button"
              className="add-camera-form__cancel"
              onClick={close}
              disabled={busy || testing}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="add-camera-form__submit"
              disabled={busy || testing || (cameraType === 'tapo' ? !tapoForm.ip.trim() : !genericForm.ip.trim())}
            >
              {busy ? 'Adding…' : 'Add Camera'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
