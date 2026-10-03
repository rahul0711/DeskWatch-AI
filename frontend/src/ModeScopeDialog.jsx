import { useState } from 'react'

// Lets the user pick which camera(s) a display-only mode (face/body/activity/ocr
// count) should switch to, instead of always applying to just the tile that was
// clicked. Face Attendance skips this dialog entirely -- it drives real
// recognition + logging per camera, so it stays a direct single-camera switch.
export default function ModeScopeDialog({ modeLabel, cameras, defaultCameraId, onCancel, onApply }) {
  const [selected, setSelected] = useState(() => new Set([defaultCameraId]))

  function toggle(id) {
    setSelected((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  return (
    <div className="mode-scope-modal-backdrop" onClick={(e) => e.target === e.currentTarget && onCancel()}>
      <div className="mode-scope-modal" role="dialog" aria-modal="true">
        <div className="mode-scope-modal__header">
          <div>
            <h2>Turn on {modeLabel}</h2>
            <p className="mode-scope-modal__subtitle">Choose one camera, or check several to turn it on for all of them at once. Other modes already running on a camera stay on.</p>
          </div>
          <button className="mode-scope-modal__close" onClick={onCancel} title="Close">
            ✕
          </button>
        </div>

        <div className="mode-scope-modal__list">
          {cameras.map((cam) => (
            <label key={cam.id} className="mode-scope-modal__option">
              <input
                type="checkbox"
                checked={selected.has(cam.id)}
                onChange={() => toggle(cam.id)}
              />
              <span>{cam.name}</span>
            </label>
          ))}
        </div>

        <div className="mode-scope-modal__actions">
          <button type="button" className="add-camera-form__cancel" onClick={onCancel}>
            Cancel
          </button>
          <button
            type="button"
            className="add-camera-form__submit"
            disabled={selected.size === 0}
            onClick={() => onApply(Array.from(selected))}
          >
            Turn on{selected.size > 1 ? ` for ${selected.size} cameras` : ''}
          </button>
        </div>
      </div>
    </div>
  )
}
