import { useCallback, useEffect, useState } from 'react'
import { AuthError, clearToken, getToken, getUser, logout } from './api'
import { fetchMe } from './db/api'
import LoginPage from './LoginPage'
import ChangePasswordPage from './ChangePasswordPage'
import RegistrationPage from './attendance/RegistrationPage'
import Dashboard from './attendance/Dashboard'
import LiveCameras from './LiveCameras'
import ReportsPage from './reports/ReportsPage'
import AttendancePage from './reports/AttendancePage'
import ActivityPage, { ACTIVITY_PAGES } from './reports/ActivityPage'
import ViolationsPage from './reports/ViolationsPage'
import AdminPage from './admin/AdminPage'
import './App.css'

const ALL = ['SUPER_ADMIN', 'ADMIN', 'VIEWER']
const ADMINS = ['SUPER_ADMIN', 'ADMIN']

// Which roles see which page, grouped as in the admin sidebar, with what each
// page does. The backends enforce the same role rules on every call.
const GROUPS = [
  {
    title: 'Live',
    pages: [
      { key: 'cameras', label: 'Live Cameras', roles: ALL,
        desc: 'Watch the cameras. Choose Face or Body movement; picking Phone, Body, Monitor or Computer runs only that.' },
      { key: 'dashboard', label: 'Dashboard', roles: ALL,
        desc: 'Attendance punches as they happen, with voice, and unknown faces you can enrol.' },
    ],
  },
  {
    title: 'People',
    pages: [
      { key: 'attendance', label: 'Attendance', roles: ADMINS,
        desc: 'Who came and who did not on any day, with first-in and last-seen. Export to CSV.' },
      { key: 'register', label: 'Register', roles: ADMINS,
        desc: 'Enrol people with face photos and choose which cameras each person is allowed on.' },
    ],
  },
  {
    title: 'Body movement',
    pages: [
      { key: 'phone', label: 'Phone usage', roles: ADMINS, desc: ACTIVITY_PAGES.phone.intro },
      { key: 'posture', label: 'Sitting & standing', roles: ADMINS, desc: ACTIVITY_PAGES.posture.intro },
      { key: 'monitor', label: 'Monitor use', roles: ADMINS, desc: ACTIVITY_PAGES.monitor.intro },
      { key: 'computer', label: 'Computer use', roles: ADMINS, desc: ACTIVITY_PAGES.computer.intro },
      { key: 'reports', label: 'All activity', roles: ALL,
        desc: 'Every activity per person in one table for a day, with each session.' },
    ],
  },
  {
    title: 'Alerts',
    pages: [
      { key: 'violations', label: 'Violations', roles: ADMINS,
        desc: 'Every confirmed phone use, with who, when, where and for how long. Also shown live on Live Cameras.' },
    ],
  },
  {
    title: 'Settings',
    pages: [
      { key: 'admin', label: 'Admin', roles: ADMINS,
        desc: 'Logins and roles, which cameras run Face / Body movement, workstations (seat + screen zones), and the audit log.' },
    ],
  },
]
const TABS = GROUPS.flatMap((g) => g.pages)

export default function App() {
  const [user, setUser] = useState(() => (getToken() ? getUser() : null))
  const [tab, setTab] = useState('cameras')
  const [changingPassword, setChangingPassword] = useState(false)
  const [prefillImage, setPrefillImage] = useState(null)
  // Admin sidebar can be hidden to give the page the full width. Remembered per browser.
  const [sidebarOpen, setSidebarOpen] = useState(() => localStorage.getItem('sidebarOpen') !== 'false')

  useEffect(() => {
    localStorage.setItem('sidebarOpen', String(sidebarOpen))
  }, [sidebarOpen])

  const handleAuthError = useCallback(() => {
    clearToken()
    setUser(null)
  }, [])

  const handleLogout = useCallback(async () => {
    await logout().catch(() => {})
    setUser(null)
    setTab('cameras')
  }, [])

  // Role, cameras or "must change password" may have changed since this tab
  // logged in -- refresh them once.
  const loggedIn = Boolean(user)
  useEffect(() => {
    if (!loggedIn) return
    fetchMe()
      .then(setUser)
      .catch((err) => {
        if (err instanceof AuthError) handleAuthError()
      })
  }, [loggedIn, handleAuthError])

  const handleEnrollFromCrop = useCallback(async (thumbnailUrl) => {
    const res = await fetch(thumbnailUrl)
    const blob = await res.blob()
    setPrefillImage(blob)
    setTab('register')
  }, [])

  if (!user) {
    return <LoginPage onLoggedIn={setUser} />
  }

  if (user.must_change_password || changingPassword) {
    return (
      <ChangePasswordPage
        user={user}
        forced={user.must_change_password}
        onDone={(me) => {
          setUser(me)
          setChangingPassword(false)
        }}
        onCancel={user.must_change_password ? handleLogout : () => setChangingPassword(false)}
        onAuthError={handleAuthError}
      />
    )
  }

  const tabs = TABS.filter((t) => t.roles.includes(user.role))
  const current = tabs.some((t) => t.key === tab) ? tab : 'cameras'

  const page =
    current === 'cameras' ? (
      <LiveCameras onAuthError={handleAuthError} />
    ) : current === 'dashboard' ? (
      <Dashboard onAuthError={handleAuthError} onEnrollFromCrop={handleEnrollFromCrop} />
    ) : current === 'attendance' ? (
      <AttendancePage onAuthError={handleAuthError} />
    ) : current in ACTIVITY_PAGES ? (
      <ActivityPage key={current} page={current} onAuthError={handleAuthError} />
    ) : current === 'violations' ? (
      <ViolationsPage onAuthError={handleAuthError} />
    ) : current === 'reports' ? (
      <ReportsPage onAuthError={handleAuthError} />
    ) : current === 'admin' ? (
      <AdminPage onAuthError={handleAuthError} />
    ) : (
      <RegistrationPage
        onAuthError={handleAuthError}
        prefillImage={prefillImage}
        onPrefillConsumed={() => setPrefillImage(null)}
      />
    )

  // Admin and Super Admin: a sidebar that says what each page does.
  if (ADMINS.includes(user.role)) {
    return (
      <div className={`app app--sidebar${sidebarOpen ? '' : ' app--sidebar-closed'}`}>
        {sidebarOpen && (
          <aside className="app__sidebar">
            <h1>CCTV Attendance</h1>
            <nav>
              {GROUPS.map((g) => {
                const pages = g.pages.filter((p) => p.roles.includes(user.role))
                if (!pages.length) return null
                return (
                  <div key={g.title} className="app__nav-group">
                    <div className="app__nav-title">{g.title}</div>
                    {pages.map((p) => (
                      <button
                        key={p.key}
                        className={`app__nav-item${current === p.key ? ' app__nav-item--active' : ''}`}
                        onClick={() => setTab(p.key)}
                      >
                        <span className="app__nav-label">{p.label}</span>
                        <span className="app__nav-desc">{p.desc}</span>
                      </button>
                    ))}
                  </div>
                )
              })}
            </nav>
            <div className="app__sidebar-user">
              <span title={user.email || user.username}>
                {user.full_name} · {user.role_name}
              </span>
              <div>
                <button className="app__logout" onClick={() => setChangingPassword(true)}>
                  Change password
                </button>
                <button className="app__logout" onClick={handleLogout}>
                  Log out
                </button>
              </div>
            </div>
          </aside>
        )}
        <main className="app__main">
          <button
            className="app__sidebar-toggle"
            onClick={() => setSidebarOpen((open) => !open)}
            aria-expanded={sidebarOpen}
            title={sidebarOpen ? 'Hide the menu for more room' : 'Show the menu'}
          >
            {sidebarOpen ? '« Hide menu' : '» Show menu'}
          </button>
          {page}
        </main>
      </div>
    )
  }

  return (
    <div className="app">
      <header className="app__header">
        <h1>CCTV Attendance</h1>
        <nav className="app__tabs">
          {tabs.map((t) => (
            <button
              key={t.key}
              className={`app__tab${current === t.key ? ' app__tab--active' : ''}`}
              onClick={() => setTab(t.key)}
            >
              {t.label}
            </button>
          ))}
        </nav>
        <span className="app__user" title={user.email || user.username}>
          {user.full_name} · {user.role_name}
        </span>
        <button className="app__logout" onClick={() => setChangingPassword(true)}>
          Change password
        </button>
        <button className="app__logout" onClick={handleLogout}>
          Log out
        </button>
      </header>

      {page}
    </div>
  )
}
