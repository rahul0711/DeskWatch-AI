# Prompt: .NET 10 backend with Super Admin / Admin / Viewer roles for the CCTV Face AI MySQL database

Open the project folder `/home/aiserver/Desktop/Facenet/CCTV_FOR_MOVEMENTS` in Antigravity, then paste everything below the line into the agent.

**Before pasting:** check the "Who can do what" table and the **Roles** column of the endpoint tables. That is where you decide what each role may do. Change any ✅/❌ or role list you want.

---

## Your task

Build a **new backend in C# on .NET 10 (ASP.NET Core, controllers)** in a new folder `CctvAdmin.Api/` inside this project. It provides **login, role-based access control (Super Admin / Admin / Viewer)** and **reports** over an existing **MySQL 8.4** database.

- Use **ADO.NET with the `MySqlConnector` NuGet package only**. No Entity Framework, no Dapper, no other ORM. Write all SQL by hand, always with `@parameters`.
- Use `BCrypt.Net-Next` for password hashing and `Microsoft.AspNetCore.Authentication.JwtBearer` for tokens.
- **Read `query.sql` in the project root before writing any code.** It is the complete schema: tables 1–13 and all `v_*` views. Its comments explain every column.
- **Do not create, alter or drop any table or view.** The schema is applied by running `mysql -u root -p < query.sql`. If you think the schema needs a change, stop and tell me instead.
- **Do not modify any Python file, `run_all.sh` or `.env`.** Frontend changes are limited to what is listed under "Frontend integration".

## The system you are joining

| Part | Tech | Port | Role |
|---|---|---|---|
| Camera / activity server | Python FastAPI (`app/web/server.py`) | 8000 | Reads RTSP cameras, tracks people, **writes** `activity_sessions` |
| Attendance server | Python FastAPI (`app/web/attendance_server.py`) | 8003 | Registers people (needs the face model), **writes** `users`, `face_embeddings`, `attendance_events`, `camera_restrictions`; serves photos under `/data/...` |
| Frontend | React 19 + Vite (`frontend/`) | 5173 | Proxies `/api/...` to the Python servers |
| **Your backend** | **ASP.NET Core / .NET 10** | **5080** | **Logins, roles, camera scope, audit, reports** |

The Python servers currently use one hardcoded shared token (`cctv-demo-token`, see `app/web/auth.py`). They keep working unchanged. Your backend is the new source of truth for **who can log in and what they may do**.

## Two kinds of "user" — never mix them up

| Table | Who | Purpose |
|---|---|---|
| `users` (+ `face_embeddings`, `camera_restrictions`) | **Tracked people / employees** whose faces the cameras recognise | `camera_restrictions` = cameras where this **person** must not be seen. The Python server raises a "not welcome" alert if they appear there |
| `app_users` (+ `app_user_cameras`, `app_user_sessions`) | **Dashboard logins**: SUPER_ADMIN, ADMIN, VIEWER | `app_user_cameras` = cameras this **login** may see and manage |

In API paths, use `persons` for tracked people and `logins` for dashboard accounts, so the two are never confused.

## Roles and the rules you must enforce

The database only stores **which role** each login has (`app_users.role_code` → `roles`: SUPER_ADMIN, ADMIN, VIEWER; `rank_level` 100 / 50 / 10). There are **no permission tables**. **What each role may do is decided in the MVC backend code**, using ASP.NET Core's built-in role authorization:

- Put the role names in one place: `Security/Roles.cs` with `const string SuperAdmin = "SUPER_ADMIN"`, `Admin = "ADMIN"`, `Viewer = "VIEWER"`, plus combined strings such as `AdminOrAbove = "SUPER_ADMIN,ADMIN"` and `Anyone = "SUPER_ADMIN,ADMIN,VIEWER"`.
- Put the role in the JWT as the standard role claim, so `[Authorize(Roles = Roles.AdminOrAbove)]` works on controllers and actions.
- Every controller action carries an explicit `[Authorize(Roles = ...)]` that matches the **Roles** column of the endpoint tables below. No action relies on a default.

Who can do what:

| Action | SUPER_ADMIN | ADMIN | VIEWER |
|---|---|---|---|
| Live view, reports, list people | ✅ | ✅ | ✅ |
| Block / allow a person on cameras | ✅ | ✅ | ❌ |
| Create, edit, disable VIEWER logins, assign their cameras | ✅ | ✅ | ❌ |
| Create, edit, disable ADMIN and SUPER_ADMIN logins | ✅ | ❌ | ❌ |
| Rename cameras / mark them inactive | ✅ | ❌ | ❌ |
| Read the audit log | ✅ | ❌ | ❌ |

`/auth/me` returns the role, and the frontend shows or hides buttons from that same table. The backend still enforces everything.

On top of the role checks, enforce these rules in the **service layer** and cover each one with a unit test:

1. **Camera scope.** A login's cameras are the rows of `v_app_user_cameras` for its id. SUPER_ADMIN and `all_cameras = 1` get every active camera. Otherwise the login gets only its `app_user_cameras` rows, and no rows means **no cameras**. Every report, list and camera-access read is filtered to that scope. Asking for a camera outside the scope returns **403**, not an empty list.
2. **Person camera access is a merge, not a replace.** `PUT /persons/{id}/camera-access` may only change blocks on cameras **inside the caller's scope**. Blocks on cameras outside the caller's scope must be kept exactly as they are. Do the delete-and-insert in one transaction.
3. **Rank rule.** A login may only create, edit, disable, reset or delete logins of a **strictly lower** `rank_level`. A SUPER_ADMIN may also manage other SUPER_ADMINs. An ADMIN may only touch VIEWERs. A VIEWER manages nobody.
4. **No privilege escalation.**
   - Nobody can give a role higher than their own.
   - An ADMIN can only assign cameras that are inside their own scope, and can only set `all_cameras = 1` if they have it themselves.
5. **Never lock the system out.**
   - Reject disabling, deleting or demoting the **last active SUPER_ADMIN** (409).
   - A login cannot disable or delete itself, or change its own role.
6. **Disabled logins stop working at once.** On disable, password reset, role change or delete, set `revoked_at` on all of that login's `app_user_sessions` rows. Every authenticated request also re-checks `app_users.is_active`, so an existing access token for a disabled login gets 401. Cache this check for 30 seconds at most.
7. **Tracked-person writes stay in Python.** Registering people, adding face photos, renaming and deleting (`users`, `face_embeddings`) need the Python face model and the in-memory index. **Never insert, update or delete `users` or `face_embeddings` from .NET.** Your backend only exposes the caller's role (via `/auth/me`) so the frontend can show or hide those buttons. The one tracked-person table you may write is `camera_restrictions`; the Python servers reload it every 5 seconds.
8. **Audit everything.** Write one `audit_log` row for each of these:
   - login success, login failure, logout
   - every create, update or delete on logins and camera-access
   - every password change or reset

   Put before/after values in `details` (JSON), and copy `username` into the row. **Never** put a password, hash or token in `details` or in any log line.

## Authentication

- `POST /api/db/auth/login` `{ username, password }`:
  - Username is case-insensitive (the column collation already handles it).
  - On success, return a **JWT access token** (HS256, 15 minutes) and a **refresh token** (random 32 bytes, base64url, 7 days). Store only the hex SHA-256 of the refresh token in `app_user_sessions.token_hash`.
  - JWT claims: `sub` (app_users.id), `username`, `role`, `must_change_password`. Read the secret from config `Auth:JwtSecret` (at least 32 bytes) and fail at startup if it's missing.
  - Use HS256 with a shared secret on purpose: the Python servers can later verify the same token with the same secret.
  - **Lockout:** 5 failed attempts set `locked_until = now + 15 min`; success resets `failed_login_count`.
  - Unknown user, wrong password, disabled and locked all return the **same** 401 message, and take the same time (verify against a dummy hash when the user doesn't exist).
- `POST /api/db/auth/refresh` `{ refresh_token }`: rotates. The old row gets `revoked_at` and a new row is issued. Reusing a revoked token revokes **all** of that login's sessions.
- `POST /api/db/auth/logout`: revokes the current refresh token.
- `GET /api/db/auth/me` returns `{ id, username, full_name, role, must_change_password, cameras[{camera_id, name}] }`. The frontend builds its menus from this.
- `POST /api/db/auth/change-password` `{ current_password, new_password }`: minimum 10 characters, must not equal the username; clears `must_change_password`; revokes all other sessions.
- While `must_change_password = 1`, every endpoint except `me`, `change-password` and `logout` returns 403 with code `password_change_required`.
- **Bootstrap:** on startup, if there is **no active SUPER_ADMIN**, create one from config `Bootstrap:SuperAdminUsername` / `Bootstrap:SuperAdminPassword` with `must_change_password = 1` and `created_by = NULL`, then log a warning to remove the password from config. Never seed a default password in code.
- Read secrets from environment variables (`Auth__JwtSecret`, `Bootstrap__SuperAdminPassword`, `ConnectionStrings__Cctv`), never from a committed `appsettings.json`. Put placeholder values in `appsettings.Development.json.example` only.

## Endpoints (all under `/api/db`; the Roles column is the `[Authorize(Roles = ...)]` for that action)

**Health and reference**

| Method | Path | Roles | Returns |
|---|---|---|---|
| GET | `/health` | none (`[AllowAnonymous]`) | `{ status, database }` after `SELECT 1` |
| GET | `/cameras` | all three | Cameras in the caller's scope, ordered by `camera_no` |
| PATCH | `/cameras/{cameraId}` | SUPER_ADMIN | Update `name`, `is_active` |
| GET | `/activity-types` | all three | `activity_types` ordered by `sort_order` |

**Logins (dashboard accounts)**

| Method | Path | Roles | Notes |
|---|---|---|---|
| GET | `/logins?search=&role=&active=&page=&pageSize=` | SUPER_ADMIN, ADMIN | Lists only logins the caller may manage (rank rule), plus their cameras. Never return `password_hash` |
| POST | `/logins` | SUPER_ADMIN, ADMIN + rank rule | `{ username, full_name, email?, role, all_cameras, camera_ids[], temporary_password }`; sets `must_change_password = 1`, `created_by = caller` |
| PATCH | `/logins/{id}` | SUPER_ADMIN, ADMIN + rank rule | `full_name, email, role, is_active, all_cameras` |
| PUT | `/logins/{id}/cameras` | SUPER_ADMIN, ADMIN + rank rule | `{ camera_ids[] }`, replace in a transaction; each id must be in the caller's scope |
| POST | `/logins/{id}/reset-password` | SUPER_ADMIN, ADMIN + rank rule | `{ temporary_password }`; sets `must_change_password = 1`, revokes sessions, unlocks |
| DELETE | `/logins/{id}` | SUPER_ADMIN, ADMIN + rank rule | Hard delete; audit rows remain |

**Roles**

| Method | Path | Roles | Notes |
|---|---|---|---|
| GET | `/roles` | SUPER_ADMIN, ADMIN | Rows of `roles`, limited to roles the caller may assign (rank rule) |

**Tracked people (read, plus camera blocking)**

| Method | Path | Roles | Notes |
|---|---|---|---|
| GET | `/persons?search=&page=&pageSize=` | all three | `id, employee_id, name, created_at, face_count, photo_url, blocked_cameras[]`; `blocked_cameras` is limited to the caller's scope |
| GET | `/persons/{id}` | all three | Plus `faces[] { id, thumbnail_url, face_width_px, det_score, created_at }`. **Never return `embedding`** |
| GET | `/persons/{id}/camera-access` | all three | From `v_person_camera_access`, limited to scope: `[{ camera_id, camera_name, access: ALLOWED/BLOCKED }]` |
| PUT | `/persons/{id}/camera-access` | SUPER_ADMIN, ADMIN | `{ blocked_cameras[] }`; merge rule 2 applies; unknown camera returns 400, camera outside scope returns 403 |

**Reports (all filtered by camera scope)**

| Method | Path | Roles | Source |
|---|---|---|---|
| GET | `/activity/daily?date=&cameraId=` | all three | With `cameraId`: `v_user_activity_daily`. Without it: aggregate `v_user_activity_daily` over the caller's cameras only. Use `v_user_activity_daily_all` only when the caller has every camera |
| GET | `/activity/persons/{employeeId}/daily?from=&to=` | all three | Max 93 days |
| GET | `/activity/persons/{employeeId}/totals?from=&to=` | all three | Per activity: `sessions, total_seconds, avg_session_seconds, longest_session_seconds` |
| GET | `/activity/sessions?employeeId=&cameraId=&activityType=&from=&to=&page=&pageSize=` | all three | `v_activity_sessions_named`, newest first |
| GET | `/activity/leaderboard?activityType=&from=&to=&top=10` | all three | People ranked by total seconds |
| GET | `/attendance?date=` | all three | Raw `attendance_events` joined to `users`, grouped by India day. Filter by camera scope via `cameras.name = attendance_events.camera_name` |
| GET | `/audit?loginId=&action=&from=&to=&page=&pageSize=` | SUPER_ADMIN | Newest first |

## Database facts you must handle correctly

- **Times:** every `DATETIME(6)` column is **UTC** without a zone. Read it with `DateTime.SpecifyKind(v, DateTimeKind.Utc)` and serialise as ISO 8601 with `Z`. Write it as `DateTime.UtcNow`. Columns ending in `_ist` are India time (UTC+05:30); return them with offset `+05:30`.
- **"Day" means India day.** `date`, `from` and `to` are India dates (`YYYY-MM-DD`), and `from`/`to` are inclusive. Reject `from > to` with 400. When filtering raw UTC columns, convert an India day to the UTC range `[day − 5:30, next day − 5:30)` so indexes are used.
- **Two person ids:** `users.id` (int) is used by `face_embeddings`, `attendance_events` and `camera_restrictions`. `users.employee_id` (text) is what `activity_sessions.user_id` stores; join on `users.employee_id = activity_sessions.user_id`. A NULL `activity_sessions.user_id` means an unidentified person; report them as "Unknown".
- **Activities overlap on purpose** (SITTING + MONITOR_INTERACTION + STATIONARY at once). Never subtract one from another. A total is `SUM(duration)`. Durations are seconds; round to 1 decimal place.
- **Camera ids:** `camera-01`, `inside`, `corridor`. Validate every camera id against the `cameras` table.
- **Photo URLs:** `thumbnail_path` / `original_path` are absolute paths under `<project>/data/`. The URL is `/data/` + the path relative to that folder, with forward slashes; the Python server on 8003 serves them. Make the data root a config setting (`Data:Root`).

## Architecture

```
CctvAdmin.Api/
  Program.cs                     DI, JSON snake_case (JsonNamingPolicy.SnakeCaseLower), CORS, JWT auth, ProblemDetails, OpenAPI (Development only)
  Controllers/                   thin: bind + validate, call a service, return DTOs
  Services/                      every rule above lives here (AuthService, LoginService, PersonAccessService, ReportService, AuditService)
  Security/
    Roles.cs                            the three role-name constants and combined role strings
    CurrentUser.cs                      id, role, rank, camera scope for the request (loaded once per request)
    PasswordHasher.cs, TokenService.cs
  Repositories/                  all SQL here, behind interfaces, one class per area
  Models/Entities, Models/Dtos
  Infrastructure/                IDbConnectionFactory (open MySqlConnection), DataReader helpers (null + UTC)
CctvAdmin.Tests/                 xUnit, fake repositories, no database needed
```

- `async` throughout; pass the `CancellationToken` into every database call. Use one connection per operation with `await using`, and rely on MySqlConnector pooling.
- Multi-statement writes go in a transaction.
- Paginate every list: default `page=1`, `pageSize=50`, max 500. Return `{ items, page, page_size, total }`. Sort only through a whitelist of columns.
- Errors are RFC 7807 ProblemDetails with a machine-readable `code`:
  - 400 invalid input
  - 401 not logged in
  - 403 wrong role or out of scope (code `forbidden`, `out_of_scope` or `password_change_required`)
  - 404 not found
  - 409 rule conflict (e.g. last super admin)
  - 503 database unreachable

  Never leak the connection string, SQL or a stack trace.
- Rate-limit `/auth/login` to 10 requests per minute per IP using the built-in `Microsoft.AspNetCore.RateLimiting`.
- Listen on `http://0.0.0.0:5080`. CORS: allow `http://localhost:5173` and `http://172.29.7.22:5173`.

## Frontend integration (the only frontend changes allowed)

1. In `frontend/vite.config.js`, add this **above** the generic `'/api'` proxy entry:
   ```js
   '/api/db': { target: process.env.DB_API_TARGET || 'http://127.0.0.1:5080', changeOrigin: true },
   ```
2. Do **not** switch the existing login screen to the new backend yet. The live video and Python routes still need the demo token. Instead, add a new **"Admin"** page (`frontend/src/admin/`) that logs in against `/api/db/auth/login`, keeps its tokens in `sessionStorage` under its own key, refreshes them, and shows:
   - **Logins:** list, create, edit, disable, reset password, assign cameras
   - **People:** a camera-access grid with an Allowed/Blocked toggle per person per camera
   - **Audit log**

   Hide every button the logged-in role may not use, based on `role` from `/auth/me` and the "Who can do what" table. The backend still enforces everything.

## Quality bar

- `dotnet build` with **zero warnings**; nullable reference types enabled.
- xUnit tests, at minimum:
  - A test per role per endpoint group (allowed vs 403), every rule (1–8) above, including: admin cannot see or modify a camera outside scope; merge keeps out-of-scope blocks; admin cannot create an admin; cannot remove the last super admin; disabled login gets 401
  - Lockout after 5 failures
  - Refresh-token rotation and reuse detection
  - India-day to UTC range conversion
  - Photo path to URL
  - Pagination limits
- `CctvAdmin.Api/README.md` covering:
  - prerequisites (.NET 10 SDK, MySQL 8.4 with `query.sql` applied)
  - every config key and its environment variable
  - how to create the first super admin
  - `dotnet run` / `dotnet test`
  - one `curl` example per endpoint (log in first, then `-H "Authorization: Bearer $TOKEN"`)
- Request log with `ILogger`: method, path, status, ms, login id. **No** tokens, passwords or request bodies.

## Done when

1. `curl http://127.0.0.1:5080/api/db/health` returns `{"status":"ok","database":"up"}`.
2. Starting with an empty `app_users` and the bootstrap config creates one SUPER_ADMIN. Logging in forces a password change.
3. The super admin creates an ADMIN limited to `inside`:
   - That admin's `/cameras` and every report show only `inside`.
   - Blocking a person on `corridor` returns 403.
   - Blocking them on `inside` works and keeps their existing `corridor` block. Run `SELECT * FROM v_person_camera_access` to confirm.
4. That ADMIN can create a VIEWER but gets 403 creating an ADMIN. The VIEWER gets 403 on every write.
5. Disabling the ADMIN makes its still-valid access token return 401 within 30 seconds.
6. Every action above shows up in `GET /api/db/audit`.
7. No response or log line anywhere contains an `embedding`, `password_hash`, token, password or the connection string.
