# Prompt: ASP.NET Core 8 MVC backend for the CCTV Face AI MySQL database

Copy everything below the line into your AI coding assistant (or hand it to a developer).

---

## Your task

Build a **new backend project in C# on .NET 8 (ASP.NET Core MVC, controllers)** that serves the data in an existing **MySQL** database to an existing React frontend. Use **ADO.NET with the `MySqlConnector` NuGet package only**. Do **not** use Entity Framework (no EF Core, no migrations, no DbContext) and do not use any other ORM. Write all SQL by hand, always with parameters.

The database is already created and filled by a separate Python system. **Do not create, alter or drop any table.** The schema is in `query.sql` at the root of the existing project. Read it before you write any SQL.

## The system you are joining

| Part | Tech | Port | Role |
|---|---|---|---|
| Camera / activity server | Python FastAPI | 8000 | Reads RTSP cameras, detects people, tracks activity, **writes** `activity_sessions` |
| Attendance server | Python FastAPI | 8003 | Registers people (needs the face model), **writes** `users`, `face_embeddings`, `attendance_events`; serves photos under `/data/...` |
| Frontend | React 19 + Vite dev server | 5174 | Proxies `/api/...` to the Python servers |
| **Your backend** | **ASP.NET Core 8** | **5080** | **Reports and admin over MySQL** |

Project root on the server: `/home/aiserver/Desktop/Facenet/CCTV_FOR_MOVEMENTS`. Create your solution next to it, in `CctvReports.Api/` (or a sibling folder). Do not modify the Python code.

### Who writes what (follow this strictly)

| Table | Python writes | Your backend may |
|---|---|---|
| `users` | create, rename, delete (Python must also rebuild its in-memory face index) | **read only** |
| `face_embeddings` | yes | **read only, and never return the `embedding` column** |
| `attendance_events` | yes | read only |
| `activity_sessions` | yes, continuously | read only |
| `activity_events` | no longer (old history) | read only |
| `camera_restrictions` | yes (Register page) | **read and replace** (the Python servers reload it every 5 seconds, so this is safe) |
| `cameras`, `activity_types` | no (reference data) | read; optionally update `cameras.name`, `cameras.is_active` |
| Views `v_*` | — | read only. Prefer them for reports. |

Registering people or adding face photos needs the Python face model. Never insert into `users` or `face_embeddings` from .NET.

## Database facts you must handle correctly

- **Times:** every `DATETIME(6)` column holds **UTC**, stored without a time zone. When you read one, call `DateTime.SpecifyKind(value, DateTimeKind.Utc)` and serialise it as ISO 8601 with `Z`. Columns ending in `_ist` (in the views) are **India time (UTC+05:30)**. Return them as local times with offset `+05:30`.
- **"Day" means India day.** The views group by `day_ist`. A `date=YYYY-MM-DD` query parameter means an India calendar day.
- **Durations** are seconds (`DOUBLE`). Return them as numbers, rounded to 1 decimal place.
- **Two different user ids:**
  - `users.id` is an integer database id, used by `face_embeddings`, `attendance_events` and `camera_restrictions`.
  - `users.employee_id` is a text employee code. `activity_sessions.user_id` stores this **employee code**, not `users.id`. Join on `users.employee_id = activity_sessions.user_id`.
  - In `activity_sessions`, `user_id` is NULL for people who were never identified. Report them as "Unknown".
- **Activities overlap on purpose.** Someone can be SITTING, MONITOR_INTERACTION and STATIONARY at the same time, so per-activity totals can add up to more than `PRESENCE`. Never subtract one from another.
- **Sessions:** one row per finished session. A person's total for an activity is `SUM(duration)`. Two phone sessions of 25s and 20s are two rows, and the total is 45s.
- **Activity codes:** SITTING, STANDING, WALKING, STATIONARY, PHONE_INTERACTION, MONITOR_INTERACTION, COMPUTER_INTERACTION, PRESENCE. Validate `activityType` parameters against the `activity_types` table.
- **Camera ids:** `camera-01`, `inside`, `corridor` (shown in the UI as Camera 1, 2, 3; see `cameras.camera_no`). Validate them against the `cameras` table.
- **Photo URLs:** `face_embeddings.thumbnail_path` / `original_path` are absolute paths on disk under `<project>/data/`. Turn them into the URL the frontend already uses: `/data/` + the path relative to `<project>/data`, with forward slashes. Example: `/home/.../CCTV_FOR_MOVEMENTS/data/enrollment/3/a.jpg` becomes `/data/enrollment/3/a.jpg`. The Python server on 8003 serves those files. Make the data root a config setting. A person's photo is their first face row (lowest `id`) with an `original_path`, otherwise the first `thumbnail_path`.

## Architecture (MVC, layered, no ORM)

```
CctvReports.Api/
  Program.cs                      DI, JSON options, CORS, auth, ProblemDetails, Swagger (Development only)
  appsettings.json                ConnectionStrings:Cctv, Auth:Token, Data:Root, Data:UrlPrefix
  Controllers/                    thin: validate input, call a service, return DTOs
  Services/                       business rules (date ranges, validation, pivots)
  Repositories/                   all SQL lives here; one class per area; interfaces for testing
  Models/Entities/                plain C# classes mirroring rows
  Models/Dtos/                    request/response shapes
  Infrastructure/
    IDbConnectionFactory.cs       creates an open MySqlConnection from configuration
    DataReaderExtensions.cs       null-safe readers, UTC handling
    TokenAuthMiddleware.cs        bearer-token check
CctvReports.Tests/                xUnit: services with fake repositories
```

Rules:
- `async` all the way (`OpenAsync`, `ExecuteReaderAsync`), and pass the `CancellationToken` from controllers down to every database call.
- Open one connection per request operation and dispose it with `await using`. Rely on MySqlConnector's built-in connection pooling.
- **Parameters only** (`@name`). Never build SQL by concatenating user input. For sort columns, use a whitelist that maps allowed names to columns.
- Wrap `camera_restrictions` replacement in a transaction: `DELETE` this user's rows, then `INSERT` the new set.
- Paginate every list. Defaults: `page=1`, `pageSize=50`, maximum 500. Return `{ items, page, page_size, total }`.
- Return errors as RFC 7807 ProblemDetails: 400 for invalid input, 404 for an unknown user or camera, 503 when the database is unreachable. Never put the connection string or password in a response or a log line.
- Connection string comes from configuration or the environment variable `ConnectionStrings__Cctv`. Example: `Server=127.0.0.1;Port=3306;Database=attendance;User ID=attendance;Password=...;SslMode=None;AllowPublicKeyRetrieval=True;`
- JSON: use `JsonNamingPolicy.SnakeCaseLower` for property names, so responses match the snake_case fields the frontend already reads from the Python servers.

## Authentication

The frontend already sends `Authorization: Bearer <token>` with the shared demo token `cctv-demo-token`. Accept the same token, configurable via `Auth:Token`. Also accept `?token=` as a query parameter, matching the Python servers. Missing or wrong token: 401. `/api/db/health` needs no token.

## Endpoints (all under `/api/db`, so they never clash with the Python routes)

| Method | Path | Returns |
|---|---|---|
| GET | `/api/db/health` | `{ status: "ok", database: "up" }` after running `SELECT 1` |
| GET | `/api/db/cameras` | Rows from `cameras` ordered by `camera_no` |
| GET | `/api/db/activity-types` | Rows from `activity_types` ordered by `sort_order` |
| GET | `/api/db/users?search=&page=&pageSize=` | `id, employee_id, name, created_at, face_count, photo_url, blocked_cameras[]`. `search` matches name or employee_id with `LIKE` |
| GET | `/api/db/users/{employeeId}` | One user plus `faces[] { id, thumbnail_url, face_width_px, det_score, created_at }`. **No embeddings** |
| GET | `/api/db/users/{id}/camera-access` | `{ user_id, allowed_cameras[], blocked_cameras[] }`. Allowed = all cameras minus blocked |
| PUT | `/api/db/users/{id}/camera-access` | Body `{ "blocked_cameras": ["corridor"] }`. Validate ids against `cameras`, replace in a transaction, return the new state |
| GET | `/api/db/activity/daily?date=&cameraId=` | One row per person for that India day. From `v_user_activity_daily` when `cameraId` is given, otherwise `v_user_activity_daily_all`. Fields: `employee_id, person_name, sitting_seconds, standing_seconds, walking_seconds, low_movement_seconds, phone_seconds, monitor_seconds, computer_seconds, present_seconds, phone_sessions, first_seen_ist, last_seen_ist`. Default date: today (IST) |
| GET | `/api/db/activity/users/{employeeId}/daily?from=&to=` | Same fields, one row per day in the range (maximum 93 days) |
| GET | `/api/db/activity/users/{employeeId}/totals?from=&to=` | Per activity type: `sessions, total_seconds, avg_session_seconds, longest_session_seconds` |
| GET | `/api/db/activity/sessions?employeeId=&cameraId=&activityType=&from=&to=&page=&pageSize=` | Paged rows from `v_activity_sessions_named`, newest first |
| GET | `/api/db/activity/leaderboard?activityType=PHONE_INTERACTION&from=&to=&top=10` | People ranked by total seconds of that activity |
| GET | `/api/db/attendance?date=` | From `v_attendance_daily`: `employee_id, person_name, first_punch_ist, last_punch_ist, punches` |
| GET | `/api/db/attendance/events?date=&employeeId=&page=&pageSize=` | Raw punches joined to `users`. Include `thumbnail_url` (same path-to-URL rule) |

`from` / `to` are India dates (`YYYY-MM-DD`), inclusive. Reject `from > to` with 400. When filtering raw UTC columns instead of a view's `day_ist`, convert the India day to a UTC range: `[day 00:00 IST, next day 00:00 IST)` = `[day − 5:30, next day − 5:30)` UTC, so indexes on `start_time` / `timestamp` are used.

## Wiring it to the existing frontend

1. CORS: allow `http://localhost:5174` and `http://172.29.7.22:5174`.
2. Listen on `http://0.0.0.0:5080` (Kestrel `Urls` in `appsettings.json`).
3. Tell the user to add this rule to `frontend/vite.config.js` (`server.proxy`), **above** the generic `'/api'` entry:
   ```js
   '/api/db': { target: process.env.DB_API_TARGET || 'http://127.0.0.1:5080', changeOrigin: true },
   ```
   The frontend then calls `/api/db/...` with the token it already has.

## Quality bar

- Builds with `dotnet build` with no warnings. Nullable reference types enabled.
- xUnit tests for: India-day to UTC range conversion; the photo path to URL rule; camera-access validation (unknown camera gives 400); pagination limits; leaderboard ordering. Use fake repositories, no database needed.
- A `README.md` with: prerequisites (.NET 8 SDK, MySQL with `query.sql` applied), the configuration keys, `dotnet run` and `dotnet test`, and one `curl` example per endpoint using `-H "Authorization: Bearer cctv-demo-token"`.
- Logging with `ILogger`: one line per request (method, path, status, ms), without tokens or passwords.

## Done when

- `curl http://127.0.0.1:5080/api/db/health` returns `{"status":"ok","database":"up"}`.
- `GET /api/db/activity/daily` returns today's per-person seconds, and they match `SELECT * FROM v_user_activity_daily_all WHERE day_ist = <today IST>` run in MySQL.
- `PUT /api/db/users/{id}/camera-access` with `["corridor"]` changes what the Register page shows within 5 seconds of a refresh.
- No response anywhere contains an `embedding` value, a password or the connection string.
