using System.Globalization;
using CCTV_Backend_Project.Dtos;
using CCTV_Backend_Project.Infrastructure;
using CCTV_Backend_Project.Security;

namespace CCTV_Backend_Project.Services
{
    /// <summary>Attendance and activity reports. Everything is limited to the
    /// caller's cameras.</summary>
    public class ReportService
    {
        private readonly Db _db;
        private readonly CameraService _cameras;
        private readonly double _punchCooldownSeconds;

        public ReportService(Db db, CameraService cameras, IConfiguration configuration)
        {
            _db = db;
            _cameras = cameras;
            // Same default as the Python PUNCH_COOLDOWN_SECONDS.
            _punchCooldownSeconds = configuration.GetValue("Attendance:PunchCooldownSeconds", 600.0);
        }

        /// <summary>Same rows and shape as the Python GET /api/attendance: `date` is a
        /// UTC calendar day (what the Dashboard sends), oldest first, and repeat
        /// punches of one person within the cooldown are collapsed. Punches from
        /// a camera outside the caller's scope are left out; mobile punches are
        /// not tied to a camera and are always shown.</summary>
        public async Task<List<AttendanceDto>> AttendanceAsync(CurrentUser user, string? date, CancellationToken ct)
        {
            DateTime? from = null;
            if (!string.IsNullOrWhiteSpace(date))
            {
                if (!DateOnly.TryParseExact(date, "yyyy-MM-dd", CultureInfo.InvariantCulture, DateTimeStyles.None, out var d))
                    throw ApiException.BadRequest("date must be YYYY-MM-DD");
                from = d.ToDateTime(TimeOnly.MinValue, DateTimeKind.Utc);
            }

            await using var conn = await _db.OpenAsync(ct);
            var hidden = await HiddenCameraNamesAsync(conn, user, ct);

            await using var cmd = conn.Command(@"
                SELECT a.id, a.user_id, u.name, u.employee_id, a.camera_name, a.timestamp, a.confidence,
                       a.face_width_px, a.low_confidence, a.thumbnail_path
                FROM attendance_events a JOIN users u ON u.id = a.user_id" +
                (from is null ? "" : " WHERE a.timestamp >= @from AND a.timestamp < @to") +
                " ORDER BY a.timestamp, a.id")
                .With("@from", from).With("@to", from?.AddDays(1));
            await using var r = await cmd.ExecuteReaderAsync(ct);

            var list = new List<AttendanceDto>();
            var lastSeen = new Dictionary<int, DateTime>();
            while (await r.ReadAsync(ct))
            {
                var uid = r.GetInt("user_id");
                var ts = r.GetUtc("timestamp");
                if (lastSeen.TryGetValue(uid, out var prev) && (ts - prev).TotalSeconds < _punchCooldownSeconds) continue;
                lastSeen[uid] = ts;

                var cameraName = r.GetString(r.GetOrdinal("camera_name"));
                if (hidden.Contains(cameraName)) continue;
                list.Add(new AttendanceDto(
                    r.GetInt("id"), uid, r.GetString(r.GetOrdinal("name")), r.GetString(r.GetOrdinal("employee_id")),
                    cameraName, ts, r.GetDouble("confidence"), r.GetInt("face_width_px"), r.GetBool("low_confidence"),
                    PersonService.PhotoUrl(r.GetString(r.GetOrdinal("thumbnail_path")))));
            }
            return list;
        }

        /// <summary>Every enrolled person for an India day, present or absent: first
        /// and last punch, punch count (repeats within the cooldown collapsed) and
        /// the cameras that saw them. Punches from cameras outside the caller's
        /// scope are ignored, so a person seen only there shows as absent.</summary>
        public async Task<List<AttendanceDailyDto>> AttendanceDailyAsync(CurrentUser user, string? date, CancellationToken ct)
        {
            var (fromUtc, toUtc) = TimeIst.DayToUtcRange(TimeIst.ParseOrToday(date));
            await using var conn = await _db.OpenAsync(ct);
            var hidden = await HiddenCameraNamesAsync(conn, user, ct);

            var people = new List<(int Id, string EmployeeId, string Name)>();
            await using (var cmd = conn.Command("SELECT id, employee_id, name FROM users ORDER BY name"))
            await using (var r = await cmd.ExecuteReaderAsync(ct))
            {
                while (await r.ReadAsync(ct))
                    people.Add((r.GetInt("id"), r.GetString(r.GetOrdinal("employee_id")), r.GetString(r.GetOrdinal("name"))));
            }

            // First enrolled photo per person, so absent people still have a face.
            var photos = new Dictionary<int, string>();
            await using (var cmd = conn.Command("SELECT user_id, thumbnail_path FROM face_embeddings ORDER BY id"))
            await using (var r = await cmd.ExecuteReaderAsync(ct))
            {
                while (await r.ReadAsync(ct))
                    photos.TryAdd(r.GetInt("user_id"), PersonService.PhotoUrl(r.GetString(r.GetOrdinal("thumbnail_path"))));
            }

            var punches = new Dictionary<int, (DateTime First, DateTime Last, int Count, SortedSet<string> Cameras)>();
            await using (var cmd = conn.Command(@"
                SELECT user_id, camera_name, timestamp FROM attendance_events
                WHERE timestamp >= @from AND timestamp < @to ORDER BY timestamp, id")
                .With("@from", fromUtc).With("@to", toUtc))
            await using (var r = await cmd.ExecuteReaderAsync(ct))
            {
                while (await r.ReadAsync(ct))
                {
                    var cameraName = r.GetString(r.GetOrdinal("camera_name"));
                    if (hidden.Contains(cameraName)) continue;
                    var uid = r.GetInt("user_id");
                    var ts = r.GetUtc("timestamp");
                    if (!punches.TryGetValue(uid, out var p))
                    {
                        punches[uid] = (ts, ts, 1, new SortedSet<string>(StringComparer.OrdinalIgnoreCase) { cameraName });
                        continue;
                    }
                    p.Cameras.Add(cameraName);
                    var count = (ts - p.Last).TotalSeconds < _punchCooldownSeconds ? p.Count : p.Count + 1;
                    punches[uid] = (p.First, ts, count, p.Cameras);
                }
            }

            return people.Select(u =>
            {
                photos.TryGetValue(u.Id, out var photo);
                return punches.TryGetValue(u.Id, out var p)
                    ? new AttendanceDailyDto(u.Id, u.EmployeeId, u.Name, photo, true,
                        new DateTimeOffset(p.First).ToOffset(TimeIst.Offset), new DateTimeOffset(p.Last).ToOffset(TimeIst.Offset),
                        p.Count, p.Cameras.ToList())
                    : new AttendanceDailyDto(u.Id, u.EmployeeId, u.Name, photo, false, null, null, 0, new List<string>());
            }).ToList();
        }

        /// <summary>One row per person for an India day: seconds of each activity,
        /// summed over the requested camera or over all of the caller's cameras.</summary>
        public async Task<List<ActivityDailyDto>> ActivityDailyAsync(CurrentUser user, string? date, string? cameraId, CancellationToken ct)
        {
            var day = TimeIst.ParseOrToday(date);
            await using var conn = await _db.OpenAsync(ct);
            var cameraIds = await CamerasToQueryAsync(conn, user, cameraId, ct);
            if (cameraIds.Count == 0) return new List<ActivityDailyDto>();

            await using var cmd = conn.Command("");
            cmd.CommandText = $@"
                SELECT employee_id, MAX(person_name) AS person_name,
                       SUM(sitting_seconds) AS sitting, SUM(standing_seconds) AS standing,
                       SUM(walking_seconds) AS walking, SUM(low_movement_seconds) AS low_movement,
                       SUM(phone_seconds) AS phone, SUM(monitor_seconds) AS monitor,
                       SUM(computer_seconds) AS computer, SUM(present_seconds) AS present,
                       SUM(phone_sessions) AS phone_sessions,
                       MIN(first_seen_ist) AS first_seen, MAX(last_seen_ist) AS last_seen
                FROM v_user_activity_daily
                WHERE day_ist = @day AND camera_id IN ({cmd.InList("c", cameraIds)})
                GROUP BY employee_id
                ORDER BY (employee_id IS NULL), person_name";
            cmd.With("@day", day.ToDateTime(TimeOnly.MinValue));
            await using var r = await cmd.ExecuteReaderAsync(ct);
            var list = new List<ActivityDailyDto>();
            while (await r.ReadAsync(ct))
            {
                list.Add(new ActivityDailyDto(
                    r.GetStringOrNull("employee_id"),
                    PersonName(r.GetStringOrNull("employee_id"), r.GetStringOrNull("person_name")),
                    Round(r.GetDouble("sitting")), Round(r.GetDouble("standing")), Round(r.GetDouble("walking")),
                    Round(r.GetDouble("low_movement")), Round(r.GetDouble("phone")), Round(r.GetDouble("monitor")),
                    Round(r.GetDouble("computer")), Round(r.GetDouble("present")), r.GetInt("phone_sessions"),
                    Ist(r, "first_seen"), Ist(r, "last_seen")));
            }
            return list;
        }

        /// <summary>Raw sessions, newest first, paged.</summary>
        public async Task<Paged<ActivitySessionDto>> ActivitySessionsAsync(
            CurrentUser user, string? date, string? cameraId, string? employeeId, string? activityType,
            int page, int pageSize, CancellationToken ct)
        {
            (page, pageSize) = Paging.Normalize(page, pageSize);
            var (fromUtc, toUtc) = TimeIst.DayToUtcRange(TimeIst.ParseOrToday(date));
            await using var conn = await _db.OpenAsync(ct);
            var cameraIds = await CamerasToQueryAsync(conn, user, cameraId, ct);
            if (cameraIds.Count == 0) return new Paged<ActivitySessionDto>(new List<ActivitySessionDto>(), page, pageSize, 0);

            await using var cmd = conn.Command("");
            var where = $"WHERE start_time_utc >= @from AND start_time_utc < @to AND camera_id IN ({cmd.InList("c", cameraIds)})";
            cmd.With("@from", fromUtc).With("@to", toUtc);
            if (!string.IsNullOrWhiteSpace(employeeId))
            {
                where += " AND employee_id = @emp";
                cmd.With("@emp", employeeId.Trim());
            }
            if (!string.IsNullOrWhiteSpace(activityType))
            {
                where += " AND activity_type = @type";
                cmd.With("@type", activityType.Trim().ToUpperInvariant());
            }

            cmd.CommandText = $"SELECT COUNT(*) FROM v_activity_sessions_named {where}";
            var total = Convert.ToInt64(await cmd.ExecuteScalarAsync(ct));

            cmd.CommandText = $@"
                SELECT id, employee_id, person_name, camera_id, camera_name, activity_type, activity_name,
                       start_time_utc, end_time_utc, duration_seconds, confidence
                FROM v_activity_sessions_named {where}
                ORDER BY start_time_utc DESC, id DESC LIMIT @limit OFFSET @offset";
            cmd.With("@limit", pageSize).With("@offset", (page - 1) * pageSize);
            var items = new List<ActivitySessionDto>();
            await using (var r = await cmd.ExecuteReaderAsync(ct))
            {
                while (await r.ReadAsync(ct))
                {
                    items.Add(new ActivitySessionDto(
                        r.GetLong("id"), r.GetStringOrNull("employee_id"),
                        PersonName(r.GetStringOrNull("employee_id"), r.GetStringOrNull("person_name")),
                        r.GetString(r.GetOrdinal("camera_id")), r.GetStringOrNull("camera_name"),
                        r.GetString(r.GetOrdinal("activity_type")), r.GetStringOrNull("activity_name"),
                        r.GetUtc("start_time_utc"), r.GetUtc("end_time_utc"), Round(r.GetDouble("duration_seconds")), r.GetDouble("confidence")));
                }
            }
            return new Paged<ActivitySessionDto>(items, page, pageSize, total);
        }

        public async Task<List<ActivityTypeDto>> ActivityTypesAsync(CancellationToken ct)
        {
            await using var conn = await _db.OpenAsync(ct);
            await using var cmd = conn.Command("SELECT code, display_name, category, sort_order FROM activity_types ORDER BY sort_order");
            await using var r = await cmd.ExecuteReaderAsync(ct);
            var list = new List<ActivityTypeDto>();
            while (await r.ReadAsync(ct))
                list.Add(new ActivityTypeDto(r.GetString(0), r.GetString(1), r.GetString(2), r.GetInt32(3)));
            return list;
        }

        private async Task<List<string>> CamerasToQueryAsync(MySql.Data.MySqlClient.MySqlConnection conn, CurrentUser user, string? cameraId, CancellationToken ct)
        {
            var scope = await _cameras.ScopeAsync(conn, user.Id, ct);
            if (string.IsNullOrWhiteSpace(cameraId)) return scope.Select(c => c.CameraId).ToList();
            CameraService.EnsureAllowed(new[] { cameraId }, await _cameras.AllAsync(conn, ct), scope);
            return new List<string> { cameraId };
        }

        /// <summary>Names of cameras outside the caller's scope; attendance rows store
        /// the camera name, not its id.</summary>
        private async Task<HashSet<string>> HiddenCameraNamesAsync(MySql.Data.MySqlClient.MySqlConnection conn, CurrentUser user, CancellationToken ct)
        {
            var all = await _cameras.AllAsync(conn, ct);
            var scope = (await _cameras.ScopeAsync(conn, user.Id, ct)).Select(c => c.Name).ToHashSet(StringComparer.OrdinalIgnoreCase);
            return all.Select(c => c.Name).Where(n => !scope.Contains(n)).ToHashSet(StringComparer.OrdinalIgnoreCase);
        }

        /// <summary>Label for a person in activity reports. Sessions of someone the
        /// cameras never matched to a registered person (no employee id) are
        /// reported as "Unidentified person" -- the view's own fallback is "Unknown".</summary>
        public const string UnidentifiedPerson = "Unidentified person";

        private static string PersonName(string? employeeId, string? name)
        {
            if (string.IsNullOrWhiteSpace(employeeId)) return UnidentifiedPerson;
            // An id without a name (person since removed from Register): keep the id visible.
            return string.IsNullOrWhiteSpace(name) || name == "Unknown" ? $"Employee {employeeId}" : name;
        }

        private static double Round(double v) => Math.Round(v, 1);

        private static DateTimeOffset? Ist(System.Data.Common.DbDataReader r, string column)
        {
            var i = r.GetOrdinal(column);
            return r.IsDBNull(i) ? null : TimeIst.AsIst(r.GetDateTime(i));
        }
    }
}
