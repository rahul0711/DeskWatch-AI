using CCTV_Backend_Project.Dtos;
using CCTV_Backend_Project.Infrastructure;
using CCTV_Backend_Project.Models;
using CCTV_Backend_Project.Security;
using MySql.Data.MySqlClient;

namespace CCTV_Backend_Project.Services
{
    /// <summary>Tracked people (users, face_embeddings, camera_restrictions).
    /// Registering, renaming and deleting people stays in the Python attendance
    /// server -- it needs the face model and rebuilds its face index. This backend
    /// reads them and owns camera_restrictions (Python re-reads it every 5 s).</summary>
    public class PersonService
    {
        private readonly Db _db;
        private readonly AuditService _audit;
        private readonly CameraService _cameras;

        public PersonService(Db db, AuditService audit, CameraService cameras)
        {
            _db = db;
            _audit = audit;
            _cameras = cameras;
        }

        /// <summary>Same shape as the Python GET /api/users. blocked_cameras only
        /// lists cameras inside the caller's scope.</summary>
        public async Task<List<PersonDto>> ListAsync(CurrentUser user, string? search, CancellationToken ct)
        {
            await using var conn = await _db.OpenAsync(ct);
            var scope = (await _cameras.ScopeAsync(conn, user.Id, ct)).Select(c => c.CameraId).ToHashSet();

            var people = new Dictionary<int, PersonDto>();
            await using (var cmd = conn.Command(
                "SELECT id, employee_id, name, created_at FROM users" +
                (string.IsNullOrWhiteSpace(search) ? "" : " WHERE name LIKE @s OR employee_id LIKE @s") +
                " ORDER BY id").With("@s", $"%{search?.Trim()}%"))
            await using (var r = await cmd.ExecuteReaderAsync(ct))
            {
                while (await r.ReadAsync(ct))
                {
                    var id = r.GetInt("id");
                    people[id] = new PersonDto(id, r.GetString(r.GetOrdinal("employee_id")), r.GetString(r.GetOrdinal("name")),
                        r.GetUtc("created_at"), new List<string>(), new List<FaceDto>());
                }
            }

            // Never select the embedding column: the browser has no use for it.
            await using (var cmd = conn.Command(
                "SELECT id, user_id, thumbnail_path, original_path, face_width_px, det_score, created_at FROM face_embeddings ORDER BY id"))
            await using (var r = await cmd.ExecuteReaderAsync(ct))
            {
                while (await r.ReadAsync(ct))
                {
                    if (!people.TryGetValue(r.GetInt("user_id"), out var p)) continue;
                    var original = r.GetStringOrNull("original_path");
                    p.Faces.Add(new FaceDto(r.GetInt("id"), PhotoUrl(r.GetString(r.GetOrdinal("thumbnail_path"))),
                        original is null ? null : PhotoUrl(original), r.GetInt("face_width_px"), r.GetDouble("det_score"), r.GetUtc("created_at")));
                }
            }

            await using (var cmd = conn.Command("SELECT user_id, camera_id FROM camera_restrictions ORDER BY camera_id"))
            await using (var r = await cmd.ExecuteReaderAsync(ct))
            {
                while (await r.ReadAsync(ct))
                {
                    var cam = r.GetString(1);
                    if (scope.Contains(cam) && people.TryGetValue(r.GetInt32(0), out var p)) p.BlockedCameras.Add(cam);
                }
            }
            return people.Values.ToList();
        }

        public async Task<List<PersonCameraAccessDto>> GetCameraAccessAsync(CurrentUser user, int personId, CancellationToken ct)
        {
            await using var conn = await _db.OpenAsync(ct);
            await EnsurePersonAsync(conn, null, personId, ct);
            var scope = await _cameras.ScopeAsync(conn, user.Id, ct);
            var blocked = await BlockedAsync(conn, null, personId, ct);
            return scope.Select(c => new PersonCameraAccessDto(
                c.CameraId, c.Name, blocked.ContainsKey(c.CameraId) ? "BLOCKED" : "ALLOWED",
                blocked.TryGetValue(c.CameraId, out var since) ? since : null)).ToList();
        }

        /// <summary>Merge, not replace: only cameras in the caller's scope change;
        /// blocks on cameras outside it are kept exactly as they are.</summary>
        public async Task<CameraAccessResponse> SetCameraAccessAsync(CurrentUser user, int personId, CameraAccessRequest body, CancellationToken ct)
        {
            var requested = (body.BlockedCameras ?? new()).Select(c => c.Trim()).Where(c => c.Length > 0).Distinct().ToList();
            await using var conn = await _db.OpenAsync(ct);
            var person = await EnsurePersonAsync(conn, null, personId, ct);
            var all = await _cameras.AllAsync(conn, ct);
            var scope = await _cameras.ScopeAsync(conn, user.Id, ct);
            CameraService.EnsureAllowed(requested, all, scope);
            var scopeIds = scope.Select(c => c.CameraId).ToHashSet();

            await using var tx = await conn.BeginTransactionAsync(ct);
            var before = (await BlockedAsync(conn, tx, personId, ct)).Keys.OrderBy(c => c).ToList();
            var final = requested.Concat(before.Where(c => !scopeIds.Contains(c))).Distinct().OrderBy(c => c).ToList();

            var toRemove = before.Except(final).ToList();
            var toAdd = final.Except(before).ToList();
            foreach (var cam in toRemove)
            {
                await using var del = conn.Command("DELETE FROM camera_restrictions WHERE user_id = @u AND camera_id = @c", tx)
                    .With("@u", personId).With("@c", cam);
                await del.ExecuteNonQueryAsync(ct);
            }
            var now = DateTime.UtcNow;
            foreach (var cam in toAdd)
            {
                await using var ins = conn.Command("INSERT INTO camera_restrictions (user_id, camera_id, created_at) VALUES (@u, @c, @now)", tx)
                    .With("@u", personId).With("@c", cam).With("@now", now);
                await ins.ExecuteNonQueryAsync(ct);
            }
            if (toAdd.Count + toRemove.Count > 0)
            {
                await _audit.WriteAsync(conn, tx, user, "person.camera_access", "person", personId.ToString(), null,
                    new { employee_id = person.EmployeeId, name = person.Name, blocked_added = toAdd, blocked_removed = toRemove }, ct);
            }
            await tx.CommitAsync(ct);
            return new CameraAccessResponse(personId, final.Where(scopeIds.Contains).ToList());
        }

        /// <summary>/data/... URL for a file under a data/ folder; served by the Python
        /// attendance server (same rule as enrollment._thumbnail_url).</summary>
        public static string PhotoUrl(string pathOnDisk)
        {
            var parts = pathOnDisk.Replace('\\', '/').Split('/', StringSplitOptions.RemoveEmptyEntries);
            var i = Array.IndexOf(parts, "data");
            return i >= 0 ? "/data/" + string.Join('/', parts.Skip(i + 1)) : "/data/" + parts.LastOrDefault();
        }

        private static async Task<User> EnsurePersonAsync(MySqlConnection conn, MySqlTransaction? tx, int personId, CancellationToken ct)
        {
            await using var cmd = conn.Command("SELECT id, employee_id, name, created_at FROM users WHERE id = @id", tx).With("@id", personId);
            await using var r = await cmd.ExecuteReaderAsync(ct);
            if (!await r.ReadAsync(ct)) throw ApiException.NotFound("Person not found");
            return new User
            {
                Id = r.GetInt("id"),
                EmployeeId = r.GetString(r.GetOrdinal("employee_id")),
                Name = r.GetString(r.GetOrdinal("name")),
                CreatedAt = r.GetUtc("created_at"),
            };
        }

        private static async Task<Dictionary<string, DateTime>> BlockedAsync(MySqlConnection conn, MySqlTransaction? tx, int personId, CancellationToken ct)
        {
            var map = new Dictionary<string, DateTime>();
            await using var cmd = conn.Command("SELECT camera_id, created_at FROM camera_restrictions WHERE user_id = @u", tx).With("@u", personId);
            await using var r = await cmd.ExecuteReaderAsync(ct);
            while (await r.ReadAsync(ct)) map[r.GetString(0)] = r.GetUtc("created_at");
            return map;
        }
    }
}
