using CCTV_Backend_Project.Dtos;
using CCTV_Backend_Project.Infrastructure;
using CCTV_Backend_Project.Models;
using CCTV_Backend_Project.Security;
using MySql.Data.MySqlClient;

namespace CCTV_Backend_Project.Services
{
    /// <summary>The cameras table and each login's camera scope (v_app_user_cameras:
    /// SUPER_ADMIN and all_cameras = 1 get every active camera, everyone else
    /// only their app_user_cameras rows).</summary>
    public class CameraService
    {
        private readonly Db _db;
        private readonly AuditService _audit;

        public CameraService(Db db, AuditService audit)
        {
            _db = db;
            _audit = audit;
        }

        public async Task<List<Camera>> AllAsync(MySqlConnection conn, CancellationToken ct)
        {
            var list = new List<Camera>();
            await using var cmd = conn.Command(
                "SELECT camera_id, camera_no, name, ip_address, brand, is_active, face_enabled, activity_enabled FROM cameras ORDER BY camera_no");
            await using var r = await cmd.ExecuteReaderAsync(ct);
            while (await r.ReadAsync(ct))
            {
                list.Add(new Camera
                {
                    CameraId = r.GetString(r.GetOrdinal("camera_id")),
                    CameraNo = r.GetInt("camera_no"),
                    Name = r.GetString(r.GetOrdinal("name")),
                    IpAddress = r.GetStringOrNull("ip_address"),
                    Brand = r.GetStringOrNull("brand"),
                    IsActive = r.GetBool("is_active"),
                    FaceEnabled = r.GetBool("face_enabled"),
                    ActivityEnabled = r.GetBool("activity_enabled"),
                });
            }
            return list;
        }

        /// <summary>Active cameras this login may see, ordered by camera_no.</summary>
        public async Task<List<Camera>> ScopeAsync(MySqlConnection conn, int appUserId, CancellationToken ct)
        {
            var ids = new HashSet<string>();
            await using (var cmd = conn.Command("SELECT camera_id FROM v_app_user_cameras WHERE app_user_id = @id").With("@id", appUserId))
            await using (var r = await cmd.ExecuteReaderAsync(ct))
            {
                while (await r.ReadAsync(ct)) ids.Add(r.GetString(0));
            }
            return (await AllAsync(conn, ct)).Where(c => ids.Contains(c.CameraId)).ToList();
        }

        public async Task<List<CameraDto>> ListForAsync(CurrentUser user, CancellationToken ct)
        {
            await using var conn = await _db.OpenAsync(ct);
            // Super admins also see inactive cameras so they can switch them back on.
            var cams = user.IsSuperAdmin ? await AllAsync(conn, ct) : await ScopeAsync(conn, user.Id, ct);
            return cams.Select(ToDto).ToList();
        }

        public async Task<CameraDto> UpdateAsync(CurrentUser actor, string cameraId, UpdateCameraRequest body, CancellationToken ct)
        {
            await using var conn = await _db.OpenAsync(ct);
            var before = (await AllAsync(conn, ct)).FirstOrDefault(c => c.CameraId == cameraId)
                ?? throw ApiException.NotFound($"Camera '{cameraId}' not found");

            var name = body.Name?.Trim();
            if (name is not null && name.Length == 0) throw ApiException.BadRequest("Name cannot be empty");

            await using var tx = await conn.BeginTransactionAsync(ct);
            await using (var cmd = conn.Command(
                "UPDATE cameras SET name = COALESCE(@name, name), is_active = COALESCE(@active, is_active) WHERE camera_id = @id", tx)
                .With("@name", name).With("@active", body.IsActive).With("@id", cameraId))
            {
                await cmd.ExecuteNonQueryAsync(ct);
            }
            await _audit.WriteAsync(conn, tx, actor, "camera.update", "camera", cameraId, cameraId,
                new { before = new { before.Name, before.IsActive }, after = new { name = name ?? before.Name, is_active = body.IsActive ?? before.IsActive } }, ct);
            await tx.CommitAsync(ct);

            var after = (await AllAsync(conn, ct)).First(c => c.CameraId == cameraId);
            return ToDto(after);
        }

        /// <summary>Switch face / body movement on or off for one camera. Admins only
        /// for cameras in their scope. The Python camera server picks the change up
        /// within a few seconds and restarts just that camera.</summary>
        public async Task<CameraDto> UpdateFeaturesAsync(CurrentUser actor, string cameraId, UpdateCameraFeaturesRequest body, CancellationToken ct)
        {
            await using var conn = await _db.OpenAsync(ct);
            var all = await AllAsync(conn, ct);
            EnsureAllowed(new[] { cameraId }, all, actor.IsSuperAdmin ? all : await ScopeAsync(conn, actor.Id, ct));
            var before = all.First(c => c.CameraId == cameraId);

            await using var tx = await conn.BeginTransactionAsync(ct);
            await using (var cmd = conn.Command(
                "UPDATE cameras SET face_enabled = COALESCE(@face, face_enabled), activity_enabled = COALESCE(@act, activity_enabled) WHERE camera_id = @id", tx)
                .With("@face", body.FaceEnabled).With("@act", body.ActivityEnabled).With("@id", cameraId))
            {
                await cmd.ExecuteNonQueryAsync(ct);
            }
            await _audit.WriteAsync(conn, tx, actor, "camera.features", "camera", cameraId, cameraId,
                new
                {
                    before = new { face = before.FaceEnabled, body_movement = before.ActivityEnabled },
                    after = new { face = body.FaceEnabled ?? before.FaceEnabled, body_movement = body.ActivityEnabled ?? before.ActivityEnabled },
                }, ct);
            await tx.CommitAsync(ct);

            var after = (await AllAsync(conn, ct)).First(c => c.CameraId == cameraId);
            return ToDto(after);
        }

        /// <summary>400 for an unknown camera id, 403 for one outside the scope.</summary>
        public static void EnsureAllowed(IEnumerable<string> cameraIds, List<Camera> allCameras, List<Camera> scope)
        {
            foreach (var id in cameraIds)
            {
                if (!allCameras.Any(c => c.CameraId == id)) throw ApiException.BadRequest($"Unknown camera '{id}'");
                if (!scope.Any(c => c.CameraId == id)) throw ApiException.OutOfScope(id);
            }
        }

        public static CameraDto ToDto(Camera c) => new(c.CameraId, c.CameraNo, c.Name, c.IpAddress, c.Brand, c.IsActive,
            c.FaceEnabled, c.ActivityEnabled);
    }
}
