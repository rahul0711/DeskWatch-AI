using System.Text.Json;
using CCTV_Backend_Project.Dtos;
using CCTV_Backend_Project.Infrastructure;
using CCTV_Backend_Project.Security;
using MySql.Data.MySqlClient;

namespace CCTV_Backend_Project.Services
{
    /// <summary>audit_log: one row per login, logout and change. `details` holds
    /// before/after values -- never passwords, hashes or tokens.</summary>
    public class AuditService
    {
        private readonly Db _db;

        public AuditService(Db db) => _db = db;

        public async Task WriteAsync(
            MySqlConnection conn, MySqlTransaction? tx,
            int? appUserId, string? username, string? ip,
            string action, string? targetType = null, string? targetId = null,
            string? cameraId = null, object? details = null, CancellationToken ct = default)
        {
            await using var cmd = conn.Command(@"
                INSERT INTO audit_log (at, app_user_id, username, action, target_type, target_id, camera_id, details, ip_address)
                VALUES (@at, @uid, @uname, @action, @ttype, @tid, @cam, @details, @ip)", tx)
                .With("@at", DateTime.UtcNow)
                .With("@uid", appUserId)
                .With("@uname", username)
                .With("@action", action)
                .With("@ttype", targetType)
                .With("@tid", targetId)
                .With("@cam", cameraId)
                .With("@details", details is null ? null : JsonSerializer.Serialize(details))
                .With("@ip", ip);
            await cmd.ExecuteNonQueryAsync(ct);
        }

        public Task WriteAsync(
            MySqlConnection conn, MySqlTransaction? tx, CurrentUser actor,
            string action, string? targetType = null, string? targetId = null,
            string? cameraId = null, object? details = null, CancellationToken ct = default)
            => WriteAsync(conn, tx, actor.Id, actor.Username, actor.IpAddress, action, targetType, targetId, cameraId, details, ct);

        public async Task<Paged<AuditDto>> ListAsync(int? loginId, string? action, int page, int pageSize, CancellationToken ct)
        {
            (page, pageSize) = Paging.Normalize(page, pageSize);
            var where = "WHERE 1=1";
            if (loginId is not null) where += " AND app_user_id = @uid";
            if (!string.IsNullOrWhiteSpace(action)) where += " AND action = @action";

            await using var conn = await _db.OpenAsync(ct);
            long total;
            await using (var count = conn.Command($"SELECT COUNT(*) FROM audit_log {where}")
                .With("@uid", loginId).With("@action", action))
            {
                total = Convert.ToInt64(await count.ExecuteScalarAsync(ct));
            }

            var items = new List<AuditDto>();
            await using var cmd = conn.Command($@"
                SELECT id, at, app_user_id, username, action, target_type, target_id, camera_id, details, ip_address
                FROM audit_log {where} ORDER BY id DESC LIMIT @limit OFFSET @offset")
                .With("@uid", loginId).With("@action", action)
                .With("@limit", pageSize).With("@offset", (page - 1) * pageSize);
            await using var r = await cmd.ExecuteReaderAsync(ct);
            while (await r.ReadAsync(ct))
            {
                items.Add(new AuditDto(
                    r.GetLong("id"), r.GetUtc("at"), r.GetIntOrNull("app_user_id"), r.GetStringOrNull("username"),
                    r.GetString(r.GetOrdinal("action")), r.GetStringOrNull("target_type"), r.GetStringOrNull("target_id"),
                    r.GetStringOrNull("camera_id"), r.GetStringOrNull("details"), r.GetStringOrNull("ip_address")));
            }
            return new Paged<AuditDto>(items, page, pageSize, total);
        }
    }

    public static class Paging
    {
        public const int DefaultPageSize = 50;
        public const int MaxPageSize = 500;

        public static (int Page, int PageSize) Normalize(int page, int pageSize)
            => (Math.Max(1, page), pageSize <= 0 ? DefaultPageSize : Math.Min(pageSize, MaxPageSize));
    }
}
