using CCTV_Backend_Project.Dtos;
using CCTV_Backend_Project.Infrastructure;
using CCTV_Backend_Project.Security;
using MySql.Data.MySqlClient;

namespace CCTV_Backend_Project.Services
{
    /// <summary>Workstations (table: workstations): a seat box and the screen box
    /// the person in that seat faces, drawn by an admin on a camera snapshot.
    /// The Python camera server re-reads them every few seconds and uses them
    /// for sitting/standing and "looking at their own screen".</summary>
    public class WorkstationService
    {
        private readonly Db _db;
        private readonly AuditService _audit;
        private readonly CameraService _cameras;

        public WorkstationService(Db db, AuditService audit, CameraService cameras)
        {
            _db = db;
            _audit = audit;
            _cameras = cameras;
        }

        public async Task<List<WorkstationDto>> ListAsync(CurrentUser user, string cameraId, CancellationToken ct)
        {
            await using var conn = await _db.OpenAsync(ct);
            await EnsureCameraAsync(conn, user, cameraId, ct);
            await using var cmd = conn.Command(
                "SELECT id, camera_id, name, seat_x1, seat_y1, seat_x2, seat_y2, screen_x1, screen_y1, screen_x2, screen_y2, created_at " +
                "FROM workstations WHERE camera_id = @cam ORDER BY id").With("@cam", cameraId);
            await using var r = await cmd.ExecuteReaderAsync(ct);
            var list = new List<WorkstationDto>();
            while (await r.ReadAsync(ct)) list.Add(Read(r));
            return list;
        }

        public async Task<WorkstationDto> CreateAsync(CurrentUser actor, string cameraId, SaveWorkstationRequest body, CancellationToken ct)
        {
            var (name, seat, screen) = Validate(body);
            await using var conn = await _db.OpenAsync(ct);
            await EnsureCameraAsync(conn, actor, cameraId, ct);
            await using var tx = await conn.BeginTransactionAsync(ct);
            long id;
            await using (var cmd = conn.Command(@"
                INSERT INTO workstations (camera_id, name, seat_x1, seat_y1, seat_x2, seat_y2,
                                          screen_x1, screen_y1, screen_x2, screen_y2, created_at, created_by)
                VALUES (@cam, @name, @sx1, @sy1, @sx2, @sy2, @cx1, @cy1, @cx2, @cy2, @at, @by)", tx))
            {
                Bind(cmd, cameraId, name, seat, screen).With("@at", DateTime.UtcNow).With("@by", actor.Id);
                await cmd.ExecuteNonQueryAsync(ct);
                id = cmd.LastInsertedId;
            }
            await _audit.WriteAsync(conn, tx, actor, "workstation.create", "workstation", id.ToString(), cameraId,
                new { name, seat, screen }, ct);
            await tx.CommitAsync(ct);
            return await GetAsync(conn, (int)id, ct);
        }

        public async Task<WorkstationDto> UpdateAsync(CurrentUser actor, string cameraId, int id, SaveWorkstationRequest body, CancellationToken ct)
        {
            var (name, seat, screen) = Validate(body);
            await using var conn = await _db.OpenAsync(ct);
            await EnsureCameraAsync(conn, actor, cameraId, ct);
            await using var tx = await conn.BeginTransactionAsync(ct);
            await using (var cmd = conn.Command(@"
                UPDATE workstations SET name = @name, seat_x1 = @sx1, seat_y1 = @sy1, seat_x2 = @sx2, seat_y2 = @sy2,
                       screen_x1 = @cx1, screen_y1 = @cy1, screen_x2 = @cx2, screen_y2 = @cy2
                WHERE id = @id AND camera_id = @cam", tx))
            {
                Bind(cmd, cameraId, name, seat, screen).With("@id", id);
                if (await cmd.ExecuteNonQueryAsync(ct) == 0) throw ApiException.NotFound("Workstation not found");
            }
            await _audit.WriteAsync(conn, tx, actor, "workstation.update", "workstation", id.ToString(), cameraId,
                new { name, seat, screen }, ct);
            await tx.CommitAsync(ct);
            return await GetAsync(conn, id, ct);
        }

        public async Task DeleteAsync(CurrentUser actor, string cameraId, int id, CancellationToken ct)
        {
            await using var conn = await _db.OpenAsync(ct);
            await EnsureCameraAsync(conn, actor, cameraId, ct);
            await using var tx = await conn.BeginTransactionAsync(ct);
            await using (var cmd = conn.Command("DELETE FROM workstations WHERE id = @id AND camera_id = @cam", tx)
                .With("@id", id).With("@cam", cameraId))
            {
                if (await cmd.ExecuteNonQueryAsync(ct) == 0) throw ApiException.NotFound("Workstation not found");
            }
            await _audit.WriteAsync(conn, tx, actor, "workstation.delete", "workstation", id.ToString(), cameraId, null, ct);
            await tx.CommitAsync(ct);
        }

        private async Task EnsureCameraAsync(MySqlConnection conn, CurrentUser user, string cameraId, CancellationToken ct)
        {
            var all = await _cameras.AllAsync(conn, ct);
            CameraService.EnsureAllowed(new[] { cameraId }, all, user.IsSuperAdmin ? all : await _cameras.ScopeAsync(conn, user.Id, ct));
        }

        private static (string Name, double[] Seat, double[]? Screen) Validate(SaveWorkstationRequest body)
        {
            var name = body.Name?.Trim() ?? "";
            if (name.Length == 0 || name.Length > 100) throw ApiException.BadRequest("Name is required (up to 100 characters)");
            return (name, Box(body.Seat, "seat") ?? throw ApiException.BadRequest("A seat box is required"), Box(body.Screen, "screen"));
        }

        /// <summary>[x1, y1, x2, y2] as fractions of the frame, normalised so x1 &lt; x2, y1 &lt; y2.</summary>
        private static double[]? Box(double[]? b, string what)
        {
            if (b is null) return null;
            if (b.Length != 4 || b.Any(v => double.IsNaN(v) || v < 0 || v > 1))
                throw ApiException.BadRequest($"{what} must be [x1, y1, x2, y2] between 0 and 1");
            var box = new[] { Math.Min(b[0], b[2]), Math.Min(b[1], b[3]), Math.Max(b[0], b[2]), Math.Max(b[1], b[3]) };
            if (box[2] - box[0] < 0.005 || box[3] - box[1] < 0.005) throw ApiException.BadRequest($"{what} box is too small");
            return box;
        }

        private static MySqlCommand Bind(MySqlCommand cmd, string cameraId, string name, double[] seat, double[]? screen)
            => cmd.With("@cam", cameraId).With("@name", name)
                .With("@sx1", seat[0]).With("@sy1", seat[1]).With("@sx2", seat[2]).With("@sy2", seat[3])
                .With("@cx1", screen?[0]).With("@cy1", screen?[1]).With("@cx2", screen?[2]).With("@cy2", screen?[3]);

        private async Task<WorkstationDto> GetAsync(MySqlConnection conn, int id, CancellationToken ct)
        {
            await using var cmd = conn.Command(
                "SELECT id, camera_id, name, seat_x1, seat_y1, seat_x2, seat_y2, screen_x1, screen_y1, screen_x2, screen_y2, created_at " +
                "FROM workstations WHERE id = @id").With("@id", id);
            await using var r = await cmd.ExecuteReaderAsync(ct);
            if (!await r.ReadAsync(ct)) throw ApiException.NotFound("Workstation not found");
            return Read(r);
        }

        private static WorkstationDto Read(System.Data.Common.DbDataReader r)
        {
            double[]? screen = r.IsDBNull(r.GetOrdinal("screen_x1")) ? null : new[]
            {
                r.GetDouble("screen_x1"), r.GetDouble("screen_y1"), r.GetDouble("screen_x2"), r.GetDouble("screen_y2"),
            };
            return new WorkstationDto(
                r.GetInt("id"), r.GetString(r.GetOrdinal("camera_id")), r.GetString(r.GetOrdinal("name")),
                new[] { r.GetDouble("seat_x1"), r.GetDouble("seat_y1"), r.GetDouble("seat_x2"), r.GetDouble("seat_y2") },
                screen, r.GetUtc("created_at"));
        }
    }
}
