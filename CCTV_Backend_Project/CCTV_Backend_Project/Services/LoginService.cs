using CCTV_Backend_Project.Dtos;
using CCTV_Backend_Project.Infrastructure;
using CCTV_Backend_Project.Models;
using CCTV_Backend_Project.Security;
using MySql.Data.MySqlClient;

namespace CCTV_Backend_Project.Services
{
    /// <summary>Dashboard logins (app_users). Rules:
    ///   * rank rule: manage only lower roles (SUPER_ADMIN may manage SUPER_ADMINs too)
    ///   * no escalation: never give a role, a camera, or all_cameras you don't have
    ///   * never lock the system out: keep at least one active SUPER_ADMIN, and a
    ///     login cannot disable, delete or change the role of itself
    ///   * disabling / role change / password reset / delete ends the login's sessions</summary>
    public class LoginService
    {
        private readonly Db _db;
        private readonly AuditService _audit;
        private readonly CameraService _cameras;
        private readonly SessionValidator _sessions;

        public LoginService(Db db, AuditService audit, CameraService cameras, SessionValidator sessions)
        {
            _db = db;
            _audit = audit;
            _cameras = cameras;
            _sessions = sessions;
        }

        public async Task<List<LoginDto>> ListAsync(CurrentUser actor, string? search, string? role, bool? active, CancellationToken ct)
        {
            await using var conn = await _db.OpenAsync(ct);
            var manageable = Roles.All.Where(r => Roles.CanManage(actor.Role, r)).ToList();
            await using var cmd = conn.Command("");
            var sql = $@"
                SELECT id, username, full_name, email, password_hash, role_code, all_cameras, is_active, must_change_password,
                       failed_login_count, locked_until, last_login_at, created_by, created_at, updated_at
                FROM app_users WHERE role_code IN ({cmd.InList("r", manageable)})";
            if (!string.IsNullOrWhiteSpace(search))
            {
                sql += " AND (username LIKE @s OR full_name LIKE @s OR email LIKE @s)";
                cmd.With("@s", $"%{search.Trim()}%");
            }
            if (!string.IsNullOrWhiteSpace(role))
            {
                sql += " AND role_code = @role";
                cmd.With("@role", role);
            }
            if (active is not null)
            {
                sql += " AND is_active = @active";
                cmd.With("@active", active.Value);
            }
            cmd.CommandText = sql + " ORDER BY FIELD(role_code, 'SUPER_ADMIN', 'ADMIN', 'VIEWER'), username";

            var users = new List<AppUser>();
            await using (var r = await cmd.ExecuteReaderAsync(ct))
            {
                while (await r.ReadAsync(ct)) users.Add(AuthService.ReadAppUser(r));
            }
            var cameraMap = await CameraIdsAsync(conn, null, ct);
            return users.Select(u => ToDto(u, cameraMap)).ToList();
        }

        public async Task<List<RoleDto>> AssignableRolesAsync(CurrentUser actor, CancellationToken ct)
        {
            await using var conn = await _db.OpenAsync(ct);
            await using var cmd = conn.Command("SELECT code, name, rank_level, description FROM roles ORDER BY rank_level DESC");
            await using var r = await cmd.ExecuteReaderAsync(ct);
            var list = new List<RoleDto>();
            while (await r.ReadAsync(ct))
            {
                var code = r.GetString(r.GetOrdinal("code"));
                if (!Roles.CanManage(actor.Role, code)) continue;
                list.Add(new RoleDto(code, r.GetString(r.GetOrdinal("name")), r.GetInt("rank_level"), r.GetString(r.GetOrdinal("description"))));
            }
            return list;
        }

        public async Task<LoginDto> CreateAsync(CurrentUser actor, CreateLoginRequest body, CancellationToken ct)
        {
            var username = (body.Username ?? "").Trim();
            var fullName = (body.FullName ?? "").Trim();
            var email = string.IsNullOrWhiteSpace(body.Email) ? null : body.Email.Trim();
            if (username.Length is < 3 or > 64) throw ApiException.BadRequest("Username must be 3-64 characters");
            if (fullName.Length == 0) throw ApiException.BadRequest("Full name is required");
            if (email is not null && !email.Contains('@')) throw ApiException.BadRequest("Email is not valid");
            EnsureKnownRole(body.Role);
            EnsureCanManage(actor, body.Role);
            AuthService.ValidateNewPassword(body.Password, username);

            await using var conn = await _db.OpenAsync(ct);
            var cameraIds = (body.CameraIds ?? new()).Distinct().ToList();
            await EnsureCameraGrantAsync(conn, actor, body.AllCameras, cameraIds, ct);

            var now = DateTime.UtcNow;
            await using var tx = await conn.BeginTransactionAsync(ct);
            int id;
            await using (var ins = conn.Command(@"
                INSERT INTO app_users (username, full_name, email, password_hash, role_code, all_cameras, is_active,
                                       must_change_password, created_by, created_at, updated_at)
                VALUES (@u, @fn, @em, @h, @role, @all, 1, 1, @by, @now, @now)", tx)
                .With("@u", username).With("@fn", fullName).With("@em", email)
                .With("@h", AuthService.HashPassword(body.Password)).With("@role", body.Role)
                .With("@all", body.Role != Roles.SuperAdmin && body.AllCameras).With("@by", actor.Id).With("@now", now))
            {
                await ins.ExecuteNonQueryAsync(ct);
                id = (int)ins.LastInsertedId;
            }
            if (body.Role != Roles.SuperAdmin && !body.AllCameras)
                await ReplaceCamerasAsync(conn, tx, id, cameraIds, actor.Id, ct);
            await _audit.WriteAsync(conn, tx, actor, "login.create", "login", id.ToString(), null,
                new { username, full_name = fullName, email, role = body.Role, all_cameras = body.AllCameras, camera_ids = cameraIds }, ct);
            await tx.CommitAsync(ct);

            return await GetDtoAsync(conn, id, ct);
        }

        public async Task<LoginDto> UpdateAsync(CurrentUser actor, int id, UpdateLoginRequest body, CancellationToken ct)
        {
            await using var conn = await _db.OpenAsync(ct);
            var target = await GetManageableAsync(conn, actor, id, ct);

            var newRole = body.Role ?? target.RoleCode;
            EnsureKnownRole(newRole);
            EnsureCanManage(actor, newRole);
            var newActive = body.IsActive ?? target.IsActive;
            var newAll = body.AllCameras ?? target.AllCameras;
            var fullName = body.FullName?.Trim();
            if (fullName is not null && fullName.Length == 0) throw ApiException.BadRequest("Full name cannot be empty");
            var email = body.Email is null ? target.Email : (body.Email.Trim().Length == 0 ? null : body.Email.Trim());
            if (email is not null && !email.Contains('@')) throw ApiException.BadRequest("Email is not valid");

            if (id == actor.Id && (newRole != target.RoleCode || !newActive))
                throw ApiException.Conflict("You cannot change your own role or disable yourself");
            if (target.RoleCode == Roles.SuperAdmin && target.IsActive && (newRole != Roles.SuperAdmin || !newActive))
                await EnsureAnotherActiveSuperAdminAsync(conn, null, id, ct);
            if (body.AllCameras == true && !target.AllCameras)
                await EnsureCameraGrantAsync(conn, actor, true, new List<string>(), ct);

            var endSessions = newRole != target.RoleCode || (target.IsActive && !newActive);
            await using var tx = await conn.BeginTransactionAsync(ct);
            await using (var upd = conn.Command(@"
                UPDATE app_users SET full_name = @fn, email = @em, role_code = @role, is_active = @active,
                       all_cameras = @all, updated_at = @now WHERE id = @id", tx)
                .With("@fn", fullName ?? target.FullName).With("@em", email).With("@role", newRole)
                .With("@active", newActive).With("@all", newRole != Roles.SuperAdmin && newAll)
                .With("@now", DateTime.UtcNow).With("@id", id))
            {
                await upd.ExecuteNonQueryAsync(ct);
            }
            var revoked = endSessions ? await AuthService.RevokeSessionsAsync(conn, tx, id, null, ct) : new List<long>();
            await _audit.WriteAsync(conn, tx, actor, "login.update", "login", id.ToString(), null, new
            {
                before = new { full_name = target.FullName, email = target.Email, role = target.RoleCode, is_active = target.IsActive, all_cameras = target.AllCameras },
                after = new { full_name = fullName ?? target.FullName, email, role = newRole, is_active = newActive, all_cameras = newAll },
                sessions_ended = revoked.Count,
            }, ct);
            await tx.CommitAsync(ct);
            _sessions.Evict(revoked);

            return await GetDtoAsync(conn, id, ct);
        }

        /// <summary>Replaces the login's cameras. An ADMIN can only grant or remove
        /// cameras inside their own scope; the target's other cameras are kept.</summary>
        public async Task<LoginDto> SetCamerasAsync(CurrentUser actor, int id, SetLoginCamerasRequest body, CancellationToken ct)
        {
            await using var conn = await _db.OpenAsync(ct);
            var target = await GetManageableAsync(conn, actor, id, ct);
            if (target.RoleCode == Roles.SuperAdmin) throw ApiException.BadRequest("A Super Admin always sees every camera");

            var requested = (body.CameraIds ?? new()).Distinct().ToList();
            var all = await _cameras.AllAsync(conn, ct);
            var scope = actor.IsSuperAdmin ? all : await _cameras.ScopeAsync(conn, actor.Id, ct);
            CameraService.EnsureAllowed(requested, all, scope);

            var current = (await CameraIdsAsync(conn, null, ct)).GetValueOrDefault(id) ?? new List<string>();
            var outside = current.Where(c => !scope.Any(s => s.CameraId == c));
            var final = requested.Concat(outside).Distinct().OrderBy(c => c).ToList();

            await using var tx = await conn.BeginTransactionAsync(ct);
            await ReplaceCamerasAsync(conn, tx, id, final, actor.Id, ct);
            await _audit.WriteAsync(conn, tx, actor, "login.cameras", "login", id.ToString(), null,
                new { before = current, after = final }, ct);
            await tx.CommitAsync(ct);
            return await GetDtoAsync(conn, id, ct);
        }

        public async Task ResetPasswordAsync(CurrentUser actor, int id, ResetPasswordRequest body, CancellationToken ct)
        {
            await using var conn = await _db.OpenAsync(ct);
            var target = await GetManageableAsync(conn, actor, id, ct);
            if (id == actor.Id) throw ApiException.BadRequest("Use 'Change password' for your own login");
            AuthService.ValidateNewPassword(body.Password, target.Username);

            await using var tx = await conn.BeginTransactionAsync(ct);
            await using (var upd = conn.Command(@"
                UPDATE app_users SET password_hash = @h, must_change_password = 1, failed_login_count = 0,
                       locked_until = NULL, updated_at = @now WHERE id = @id", tx)
                .With("@h", AuthService.HashPassword(body.Password)).With("@now", DateTime.UtcNow).With("@id", id))
            {
                await upd.ExecuteNonQueryAsync(ct);
            }
            var revoked = await AuthService.RevokeSessionsAsync(conn, tx, id, null, ct);
            await _audit.WriteAsync(conn, tx, actor, "login.reset_password", "login", id.ToString(), null,
                new { username = target.Username, sessions_ended = revoked.Count }, ct);
            await tx.CommitAsync(ct);
            _sessions.Evict(revoked);
        }

        public async Task DeleteAsync(CurrentUser actor, int id, CancellationToken ct)
        {
            await using var conn = await _db.OpenAsync(ct);
            var target = await GetManageableAsync(conn, actor, id, ct);
            if (id == actor.Id) throw ApiException.Conflict("You cannot delete yourself");
            if (target.RoleCode == Roles.SuperAdmin && target.IsActive)
                await EnsureAnotherActiveSuperAdminAsync(conn, null, id, ct);

            await using var tx = await conn.BeginTransactionAsync(ct);
            var revoked = await AuthService.RevokeSessionsAsync(conn, tx, id, null, ct);
            // Sessions and camera grants go with it (ON DELETE CASCADE); audit rows stay.
            await using (var del = conn.Command("DELETE FROM app_users WHERE id = @id", tx).With("@id", id))
            {
                await del.ExecuteNonQueryAsync(ct);
            }
            await _audit.WriteAsync(conn, tx, actor, "login.delete", "login", id.ToString(), null,
                new { username = target.Username, role = target.RoleCode }, ct);
            await tx.CommitAsync(ct);
            _sessions.Evict(revoked);
        }

        // ---- helpers ---------------------------------------------------------------

        private static void EnsureKnownRole(string? role)
        {
            if (role is null || !Roles.All.Contains(role)) throw ApiException.BadRequest("Role must be SUPER_ADMIN, ADMIN or VIEWER");
        }

        private static void EnsureCanManage(CurrentUser actor, string targetRole)
        {
            if (!Roles.CanManage(actor.Role, targetRole))
                throw ApiException.Forbidden($"A {actor.Role} cannot manage {targetRole} logins");
        }

        private async Task<AppUser> GetManageableAsync(MySqlConnection conn, CurrentUser actor, int id, CancellationToken ct)
        {
            var target = await AuthService.FindAsync(conn, null, "id = @id", ("@id", id), ct)
                ?? throw ApiException.NotFound("Login not found");
            EnsureCanManage(actor, target.RoleCode);
            return target;
        }

        /// <summary>No escalation: an ADMIN may grant only cameras in their own scope,
        /// and all_cameras only when they have it themselves.</summary>
        private async Task EnsureCameraGrantAsync(MySqlConnection conn, CurrentUser actor, bool allCameras, List<string> cameraIds, CancellationToken ct)
        {
            var all = await _cameras.AllAsync(conn, ct);
            if (actor.IsSuperAdmin)
            {
                CameraService.EnsureAllowed(cameraIds, all, all);
                return;
            }
            var me = await AuthService.FindAsync(conn, null, "id = @id", ("@id", actor.Id), ct) ?? throw ApiException.Unauthorized();
            if (allCameras && !me.AllCameras) throw ApiException.Forbidden("You cannot grant all cameras: you do not have all cameras yourself");
            CameraService.EnsureAllowed(cameraIds, all, await _cameras.ScopeAsync(conn, actor.Id, ct));
        }

        private static async Task EnsureAnotherActiveSuperAdminAsync(MySqlConnection conn, MySqlTransaction? tx, int exceptId, CancellationToken ct)
        {
            await using var cmd = conn.Command(
                "SELECT COUNT(*) FROM app_users WHERE role_code = 'SUPER_ADMIN' AND is_active = 1 AND id <> @id", tx).With("@id", exceptId);
            if (Convert.ToInt64(await cmd.ExecuteScalarAsync(ct)) == 0)
                throw ApiException.Conflict("This is the last active Super Admin; create another one first");
        }

        private static async Task ReplaceCamerasAsync(MySqlConnection conn, MySqlTransaction tx, int id, List<string> cameraIds, int grantedBy, CancellationToken ct)
        {
            await using (var del = conn.Command("DELETE FROM app_user_cameras WHERE app_user_id = @id", tx).With("@id", id))
            {
                await del.ExecuteNonQueryAsync(ct);
            }
            var now = DateTime.UtcNow;
            foreach (var cam in cameraIds)
            {
                await using var ins = conn.Command(
                    "INSERT INTO app_user_cameras (app_user_id, camera_id, granted_by, created_at) VALUES (@id, @cam, @by, @now)", tx)
                    .With("@id", id).With("@cam", cam).With("@by", grantedBy).With("@now", now);
                await ins.ExecuteNonQueryAsync(ct);
            }
        }

        private static async Task<Dictionary<int, List<string>>> CameraIdsAsync(MySqlConnection conn, MySqlTransaction? tx, CancellationToken ct)
        {
            var map = new Dictionary<int, List<string>>();
            await using var cmd = conn.Command("SELECT app_user_id, camera_id FROM app_user_cameras ORDER BY camera_id", tx);
            await using var r = await cmd.ExecuteReaderAsync(ct);
            while (await r.ReadAsync(ct))
            {
                var uid = r.GetInt32(0);
                if (!map.TryGetValue(uid, out var list)) map[uid] = list = new List<string>();
                list.Add(r.GetString(1));
            }
            return map;
        }

        private static async Task<LoginDto> GetDtoAsync(MySqlConnection conn, int id, CancellationToken ct)
        {
            var user = await AuthService.FindAsync(conn, null, "id = @id", ("@id", id), ct) ?? throw ApiException.NotFound("Login not found");
            return ToDto(user, await CameraIdsAsync(conn, null, ct));
        }

        private static LoginDto ToDto(AppUser u, Dictionary<int, List<string>> cameras) => new(
            u.Id, u.Username, u.FullName, u.Email, u.RoleCode, u.IsActive,
            u.RoleCode == Roles.SuperAdmin || u.AllCameras,
            cameras.GetValueOrDefault(u.Id) ?? new List<string>(),
            u.MustChangePassword, u.LockedUntil, u.LastLoginAt, u.CreatedBy, u.CreatedAt, u.UpdatedAt);
    }
}
