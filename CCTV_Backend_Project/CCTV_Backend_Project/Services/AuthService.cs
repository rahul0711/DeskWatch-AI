using CCTV_Backend_Project.Dtos;
using CCTV_Backend_Project.Infrastructure;
using CCTV_Backend_Project.Models;
using CCTV_Backend_Project.Security;
using MySql.Data.MySqlClient;

namespace CCTV_Backend_Project.Services
{
    public class AuthService
    {
        public const int MaxFailedLogins = 5;
        public static readonly TimeSpan LockFor = TimeSpan.FromMinutes(15);
        public const int MinPasswordLength = 8;

        // Verified when the username does not exist, so a wrong username and a
        // wrong password take the same time.
        private static readonly string DummyHash = BCrypt.Net.BCrypt.HashPassword("dummy-password-for-timing", 11);
        private const string LoginFailed = "Invalid username or password";

        private readonly Db _db;
        private readonly TokenService _tokens;
        private readonly AuditService _audit;
        private readonly CameraService _cameras;
        private readonly SessionValidator _sessions;

        public AuthService(Db db, TokenService tokens, AuditService audit, CameraService cameras, SessionValidator sessions)
        {
            _db = db;
            _tokens = tokens;
            _audit = audit;
            _cameras = cameras;
            _sessions = sessions;
        }

        public async Task<LoginResponse> LoginAsync(LoginRequest body, string? ip, string? userAgent, CancellationToken ct)
        {
            var username = (body.Username ?? "").Trim();
            if (username.Length == 0 || string.IsNullOrEmpty(body.Password)) throw ApiException.BadRequest("Username and password are required");

            await using var conn = await _db.OpenAsync(ct);
            // Username or email; both are case-insensitive (utf8mb4_unicode_ci).
            var user = await FindAsync(conn, null, "username = @u OR email = @u", ("@u", username), ct);
            var now = DateTime.UtcNow;

            if (user is null)
            {
                BCrypt.Net.BCrypt.Verify(body.Password, DummyHash);
                await _audit.WriteAsync(conn, null, null, username, ip, "login_failed", "login", null, null, new { reason = "unknown_user" }, ct);
                throw ApiException.Unauthorized(LoginFailed);
            }

            var passwordOk = BCrypt.Net.BCrypt.Verify(body.Password, user.PasswordHash);
            var locked = user.LockedUntil is { } until && until > now;
            if (!passwordOk || locked || !user.IsActive)
            {
                if (!passwordOk && !locked)
                {
                    var failed = user.FailedLoginCount + 1;
                    DateTime? lockUntil = failed >= MaxFailedLogins ? now + LockFor : null;
                    await using var upd = conn.Command(@"
                        UPDATE app_users SET failed_login_count = @f, locked_until = COALESCE(@lock, locked_until) WHERE id = @id")
                        .With("@f", lockUntil is null ? failed : 0).With("@lock", lockUntil).With("@id", user.Id);
                    await upd.ExecuteNonQueryAsync(ct);
                }
                var reason = locked ? "locked" : !user.IsActive ? "disabled" : "wrong_password";
                await _audit.WriteAsync(conn, null, user.Id, user.Username, ip, "login_failed", "login", user.Id.ToString(), null, new { reason }, ct);
                // Same message for every case, so nobody can probe which usernames exist.
                throw ApiException.Unauthorized(locked ? "Too many failed attempts, try again in 15 minutes" : LoginFailed);
            }

            var jti = TokenService.NewJti();
            var expires = now + _tokens.Lifetime;
            await using var tx = await conn.BeginTransactionAsync(ct);
            long sessionId;
            await using (var ins = conn.Command(@"
                INSERT INTO app_user_sessions (app_user_id, token_hash, created_at, expires_at, ip_address, user_agent)
                VALUES (@uid, @hash, @now, @exp, @ip, @ua)", tx)
                .With("@uid", user.Id).With("@hash", TokenService.Sha256Hex(jti)).With("@now", now).With("@exp", expires)
                .With("@ip", ip).With("@ua", userAgent is null ? null : userAgent[..Math.Min(userAgent.Length, 255)]))
            {
                await ins.ExecuteNonQueryAsync(ct);
                sessionId = ins.LastInsertedId;
            }
            await using (var upd = conn.Command(
                "UPDATE app_users SET failed_login_count = 0, locked_until = NULL, last_login_at = @now WHERE id = @id", tx)
                .With("@now", now).With("@id", user.Id))
            {
                await upd.ExecuteNonQueryAsync(ct);
            }
            await _audit.WriteAsync(conn, tx, user.Id, user.Username, ip, "login", "login", user.Id.ToString(), null, null, ct);
            await tx.CommitAsync(ct);

            var token = _tokens.CreateToken(user.Id, user.Username, user.RoleCode, sessionId, jti, expires);
            return new LoginResponse(token, expires, await MeAsync(conn, user, ct));
        }

        public async Task<MeResponse> MeAsync(CurrentUser current, CancellationToken ct)
        {
            await using var conn = await _db.OpenAsync(ct);
            var user = await FindAsync(conn, null, "id = @id", ("@id", current.Id), ct) ?? throw ApiException.Unauthorized();
            return await MeAsync(conn, user, ct);
        }

        public async Task ChangePasswordAsync(CurrentUser current, ChangePasswordRequest body, CancellationToken ct)
        {
            await using var conn = await _db.OpenAsync(ct);
            var user = await FindAsync(conn, null, "id = @id", ("@id", current.Id), ct) ?? throw ApiException.Unauthorized();
            if (!BCrypt.Net.BCrypt.Verify(body.CurrentPassword ?? "", user.PasswordHash))
                throw ApiException.BadRequest("Current password is wrong");
            ValidateNewPassword(body.NewPassword, user.Username);
            if (BCrypt.Net.BCrypt.Verify(body.NewPassword, user.PasswordHash))
                throw ApiException.BadRequest("New password must be different from the current one");

            await using var tx = await conn.BeginTransactionAsync(ct);
            await using (var upd = conn.Command(
                "UPDATE app_users SET password_hash = @h, must_change_password = 0, updated_at = @now WHERE id = @id", tx)
                .With("@h", HashPassword(body.NewPassword)).With("@now", DateTime.UtcNow).With("@id", user.Id))
            {
                await upd.ExecuteNonQueryAsync(ct);
            }
            // Every other device has to log in again with the new password.
            var revoked = await RevokeSessionsAsync(conn, tx, user.Id, exceptSessionId: current.SessionId, ct);
            await _audit.WriteAsync(conn, tx, current, "password.change", "login", user.Id.ToString(), null,
                new { other_sessions_ended = revoked.Count }, ct);
            await tx.CommitAsync(ct);
            _sessions.Evict(revoked);
            _sessions.Evict(current.SessionId);
        }

        public async Task LogoutAsync(CurrentUser current, CancellationToken ct)
        {
            await using var conn = await _db.OpenAsync(ct);
            await using (var cmd = conn.Command(
                "UPDATE app_user_sessions SET revoked_at = @now WHERE id = @sid AND revoked_at IS NULL")
                .With("@now", DateTime.UtcNow).With("@sid", current.SessionId))
            {
                await cmd.ExecuteNonQueryAsync(ct);
            }
            await _audit.WriteAsync(conn, null, current, "logout", "login", current.Id.ToString(), ct: ct);
            _sessions.Evict(current.SessionId);
        }

        // ---- shared helpers (also used by LoginService) -----------------------------

        public static string HashPassword(string password) => BCrypt.Net.BCrypt.HashPassword(password, 11);

        public static void ValidateNewPassword(string? password, string username)
        {
            if (string.IsNullOrEmpty(password) || password.Length < MinPasswordLength)
                throw ApiException.BadRequest($"Password must be at least {MinPasswordLength} characters");
            if (string.Equals(password, username, StringComparison.OrdinalIgnoreCase))
                throw ApiException.BadRequest("Password must not be the same as the username");
        }

        /// <summary>Ends sessions of a login; returns the ended session ids (to evict from the cache).</summary>
        public static async Task<List<long>> RevokeSessionsAsync(MySqlConnection conn, MySqlTransaction? tx, int appUserId, long? exceptSessionId, CancellationToken ct)
        {
            var ids = new List<long>();
            await using (var sel = conn.Command(
                "SELECT id FROM app_user_sessions WHERE app_user_id = @uid AND revoked_at IS NULL AND (@except IS NULL OR id <> @except)", tx)
                .With("@uid", appUserId).With("@except", exceptSessionId))
            await using (var r = await sel.ExecuteReaderAsync(ct))
            {
                while (await r.ReadAsync(ct)) ids.Add(r.GetInt64(0));
            }
            if (ids.Count == 0) return ids;
            await using var upd = conn.Command(
                "UPDATE app_user_sessions SET revoked_at = @now WHERE app_user_id = @uid AND revoked_at IS NULL AND (@except IS NULL OR id <> @except)", tx)
                .With("@now", DateTime.UtcNow).With("@uid", appUserId).With("@except", exceptSessionId);
            await upd.ExecuteNonQueryAsync(ct);
            return ids;
        }

        public static async Task<AppUser?> FindAsync(MySqlConnection conn, MySqlTransaction? tx, string where, (string Name, object Value) param, CancellationToken ct)
        {
            await using var cmd = conn.Command($@"
                SELECT id, username, full_name, email, password_hash, role_code, all_cameras, is_active, must_change_password,
                       failed_login_count, locked_until, last_login_at, created_by, created_at, updated_at
                FROM app_users WHERE {where} LIMIT 1", tx).With(param.Name, param.Value);
            await using var r = await cmd.ExecuteReaderAsync(ct);
            if (!await r.ReadAsync(ct)) return null;
            return ReadAppUser(r);
        }

        public static AppUser ReadAppUser(System.Data.Common.DbDataReader r) => new()
        {
            Id = r.GetInt("id"),
            Username = r.GetString(r.GetOrdinal("username")),
            FullName = r.GetString(r.GetOrdinal("full_name")),
            Email = r.GetStringOrNull("email"),
            PasswordHash = r.GetString(r.GetOrdinal("password_hash")),
            RoleCode = r.GetString(r.GetOrdinal("role_code")),
            AllCameras = r.GetBool("all_cameras"),
            IsActive = r.GetBool("is_active"),
            MustChangePassword = r.GetBool("must_change_password"),
            FailedLoginCount = r.GetInt("failed_login_count"),
            LockedUntil = r.GetUtcOrNull("locked_until"),
            LastLoginAt = r.GetUtcOrNull("last_login_at"),
            CreatedBy = r.GetIntOrNull("created_by"),
            CreatedAt = r.GetUtc("created_at"),
            UpdatedAt = r.GetUtc("updated_at"),
        };

        private async Task<MeResponse> MeAsync(MySqlConnection conn, AppUser user, CancellationToken ct)
        {
            var scope = await _cameras.ScopeAsync(conn, user.Id, ct);
            var roleName = user.RoleCode switch
            {
                Roles.SuperAdmin => "Super Admin",
                Roles.Admin => "Admin",
                Roles.Viewer => "Viewer",
                _ => user.RoleCode,
            };
            return new MeResponse(
                user.Id, user.Username, user.FullName, user.Email, user.RoleCode, roleName, user.MustChangePassword,
                user.RoleCode == Roles.SuperAdmin || user.AllCameras,
                scope.Select(c => new CameraRef(c.CameraId, c.CameraNo, c.Name)).ToList());
        }
    }
}
