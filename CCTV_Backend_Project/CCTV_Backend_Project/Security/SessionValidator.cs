using System.Security.Claims;
using CCTV_Backend_Project.Infrastructure;
using Microsoft.Extensions.Caching.Memory;

namespace CCTV_Backend_Project.Security
{
    /// <summary>Runs after the JWT signature/expiry check. A valid signature is not
    /// enough: the session must not be revoked (logout, password change) and the
    /// login must still be active with the same role. Cached briefly so this is
    /// not a database query per request; Evict() drops the cache at once on logout
    /// and on changes made through this backend.</summary>
    public class SessionValidator
    {
        private static readonly TimeSpan CacheFor = TimeSpan.FromSeconds(30);
        private readonly Db _db;
        private readonly IMemoryCache _cache;

        public SessionValidator(Db db, IMemoryCache cache)
        {
            _db = db;
            _cache = cache;
        }

        public sealed record SessionState(int UserId, string Role, bool IsActive, bool MustChangePassword, DateTime ExpiresAt, DateTime? RevokedAt);

        public const string MustChangePasswordItem = "must_change_password";

        public async Task<string?> ValidateAsync(ClaimsPrincipal principal, HttpContext http, CancellationToken ct)
        {
            if (!int.TryParse(principal.FindFirstValue(TokenService.ClaimUserId), out var userId)
                || !long.TryParse(principal.FindFirstValue(TokenService.ClaimSessionId), out var sessionId))
                return "Malformed token";

            var state = await GetAsync(sessionId, ct);
            if (state is null || state.UserId != userId) return "Session not found";
            if (state.RevokedAt is not null) return "Session ended, log in again";
            if (state.ExpiresAt <= DateTime.UtcNow) return "Session expired, log in again";
            if (!state.IsActive) return "This login is disabled";
            if (state.Role != principal.FindFirstValue(TokenService.ClaimRole)) return "Role changed, log in again";
            http.Items[MustChangePasswordItem] = state.MustChangePassword;
            return null;
        }

        public void Evict(long sessionId) => _cache.Remove(Key(sessionId));

        public void Evict(IEnumerable<long> sessionIds)
        {
            foreach (var id in sessionIds) Evict(id);
        }

        private static string Key(long sessionId) => $"session:{sessionId}";

        private async Task<SessionState?> GetAsync(long sessionId, CancellationToken ct)
        {
            if (_cache.TryGetValue(Key(sessionId), out SessionState? cached)) return cached;

            await using var conn = await _db.OpenAsync(ct);
            await using var cmd = conn.Command(@"
                SELECT s.app_user_id, s.expires_at, s.revoked_at, a.role_code, a.is_active, a.must_change_password
                FROM app_user_sessions s
                JOIN app_users a ON a.id = s.app_user_id
                WHERE s.id = @sid").With("@sid", sessionId);
            await using var r = await cmd.ExecuteReaderAsync(ct);
            SessionState? state = null;
            if (await r.ReadAsync(ct))
            {
                state = new SessionState(
                    r.GetInt("app_user_id"), r.GetString(r.GetOrdinal("role_code")), r.GetBool("is_active"),
                    r.GetBool("must_change_password"), r.GetUtc("expires_at"), r.GetUtcOrNull("revoked_at"));
            }
            _cache.Set(Key(sessionId), state, CacheFor);
            return state;
        }
    }
}
