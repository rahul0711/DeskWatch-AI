namespace CCTV_Backend_Project.Models
{
    /// <summary>Refresh-token session of a login (table: app_user_sessions).</summary>
    public class AppUserSession
    {
        public long Id { get; set; }

        /// <summary>FK to app_users.id (ON DELETE CASCADE).</summary>
        public int AppUserId { get; set; }

        /// <summary>Hex SHA-256 of the refresh token (64 chars, unique).</summary>
        public string TokenHash { get; set; } = string.Empty;

        /// <summary>UTC.</summary>
        public DateTime CreatedAt { get; set; }

        /// <summary>UTC.</summary>
        public DateTime ExpiresAt { get; set; }

        /// <summary>UTC; set on logout, password change, or disable.</summary>
        public DateTime? RevokedAt { get; set; }

        public string? IpAddress { get; set; }

        public string? UserAgent { get; set; }
    }
}
