namespace CCTV_Backend_Project.Models
{
    /// <summary>Login account for the admin backend (table: app_users).</summary>
    public class AppUser
    {
        public int Id { get; set; }

        /// <summary>Login name, case-insensitive (unique).</summary>
        public string Username { get; set; } = string.Empty;

        public string FullName { get; set; } = string.Empty;

        /// <summary>Unique when set.</summary>
        public string? Email { get; set; }

        /// <summary>BCrypt/Argon2 hash. NEVER plain text.</summary>
        public string PasswordHash { get; set; } = string.Empty;

        /// <summary>FK to roles.code.</summary>
        public string RoleCode { get; set; } = string.Empty;

        /// <summary>True = every camera, incl. ones added later. Ignored for SUPER_ADMIN (always all).</summary>
        public bool AllCameras { get; set; }

        /// <summary>False = blocked from logging in.</summary>
        public bool IsActive { get; set; } = true;

        public bool MustChangePassword { get; set; } = true;

        public int FailedLoginCount { get; set; }

        /// <summary>UTC; set after too many failed logins.</summary>
        public DateTime? LockedUntil { get; set; }

        /// <summary>UTC.</summary>
        public DateTime? LastLoginAt { get; set; }

        /// <summary>app_users.id of the creator; null = bootstrap.</summary>
        public int? CreatedBy { get; set; }

        /// <summary>UTC.</summary>
        public DateTime CreatedAt { get; set; }

        /// <summary>UTC.</summary>
        public DateTime UpdatedAt { get; set; }
    }
}
