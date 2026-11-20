namespace CCTV_Backend_Project.Models
{
    /// <summary>Table: audit_log.</summary>
    public class AuditLog
    {
        public long Id { get; set; }

        /// <summary>UTC.</summary>
        public DateTime At { get; set; }

        /// <summary>Who did it; null = system.</summary>
        public int? AppUserId { get; set; }

        /// <summary>Copied, so it survives the login being deleted.</summary>
        public string? Username { get; set; }

        /// <summary>e.g. login, login_failed, person.camera_access, login.create.</summary>
        public string Action { get; set; } = string.Empty;

        /// <summary>person | login | camera | settings.</summary>
        public string? TargetType { get; set; }

        public string? TargetId { get; set; }

        public string? CameraId { get; set; }

        /// <summary>JSON with before/after values. Never passwords or tokens.</summary>
        public string? Details { get; set; }

        public string? IpAddress { get; set; }
    }
}
