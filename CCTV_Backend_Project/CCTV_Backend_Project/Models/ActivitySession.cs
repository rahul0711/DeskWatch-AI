namespace CCTV_Backend_Project.Models
{
    /// <summary>Table: activity_sessions. Written by the Python camera server.</summary>
    public class ActivitySession
    {
        public long Id { get; set; }

        /// <summary>Employee id (users.employee_id); null = unidentified.</summary>
        public string? UserId { get; set; }

        public string? UserName { get; set; }

        /// <summary>camera-01 | inside | corridor.</summary>
        public string CameraId { get; set; } = string.Empty;

        /// <summary>See activity_types.code.</summary>
        public string ActivityType { get; set; } = string.Empty;

        /// <summary>UTC.</summary>
        public DateTime StartTime { get; set; }

        /// <summary>UTC.</summary>
        public DateTime EndTime { get; set; }

        /// <summary>Seconds.</summary>
        public double Duration { get; set; }

        /// <summary>Share of detection passes that agreed (0-1).</summary>
        public double Confidence { get; set; }

        /// <summary>Tracker ids that contributed (debug only).</summary>
        public string? TrackIds { get; set; }

        /// <summary>UTC, when the row was written.</summary>
        public DateTime CreatedAt { get; set; }
    }
}
