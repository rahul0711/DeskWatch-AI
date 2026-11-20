namespace CCTV_Backend_Project.Models
{
    /// <summary>Table: activity_events.</summary>
    public class ActivityEvent
    {
        public long Id { get; set; }

        public string CameraId { get; set; } = string.Empty;

        public int TrackId { get; set; }

        public string? PersonId { get; set; }

        public string? PersonName { get; set; }

        public string ActivityType { get; set; } = string.Empty;

        public DateTime StartTime { get; set; }

        public DateTime EndTime { get; set; }

        public double DurationSeconds { get; set; }

        public double Confidence { get; set; }

        public DateTime CreatedAt { get; set; }
    }
}
