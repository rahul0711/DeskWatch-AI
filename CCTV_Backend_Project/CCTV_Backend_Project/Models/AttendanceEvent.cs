namespace CCTV_Backend_Project.Models
{
    /// <summary>Table: attendance_events.</summary>
    public class AttendanceEvent
    {
        public int Id { get; set; }

        /// <summary>FK to users.id (ON DELETE CASCADE).</summary>
        public int UserId { get; set; }

        public string CameraName { get; set; } = string.Empty;

        /// <summary>UTC.</summary>
        public DateTime Timestamp { get; set; }

        /// <summary>Cosine similarity of the match.</summary>
        public double Confidence { get; set; }

        public int FaceWidthPx { get; set; }

        public bool LowConfidence { get; set; }

        public string ThumbnailPath { get; set; } = string.Empty;
    }
}
