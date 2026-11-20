namespace CCTV_Backend_Project.Models
{
    /// <summary>Table: camera_restrictions. A row means the person is BLOCKED on that camera.</summary>
    public class CameraRestriction
    {
        public int Id { get; set; }

        /// <summary>FK to users.id (ON DELETE CASCADE). Unique together with CameraId.</summary>
        public int UserId { get; set; }

        /// <summary>camera-01 | inside | corridor.</summary>
        public string CameraId { get; set; } = string.Empty;

        /// <summary>UTC.</summary>
        public DateTime CreatedAt { get; set; }
    }
}
